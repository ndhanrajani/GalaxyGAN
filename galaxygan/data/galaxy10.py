"""
Galaxy10 DECaLS (Zenodo / astroNN).

The published ``Galaxy10_DECals.h5`` stores RGB images at 256×256. For 69×69
training (lighter GPU load), set ``target_size=69`` — images are resized on the fly.

Get the file either:
  - Run ``bash scripts/download_galaxy10.sh`` (curl from Zenodo), or
  - ``pip install astroNN`` and use ``astroNN.datasets.load_galaxy10()`` once (downloads
    to ``~/.astroNN/datasets/``), then point ``GALAXY10_H5`` at that path.

Env: ``GALAXY10_H5`` — path to ``Galaxy10_DECals.h5`` (default: ``./data/Galaxy10_DECals.h5``).
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


# Names from https://astronn.readthedocs.io/en/latest/galaxy10.html
CLASS_NAMES: tuple[str, ...] = (
    "Disturbed",
    "Merging",
    "Round smooth",
    "In-between round smooth",
    "Cigar-shaped smooth",
    "Barred spiral",
    "Unbarred tight spiral",
    "Unbarred loose spiral",
    "Edge-on without bulge",
    "Edge-on with bulge",
)


def default_h5_path() -> Path:
    env = os.environ.get("GALAXY10_H5")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "data" / "Galaxy10_DECals.h5"


class Galaxy10DECaLS(Dataset):
    """
    PyTorch ``Dataset`` reading ``images`` and ``ans`` from the official HDF5 file.

    Returns ``image`` in CHW float32 in ``[-1, 1]`` and ``label`` as int64 in ``[0, 9]``.

    Set ``preload=True`` to load the split into RAM once (sorted HDF5 reads). Use on slow NFS
    (e.g. cluster ``/projects``); uses on the order of ~1–2 GB RAM for the train split at 69².
    """

    def __init__(
        self,
        h5_path: str | Path | None = None,
        *,
        split: str = "train",
        train_fraction: float = 0.9,
        target_size: int | None = 69,
        random_state: int = 42,
        preload: bool = False,
    ) -> None:
        if split not in ("train", "test"):
            raise ValueError("split must be 'train' or 'test'")

        path = Path(h5_path) if h5_path is not None else default_h5_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"Galaxy10 HDF5 not found at {path}. "
                "Run `bash scripts/download_galaxy10.sh` or set GALAXY10_H5."
            )

        self._h5_path = path
        self._file = h5py.File(path, "r")
        self._images = self._file["images"]
        labels = np.asarray(self._file["ans"], dtype=np.int64)
        if labels.ndim != 1 or labels.shape[0] != self._images.shape[0]:
            raise ValueError("Unexpected labels shape in HDF5")

        idx = np.arange(labels.shape[0])
        train_idx, test_idx = train_test_split(
            idx, train_size=train_fraction, random_state=random_state, stratify=labels
        )
        self._indices = train_idx if split == "train" else test_idx
        self._labels = labels
        self.target_size = target_size
        self._preload_images: Tensor | None = None
        self._preload_labels: Tensor | None = None

        if preload:
            self._preload_into_ram()

    def _preload_into_ram(self) -> None:
        """Load all split samples into RAM once (sorted HDF5 index → fewer seeks on NFS)."""
        n = int(self._indices.shape[0])
        images = self._images
        labels_arr = self._labels
        target_size = self.target_size
        slots: list[Tensor | None] = [None] * n
        lab_slots = [0] * n

        order = sorted(range(n), key=lambda i: int(self._indices[i]))
        for i in order:
            j = int(self._indices[i])
            x = np.asarray(images[j])
            if x.dtype != np.float32:
                x = x.astype(np.float32) / 255.0
            else:
                x = np.clip(x, 0.0, 1.0)
            t = torch.from_numpy(x).permute(2, 0, 1)
            if target_size is not None and t.shape[-1] != target_size:
                t = TF.resize(t, [target_size, target_size], antialias=True)
            t = t * 2.0 - 1.0
            slots[i] = t
            lab_slots[i] = int(labels_arr[j])

        self._preload_images = torch.stack(slots, dim=0)
        self._preload_labels = torch.tensor(lab_slots, dtype=torch.long)
        if self._file is not None:
            self._file.close()
            self._file = None

    def close(self) -> None:
        if self._file is not None:
            self._file.close()

    def __len__(self) -> int:
        return int(self._indices.shape[0])

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        if self._preload_images is not None:
            return self._preload_images[i], self._preload_labels[i]

        j = int(self._indices[i])
        # h5py slice: (H, W, C) uint8 or float depending on file
        x = np.asarray(self._images[j])
        if x.dtype != np.float32:
            x = x.astype(np.float32) / 255.0
        else:
            x = np.clip(x, 0.0, 1.0)

        t = torch.from_numpy(x).permute(2, 0, 1)  # CHW
        if self.target_size is not None and t.shape[-1] != self.target_size:
            t = TF.resize(t, [self.target_size, self.target_size], antialias=True)
        t = t * 2.0 - 1.0
        y = torch.tensor(self._labels[j], dtype=torch.long)
        return t, y


if __name__ == "__main__":
    p = default_h5_path()
    print(f"Expected HDF5 path: {p}")
    if p.is_file():
        ds = Galaxy10DECaLS(split="train", target_size=69)
        img, lab = ds[0]
        print("ok:", img.shape, img.dtype, img.min().item(), img.max().item(), "label", lab.item())
        ds.close()
    else:
        print("File missing — run scripts/download_galaxy10.sh")
