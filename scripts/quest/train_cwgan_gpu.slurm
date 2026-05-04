#!/bin/bash
# GalaxyGAN training on Northwestern Quest (GPU batch job).
#
# Docs:
#   - Quest intro: https://rcdsdocs.it.northwestern.edu/start/quest/quest-intro.html
#   - Slurm:       https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/slurm/slurm.html
#   - GPUs:        https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/gpu/gpu.html
#   - Modules:     https://rcdsdocs.it.northwestern.edu/systems/quest/user-guide/modules/modules.html
#
# BEFORE FIRST RUN — edit the lines marked EDIT below, then:
#   sbatch scripts/quest/train_cwgan_gpu.slurm
#
# Put code + venv on /projects/... (durable). Put the HDF5 + run outputs on /projects/... or
# /scratch/... per RCDS storage guidance (scratch is for short-lived / high-I/O work).

#SBATCH --account=e32706
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1                         ## any GPU; use gpu:a100:1 or gpu:h100:1 if you need a type
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=48:00:00                      ## gengpu max 48h (8h hit TIMEOUT before); shorten for smoke tests
#SBATCH --job-name=galaxygan_cwgan
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=NatashaDhanrajani2026@u.northwestern.edu

set -euo pipefail

# Default Quest clone path; override before sbatch if needed: export REPO_ROOT=/path/to/GalaxyGAN
REPO_ROOT="${REPO_ROOT:-/projects/e32706/ahh2048/GalaxyGAN}"

# EDIT: where Galaxy10_DECals.h5 lives (download once; Zenodo ~2.54 GB)
export GALAXY10_H5="${GALAXY10_H5:-$REPO_ROOT/data/Galaxy10_DECals.h5}"

# EDIT: how to activate the Python env that has torch (see RCDS conda/mamba tutorial).
# Example after `ssh` to Quest:
#   module purge && module load mamba/24.3.0 && source "$HOME/mambaforge/etc/profile.d/conda.sh" && mamba activate galaxygan
# Put the same activation in scripts/quest/activate_env.sh (sourced here) or export ACTIVATE_CMD before sbatch.
module purge
if [[ -n "${ACTIVATE_CMD:-}" ]]; then
  eval "$ACTIVATE_CMD"
elif [[ -f "$REPO_ROOT/scripts/quest/activate_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/scripts/quest/activate_env.sh"
else
  echo "ERROR: Set ACTIVATE_CMD or create $REPO_ROOT/scripts/quest/activate_env.sh"
  exit 1
fi

cd "$REPO_ROOT"
echo "PWD=$(pwd)"
echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
python3 - <<'PY'
import sys
import torch

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print(
        "ERROR: CUDA is not available in this job (CPU-only PyTorch or driver/runtime issue).",
        file=sys.stderr,
    )
    sys.exit(1)
PY

if [[ ! -f "$GALAXY10_H5" ]]; then
  echo "ERROR: GALAXY10_H5 not found: $GALAXY10_H5"
  echo "Download with: bash scripts/download_galaxy10.sh (from repo) or place the file and set GALAXY10_H5."
  exit 1
fi

echo "SLURM_TMPDIR=${SLURM_TMPDIR:-<unset>} TMPDIR=${TMPDIR:-<unset>}"

# One copy to node-local disk (when Slurm provides it) so preload + training do not hammer /projects NFS.
# Skip with: export SKIP_H5_STAGE=1
_LOCAL_TMP="${SLURM_TMPDIR:-${TMPDIR:-}}"
if [[ -z "${SKIP_H5_STAGE:-}" && -n "$_LOCAL_TMP" && -d "$_LOCAL_TMP" ]]; then
  _STAGED="$_LOCAL_TMP/Galaxy10_DECals_job${SLURM_JOB_ID:-$$}.h5"
  echo "Staging HDF5 to node-local temp: $_STAGED"
  cp -f "$GALAXY10_H5" "$_STAGED"
  export GALAXY10_H5="$_STAGED"
  echo "GALAXY10_H5 -> $GALAXY10_H5"
fi

export PYTHONUNBUFFERED=1

# Training (add --wandb after `wandb login` on Quest if you use W&B)
# After a TIMEOUT: reuse the same run dir and pass the latest checkpoint, e.g.
#   export OUT=$REPO_ROOT/runs/cwgan_gp_7037508
#   export RESUME=$OUT/checkpoint_epoch0042.pt
#   sbatch scripts/quest/train_cwgan_gpu.slurm
OUT="${OUT:-$REPO_ROOT/runs/cwgan_gp_${SLURM_JOB_ID:-local}}"
extra=()
if [[ -n "${RESUME:-}" ]]; then
  extra+=(--resume "$RESUME")
fi
# Optional: export N_CRITIC=1 for faster smoke tests (default in train.py is 5 for WGAN-GP).
ncargs=()
if [[ -n "${N_CRITIC:-}" ]]; then
  ncargs+=(--n-critic "$N_CRITIC")
fi
# --preload: one slower read from HDF5, then epochs are GPU-bound (important on /projects NFS).
python3 -m galaxygan.train \
  --epochs "${EPOCHS:-50}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --preload \
  --out "$OUT" \
  "${ncargs[@]}" \
  "${extra[@]}"

echo "Done. Outputs under $OUT"
