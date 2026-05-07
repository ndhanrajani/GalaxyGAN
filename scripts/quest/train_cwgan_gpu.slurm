#!/bin/bash
# GalaxyGAN training on Northwestern Quest (GPU batch job).
#
# BEFORE FIRST RUN:
#   sbatch scripts/quest/train_cwgan_gpu.slurm

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

REPO_ROOT="${REPO_ROOT:-/projects/e32706/ahh2048/GalaxyGAN}"

export GALAXY10_H5="${GALAXY10_H5:-$REPO_ROOT/data/Galaxy10_DECals.h5}"

module purge
if [[ -n "${ACTIVATE_CMD:-}" ]]; then
  eval "$ACTIVATE_CMD"
elif [[ -f "$REPO_ROOT/scripts/quest/activate_env.sh" ]]; then
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

_LOCAL_TMP="${SLURM_TMPDIR:-${TMPDIR:-}}"
if [[ -z "${SKIP_H5_STAGE:-}" && -n "$_LOCAL_TMP" && -d "$_LOCAL_TMP" ]]; then
  _STAGED="$_LOCAL_TMP/Galaxy10_DECals_job${SLURM_JOB_ID:-$$}.h5"
  echo "Staging HDF5 to node-local temp: $_STAGED"
  cp -f "$GALAXY10_H5" "$_STAGED"
  export GALAXY10_H5="$_STAGED"
  echo "GALAXY10_H5 -> $GALAXY10_H5"
fi

export PYTHONUNBUFFERED=1

echo "Train settings: EPOCHS=${EPOCHS:-50} N_CRITIC=${N_CRITIC:-} LR=${LR:-} RESUME=${RESUME:-} OUT=$OUT"

OUT="${OUT:-$REPO_ROOT/runs/cwgan_gp_${SLURM_JOB_ID:-local}}"
extra=()
if [[ -n "${RESUME:-}" ]]; then
  extra+=(--resume "$RESUME")
fi
ncargs=()
if [[ -n "${N_CRITIC:-}" ]]; then
  ncargs+=(--n-critic "$N_CRITIC")
fi
nwargs=()
if [[ -n "${NUM_WORKERS:-}" ]]; then
  nwargs+=(--num-workers "$NUM_WORKERS")
fi
lrargs=()
if [[ -n "${LR:-}" ]]; then
  lrargs+=(--lr "$LR")
fi
python3 -m galaxygan.train \
  --epochs "${EPOCHS:-50}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --preload \
  --out "$OUT" \
  "${ncargs[@]}" \
  "${nwargs[@]}" \
  "${lrargs[@]}" \
  "${extra[@]}"

echo "Done. Outputs under $OUT"
