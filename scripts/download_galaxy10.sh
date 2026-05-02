#!/usr/bin/env bash
# Downloads Galaxy10 DECaLS (~2.54 GB) from Zenodo into ./data/
# SHA256 (official): 19AEFC477C41BB7F77FF07599A6B82A038DC042F889A111B0D4D98BB755C1571

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/data/Galaxy10_DECals.h5"
URL="https://zenodo.org/records/10845026/files/Galaxy10_DECals.h5"

mkdir -p "${ROOT}/data"
if [[ -f "${DEST}" ]]; then
  echo "Already exists: ${DEST}"
  exit 0
fi

echo "Downloading to ${DEST} ..."
curl -L --fail --retry 3 --retry-delay 5 -o "${DEST}.partial" "${URL}"
mv "${DEST}.partial" "${DEST}"
echo "Done."
