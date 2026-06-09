#!/bin/bash
# Driver: pixel-MI (XOR) + top-K LinearSVC sweep for all 5 drives at 16x16.
# Runs sequentially per drive: MI cube → top-K SVM sweep. ~30 min/drive expected.
set -e
cd "$(dirname "$0")"  # run from this folder
PY=${PY:-python}     # override with: PY=/path/to/.venv/bin/python ./run_sw_reservoir_sweep.sh
CUBE_DIR=${CUBE_DIR:?Set CUBE_DIR to the folder holding nlt4_*_mz_cube.npy}

drives=(
  "45mT     nlt4_mz_cube.npy"
  "15mT200  nlt4_15mT200_mz_cube.npy"
  "15mT1100 nlt4_15mT1100_mz_cube.npy"
  "30mT500  nlt4_30mT500_mz_cube.npy"
  "60mT     nlt4_60mT_mz_cube.npy"
)

mkdir -p data
echo "=== Top-K SVM + pixel-MI sweep started $(date) ==="
for entry in "${drives[@]}"; do
  drive=$(echo "$entry" | awk '{print $1}')
  cube=$(echo "$entry" | awk '{print $2}')
  echo
  echo "--- drive=$drive  cube=$cube ---"

  # Build XOR pixel-MI cube (unless already there)
  mi_file="data/pixel_mi_xor_cube_16_${drive}.npy"
  if [ ! -f "$mi_file" ]; then
    $PY build_xor_pixel_mi.py \
        --cube  "$CUBE_DIR/$cube" \
        --grid  16 \
        --drive "$drive" \
        --xor_ds xor_datasets.npz \
        --out_dir data
  else
    echo "[skip MI] $mi_file already exists"
  fi

  # Top-K SVM sweep with analog/binary/ternary readouts
  $PY top_k_svm_sweep.py \
      --cube  "$CUBE_DIR/$cube" \
      --mi_cube "$mi_file" \
      --xor_ds xor_datasets.npz \
      --grid  16 \
      --drive "$drive" \
      --out   "data/topk_svm_16_${drive}.npz"
done
echo
echo "=== Sweep finished $(date) ==="
ls -lh data/
