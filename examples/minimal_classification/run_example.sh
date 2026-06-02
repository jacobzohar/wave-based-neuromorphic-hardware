#!/bin/bash
# Minimal SWRC end-to-end example. See README.md for the full description.
#
# Pipeline:
#   1. simulator/run_sweep.py  (mumax3) -> 4-sample 64x64 sweep at Tmax=2 ns
#   2. simulator/build_cube.py            -> (4, 101, 64, 64) float32 cube
#   3. train_minimal.py                   -> top-K MI + LinearSVC + LOO-CV
#
# Assumes a working `mumax3` binary on PATH (or set MUMAX3_PATH). One GPU.
# Tunable knobs are all env vars; defaults give a 2 x 2 XOR in ~5 min on
# a modern GPU.

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"

# Sweep parameters: 2 x 2 = 4 samples, Tmax = 2 ns, one GPU.
# T_step = 0.2 ns gives well-separated pulse-centre times within the window.
export SWRC_OUTPUT_DIR="$HERE/sweep_out"
export SWRC_GRID=2
export SWRC_TMAX=2e-9
export SWRC_TSTEP=2e-10
export SWRC_N_GPUS=1

mkdir -p "$SWRC_OUTPUT_DIR"

echo "=========================================================="
echo "Step 1/3: mumax3 sweep (4 samples, ~2-3 min on one GPU)"
echo "=========================================================="
python "$REPO_ROOT/simulator/run_sweep.py"

echo ""
echo "=========================================================="
echo "Step 2/3: build m_z cube from sweep dirs (~10 s)"
echo "=========================================================="
# Tmax = 2 ns, save every 20 ps -> 100 frames (mumax3 emits N+1: t=0..Tmax).
# --out is resolved as an absolute path; run_index.json lands beside the cube.
python "$REPO_ROOT/simulator/build_cube.py" \
    --src "$SWRC_OUTPUT_DIR" \
    --out "$HERE/mini_cube.npy" \
    --frames 101 \
    --n 4 \
    --side 64

echo ""
echo "=========================================================="
echo "Step 3/3: classifier + figure (~30 s)"
echo "=========================================================="
python "$HERE/train_minimal.py" --cube "$HERE/mini_cube.npy" --out_dir "$HERE"

echo ""
echo "=========================================================="
echo "Done. See result.json + response_snapshot.png in $HERE"
echo "=========================================================="
