# Minimal worked example — 2-class XOR end-to-end

A self-contained, **~5 minute end-to-end** demonstration of the SWRC pipeline:
substrate simulation → m_z cube → mutual-information feature ranking →
LinearSVC classifier → cross-validated accuracy + diagnostic figure.

The classification task is the smallest non-trivial XOR: 4 input coordinate
pairs `(X1, X2) ∈ {(0,0), (0,1), (1,0), (1,1)}` with `XOR` labels
`{0, 1, 1, 0}` (balanced 50/50). Each pair is encoded as a pair of
Gaussian-windowed 800 MHz pulses, fed through the spin-wave-disk reservoir,
read out as the time series of the disk's m_z field, and classified by an
SVM on top-K mutual-information features.

This example exercises **the same** [`../../simulator/run_sweep.py`](../../simulator/run_sweep.py)
and [`../../simulator/build_cube.py`](../../simulator/build_cube.py) used for
the production figures — only the sweep grid (2 × 2 = 4 samples), simulation
window (`Tmax = 2 ns`), and GPU count (1) are overridden.

## Requirements

- A working `mumax3` binary (v3.10 or compatible) on `PATH` or via the
  `MUMAX3_PATH` env var. <https://mumax.github.io/>
- One CUDA-capable GPU (modern, ≥RTX 20-series equivalent).
- Python ≥3.10 with `numpy`, `scikit-learn`, `matplotlib`, `pandas`
  (see top-level [`../../requirements.txt`](../../requirements.txt)).

## Run

```bash
# from this folder
bash run_example.sh
```

On a single RTX 3070 / 5070 class GPU this completes end-to-end in **~5
minutes** (relax cache: ~30 s, 4-sample sweep: ~2 min, cube build: ~10 s,
classifier + figure: ~30 s). Older GPUs will be slower in proportion to
their mumax3 throughput; if the sweep step approaches the 10-minute budget,
either shrink `Tmax` further (see `run_example.sh` for the env-var knob) or
run the sweep step on a faster machine and rerun the analysis locally.

## Expected output

After the run completes, this folder will contain:

```
examples/minimal_classification/
├── sweep_out/                # 4 mumax3 run dirs + relax cache (~20 MB)
│   ├── relax_sw_disk_512.ovf
│   ├── sample_X10X20.out/
│   ├── sample_X10X21.out/
│   ├── sample_X11X20.out/
│   └── sample_X11X21.out/
├── mini_cube.npy             # (4, 101, 64, 64) float32 m_z cube, ~10 MB
├── result.json               # accuracy + per-fold record
└── response_snapshot.png     # m_z snapshot for each of the 4 input pairs
```

`train_minimal.py` prints a one-line summary to stdout, e.g.:

```
[result] LOO-CV balanced accuracy = 1.000  (4/4 folds correct)
         top-K MI features used: K = 4  (peak MI = 0.918 bits / 1.000 bit ceiling)
         saved result.json + response_snapshot.png
```

(Exact numbers will vary slightly across mumax3 builds and seeds; the
classifier should reach perfect accuracy on this 4-sample problem as long
as the substrate dynamics span the encoding window.)

## What this example does NOT show

This is a *minimal* example to demonstrate end-to-end pipeline correctness,
not a meaningful benchmark. The 4-sample LOO-CV result has no statistical
power; for the production-scale results (32×32 = 1024 samples with 5×3 repeated
stratified k-fold CV), see [`../../experiments/xor_checkerboard/`](../../experiments/xor_checkerboard/).

## Files

| File | Role |
|------|------|
| `run_example.sh` | One-command driver: sets env vars, runs `simulator/run_sweep.py` → `simulator/build_cube.py` → `train_minimal.py`. |
| `train_minimal.py` | Loads the cube, computes per-voxel MI on the XOR labels, fits LinearSVC with leave-one-out CV, prints accuracy, saves the figure. |
| `README.md` | This file. |
