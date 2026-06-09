# SWRC — spin-wave reservoir computer

A micromagnetic (mumax³) simulator for a **spin-wave-disk reservoir computer**:
a 1 µm circular magnetic disk driven by RF pulses through actuator regions, used
as the nonlinear feature map of a physical reservoir computer. This repository
contains the simulator, a minimal worked example that runs end-to-end on a
single GPU, a tutorial walkthrough, the input-data generators for the
water-wave reservoir tasks, and the analyses that back
**"Autonomous robotic operation controlled by wave-based neuromorphic hardware"**
(*Nature Communications*; Jacob Zohar, Daniele Pinna, Gerrit van der Laan,
Thorsten Hesjedal, C. K. Safeer).

## Layout

```
SWRC/
├── README.md                ← this file
├── LICENSE                  ← MIT license
├── PSEUDOCODE.md            ← formal algorithm specification (the four pipeline stages)
├── requirements.txt         ← pinned Python dependencies
├── simulator/               ← THE simulator — mumax³ code + sweep driver + cube assembler
│   ├── README.md
│   ├── GEOMETRY.md          ← canonical parameter table (incl. VCMA convention)
│   ├── run_sweep.py
│   ├── build_relax_cache.mx3
│   ├── test_sample.mx3
│   └── build_cube.py
├── examples/                ← runnable end-to-end examples
│   └── minimal_classification/                  ← 2x2 XOR, 4 samples, ~5 min on one GPU
├── tutorial/                ← single-notebook walkthrough of the simulator (~200 lines)
│   ├── README.md
│   └── SWRC_MuMax3.ipynb
└── Demo/                    ← runnable demonstrations of the software (one per manuscript figure)
    ├── README.md
    ├── 2_3_5_Robotic_output_classification_task/      ← full workflow: dataset → simulator → SVM (Fig. 2e)
    └── reservoir_metrics/                              ← intrinsic-substrate characterisations
        ├── README.md
        ├── xor_chequerboard/                              ← SWR + MLP + CNN on the XOR-chequerboard task (Supplementary Fig. 7)
        └── kernel_rank_cg/                                ← kernel rank / d95 / participation ratio vs spatial coarse-graining (Section S9, Fig. S9.2 + Table S9.1)
```

## System requirements

### Software dependencies and operating systems (including version numbers)

| Component | Version | Notes |
|-----------|---------|-------|
| Python | ≥3.10 (tested on 3.13.13) | |
| MuMax3 | v3.10 (or compatible) | <https://mumax.github.io/>. Required for any simulator step; not needed to regenerate the figures from the bundled `.npz` caches. Install instructions below. |
| numpy | 2.2.6 | Figure stage also verified under numpy 2.4.2 |
| opencv-python | 4.12.0 | Used by the kernel-rank analysis |
| scikit-learn | 1.8.0 | LinearSVC + cross-validation |
| torch | 2.11.0 | CPU build sufficient; published MLP/CNN runs were CPU |
| matplotlib | 3.10.8 | Figure rendering |
| pandas | (any recent) | Faster CSV reads in `build_cube.py` |

Operating systems used to produce the published results: **Ubuntu 22.04 LTS**
(simulator + analysis compute stages) and **Windows 11** (figure regeneration
and supplementary-table `.docx` builder). All pinned dependency versions are
listed in [`requirements.txt`](requirements.txt).

### Versions the software has been tested on

- Ubuntu 22.04 LTS, Python 3.13.13, with the `requirements.txt` versions above.
- Windows 11, Python 3.13.13, with the same `requirements.txt` versions; the
  figure-regeneration stage is also verified under numpy 2.4.2.

### Required non-standard hardware

- One CUDA-capable GPU (≥ RTX 20-series equivalent recommended) is required to
  run the simulator stage. The figure / analysis stages run on CPU; the demo
  also includes a no-GPU path (see *Demo* below) that exercises the install
  end-to-end without a GPU.
- No other non-standard hardware is required.

## Installation guide

### Instructions

#### 1. Python environment

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is grouped by experiment and by pipeline stage; the
compute stages were run on Linux, the figure stages also run on Windows.

#### 2. MuMax3 binary

MuMax3 is required only to run the simulator (Algorithm 1 in
[`PSEUDOCODE.md`](PSEUDOCODE.md)). To install:

1. Download a v3.10 build for your platform from <https://mumax.github.io/>
   (Linux: `mumax3.10_linux_cuda12.0.tar.gz`; Windows:
   `mumax3.10_windows_cuda12.0.zip`).
2. Extract the archive and either:
   - place the `mumax3` (Linux) / `mumax3.exe` (Windows) binary on your
     `PATH`, **or**
   - set the `MUMAX3_PATH` environment variable to the absolute path of the
     binary.
3. Verify with `mumax3 -test` (Linux) or `mumax3.exe -test` (Windows); the
   binary will print its build info and the CUDA device it sees.

### Typical install time on a "normal" desktop computer

**~7 minutes total**, broken down as:

- ~5 min for the Python environment (`pip install -r requirements.txt` on a
  normal broadband connection).
- ~2 min for the mumax3 binary download.

This assumes a working CUDA driver is already present. The CUDA driver
installation itself (if not present) is platform-specific and not counted
above; follow NVIDIA's instructions for your platform.

## Demo

### Instructions to run on data

Two demo paths are bundled. Run **either or both** — they exercise different
parts of the pipeline.

**GPU demo** — full end-to-end (substrate → cube → classifier) on a 4-sample
2 × 2 XOR task:

```bash
cd examples/minimal_classification
bash run_example.sh
```

**No-GPU demo** — regenerate Supplementary Fig. 7 from the bundled `.npz`
result caches; runs entirely on CPU and does not invoke mumax3:

```bash
cd Demo/reservoir_metrics/xor_chequerboard
python make_xor_comparison_figure.py
```

### Expected output

**GPU demo** writes the following into `examples/minimal_classification/`:

- `sweep_out/` — 4 mumax3 run directories + the relax cache (~20 MB total).
- `mini_cube.npy` — `(4, 101, 64, 64)` float32 `m_z` cube (~10 MB).
- `result.json` — leave-one-out balanced accuracy and per-fold record. On
  this 4-sample task the balanced accuracy is typically **1.000**
  (4 / 4 folds correct).
- `response_snapshot.png` — 4-panel `m_z` snapshot at the peak-MI frame.

The terminal also prints a one-line summary, e.g.:

```
[result] LOO-CV balanced accuracy = 1.000  (4/4 folds correct)
         top-K MI features used: K = 4  (peak MI = 0.918 bits / 1.000 bit ceiling)
         saved result.json + response_snapshot.png
```

**No-GPU demo** writes a single figure file:

- `Demo/reservoir_metrics/xor_chequerboard/figures/figS7_xor_ladder_comparison.png` —
  the Supplementary Fig. 7 panel (balanced accuracy vs FLOPs for the
  spin-wave reservoir, MLP-on-coords, and CNN-on-coords on the
  XOR-chequerboard task ladder).

### Expected run time for demo on a "normal" desktop computer

- **GPU demo:** ~5 minutes on a single RTX 3070 / RTX 5070-class GPU (relax
  cache ~30 s, 4-sample sweep ~2 min, cube build ~10 s, classifier +
  figure ~30 s). Older GPUs scale roughly with mumax3 throughput.
- **No-GPU demo:** ~5 seconds on a normal laptop CPU.

## Instructions for use

### How to run the software on your data

The pipeline has two stages: the simulator that produces an `m_z` cube from
a coordinate-encoded input grid, and the analysis stage that reads the cube
and produces a classification or dimensionality result.

**1. Generate an `m_z` cube from your own input grid.** The simulator stage
is configured entirely via environment variables (see the file header of
[`simulator/run_sweep.py`](simulator/run_sweep.py) for the full list and
defaults):

```bash
cd simulator
SWRC_GRID=16 SWRC_TMAX=4e-9 SWRC_N_GPUS=1 python run_sweep.py
python build_cube.py --src sweep_out --out my_cube.npy --frames 201
```

Output is a `(N_samples, T, H, W)` float32 `.npy` cube plus a
`run_index.json` that maps cube rows to `(X1, X2)` input coordinates. The
full physical model and parameter reference is in
[`simulator/GEOMETRY.md`](simulator/GEOMETRY.md); a minimal walkthrough of
the substrate physics in inline Python is in
[`tutorial/SWRC_MuMax3.ipynb`](tutorial/SWRC_MuMax3.ipynb).

**2. Run a demonstration on your cube.** Each demo in `Demo/` exposes a
`--cube` CLI argument or `CUBE_PATH` environment variable to redirect the
analysis at your own cube. The three bundled demos:

- [`Demo/2_3_5_Robotic_output_classification_task/`](Demo/2_3_5_Robotic_output_classification_task/) —
  full obstacle-classification workflow: dataset generation (three
  notebooks for the 2-, 3- and 5-class variants), instructions for
  feeding the dataset to the simulator, expected simulator output, and
  a linear-SVM training + testing script
  (`sklearn.svm.SVC(kernel='linear')`) that produces the Fig. 2e
  comparison panels (`FinalResultsMain`, `SWvsSVMW`, `WWvsSVMvsSW`).
- [`Demo/reservoir_metrics/xor_chequerboard/`](Demo/reservoir_metrics/xor_chequerboard/) —
  SWR + MLP + CNN on an XOR-chequerboard task ladder. Generates
  Supplementary Fig. 7.
- [`Demo/reservoir_metrics/kernel_rank_cg/`](Demo/reservoir_metrics/kernel_rank_cg/) —
  kernel rank, d95, and participation ratio of the SWR feature matrix
  vs spatial coarse-graining. Generates Supplementary Table 2 / Table
  S9.1 and Fig. S9.2.

See each demo's `README.md` for the exact CLI / env-var knobs, the
bundled inputs, and the data shape it expects.

### Reproduction instructions

Every figure in the manuscript regenerates from the bundled `.npz` result
caches without the multi-GB `m_z` cubes; they run in seconds on a laptop.

| Manuscript figure | Command (from repo root) | Output |
|-------------------|--------------------------|--------|
| **Fig. 2e** (WW vs Raw SVM vs SWRC training-size sweep) | `cd Demo/2_3_5_Robotic_output_classification_task && python train_test_svm.py` | `figures/FinalResultsMain.png`, `figures/SWvsSVMW.png`, `figures/WWvsSVMvsSW.png` |
| **Supplementary Fig. 7** (XOR ladder, SWR vs MLP vs CNN) | `cd Demo/reservoir_metrics/xor_chequerboard && python make_xor_comparison_figure.py` | `figures/figS7_xor_ladder_comparison.png` |
| **Section S9.1** (kernel rank, d95, PR vs spatial CG — display panel) | `cd Demo/reservoir_metrics/kernel_rank_cg && python make_fig_s91.py` | `figures/figS91_kr_pr.png` |
| **Fig. S9.2** (per-frame spatial rank) | `cd Demo/reservoir_metrics/kernel_rank_cg && python make_fig_spatial.py` | `figures/figS92_spatial_rank.png` |
| Supporting analysis figs 1–5 (rank, SV spectra, cum. variance, multi-metric rank, CG montage) | `cd Demo/reservoir_metrics/kernel_rank_cg && python make_figures.py` | `figures/fig{1..5}_*.png` |
| **Table S9.1** / **Supplementary Table 2** (.docx, Windows + MS Office only) | `cd Demo/reservoir_metrics/kernel_rank_cg && python build_docx_s91.py` | `SI_S9_1_Kernel_Rank.docx` |

End-to-end regeneration *including* the simulator step (for one demo at a
time) follows the recipe in each demo's `README.md`. The simulator stage
requires mumax3 + a GPU and takes hours on multiple GPUs for the
production sweeps; see [`simulator/README.md`](simulator/README.md).

## Data availability

The large spin-wave (`m_z`) micromagnetic data cubes the experiments consume
are **not** bundled here (each is multiple GB). They are reproducible from the
simulator — every cube was generated by [`simulator/run_sweep.py`](simulator/run_sweep.py)
+ [`simulator/build_cube.py`](simulator/build_cube.py) with the parameters
documented in [`simulator/GEOMETRY.md`](simulator/GEOMETRY.md). The
precomputed `.npz` result caches **are** bundled in each experiment folder,
so every published figure regenerates without the raw cubes. The cubes
themselves are available from the corresponding author
(safeer.chenattukuzhiyil@physics.ox.ac.uk) on reasonable request.

## Code description / pseudocode

A formal specification of the four pipeline stages — the substrate
simulation sweep, the cube assembler, the top-`K` mutual-information +
LinearSVC readout, and the kernel-rank / d95 / participation-ratio
analysis — is in [`PSEUDOCODE.md`](PSEUDOCODE.md). Each algorithm block is
paired with a pointer to the canonical Python / mumax³ implementation in
this repository.

## License

This software is released under the MIT License — see [`LICENSE`](LICENSE) for
the full text. The MIT License is an
[Open Source Initiative–approved](https://opensource.org/licenses/MIT)
permissive licence: users may use, copy, modify, redistribute, and incorporate
the code in derivative works (including closed-source ones), subject only to
preservation of the copyright notice and the licence text.

---
