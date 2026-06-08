# SWRC — spin-wave reservoir computer

A micromagnetic (mumax³) simulator for a **spin-wave-disk reservoir computer**:
a 1 µm circular magnetic disk driven by RF pulses through actuator regions, used
as the nonlinear feature map of a physical reservoir computer. This repository
contains the simulator, a minimal worked example that runs end-to-end on a
single GPU, a tutorial walkthrough, and the analyses that back
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
├── datasets/                ← input-dataset generators
│   └── water_wave_obstacle/      ← balanced random distance-reading generators (2/3/5-class)
├── simulator/               ← THE simulator — mumax³ code + sweep driver + cube assembler
│   ├── README.md
│   ├── GEOMETRY.md          ← canonical parameter table (incl. VCMA convention)
│   ├── run_sweep.py
│   ├── build_relax_cache.mx3
│   ├── test_sample.mx3
│   └── build_cube.py
├── examples/                ← runnable end-to-end examples
│   └── minimal_classification/   ← 2x2 XOR, 4 samples, ~5 min on one GPU
├── tutorial/                ← single-notebook walkthrough of the simulator (~200 lines)
│   ├── README.md
│   └── SWRC_MuMax3.ipynb
└── experiments/             ← downstream analyses backing the manuscript figures
    ├── README.md
    ├── xor_checkerboard/    ← SWR + MLP + CNN on the XOR-checkerboard task (Supplementary Fig. 7)
    └── kernel_rank_cg/      ← kernel rank / d95 / participation ratio vs spatial coarse-graining (Section S9, Fig. S9.2 + Table S9.1)
```

## Start here — the simulator

The simulator lives in [`simulator/`](simulator/). It produces, for any
coordinate-encoded pair of RF input pulses, the time series of the out-of-plane
magnetisation `m_z(t)` of the disk on a coarse-grained spatial grid. The output
of one sweep is a `(N_samples, T, H, W)` float32 `.npy` cube that downstream
analyses load.

Key files:

- [`simulator/run_sweep.py`](simulator/run_sweep.py) — the production multi-GPU sweep driver. Embeds the mumax³ templates, builds the relax cache once, runs all `GRID²` samples (one GPU per worker, `N_GPUS` in parallel), reads each OVF and writes per-sample CSV directories. All parameters (grid, Tmax, T_step, drive, crop) are env-var configurable.
- [`simulator/build_relax_cache.mx3`](simulator/build_relax_cache.mx3) and [`simulator/test_sample.mx3`](simulator/test_sample.mx3) — standalone, human-readable mumax³ sources for the equilibrium relax and for one swept sample. Read these first to understand the substrate physics.
- [`simulator/build_cube.py`](simulator/build_cube.py) — assembles the per-sample CSV run directories into the `(N, T, H, W)` cube the analysis loads.
- [`simulator/GEOMETRY.md`](simulator/GEOMETRY.md) — the canonical parameter table: every constant in the simulator with symbol, value, units, physical meaning, and the **VCMA-coefficient convention** that relates the simulator's dynamic-Ku1 amplitude to an experimental device voltage.

Full physical model and a regenerate-the-cube recipe in
[`simulator/README.md`](simulator/README.md).

If you want a **minimal walkthrough** of the simulator before reading the
production code, open [`tutorial/SWRC_MuMax3.ipynb`](tutorial/SWRC_MuMax3.ipynb).
It implements the same substrate physics in ~200 lines of inline Python.

## Run the minimal example

The fastest way to confirm everything works is to run the bundled minimal
example — a 4-sample 2 × 2 XOR task that goes substrate → cube → classifier in
**~5 minutes on a single modern GPU**:

```bash
cd examples/minimal_classification
bash run_example.sh
# -> sweep_out/, mini_cube.npy, result.json, response_snapshot.png
```

Expected output: leave-one-out balanced accuracy on the 4 samples (typically
1.000), a JSON record of every fold, and a 4-panel `m_z` snapshot. See
[`examples/minimal_classification/README.md`](examples/minimal_classification/README.md)
for prerequisites and a full description of the expected output.

## System requirements

**Tested on:** Ubuntu 22.04 LTS (simulator + analysis compute stages) and
Windows 11 (figure regeneration + supplementary-table `.docx` builder), under
Python 3.13.13 with the dependency versions pinned in
[`requirements.txt`](requirements.txt). The figure-regeneration stage is also
verified under numpy 2.4.2.

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
| GPU | 1 × CUDA-capable (≥RTX 20-series equivalent recommended) | Required **only** for the simulator. The figure / analysis stages run on CPU; without a GPU you can still regenerate every manuscript figure from the bundled `.npz` caches — see *Quick smoke test (no GPU)* below. |

No other non-standard hardware is required.

## Installation guide

### 1. Python environment (~5 minutes on a normal desktop)

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` is grouped by experiment and by pipeline stage; the
compute stages were run on Linux, the figure stages also run on Windows.

### 2. MuMax3 binary (~2 minutes; skip if you only need to regenerate figures)

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

**Typical total install time on a normal desktop computer:** ~7 minutes
(≈5 min Python env over broadband + ≈2 min mumax3 download), assuming a
working CUDA driver.

### 3. Quick smoke test (no GPU, ~5 seconds)

If you do not have a GPU, you can still confirm the install end-to-end by
regenerating Supplementary Figure 7 from the bundled `.npz` caches:

```bash
cd experiments/xor_checkerboard
python make_xor_comparison_figure.py
# -> figures/figS7_xor_ladder_comparison.png
```

This runs entirely on CPU in a few seconds and exercises numpy, scikit-learn,
and matplotlib without invoking mumax3. The GPU-based minimal example in the
next section is the full-pipeline smoke test for users who also want to
exercise the simulator.

## Reproducing the manuscript figures

Every figure in the manuscript regenerates from the bundled `.npz` result
caches without the multi-GB `m_z` cubes; they run in seconds on a laptop.

| Manuscript figure | Command (from repo root) | Output |
|-------------------|--------------------------|--------|
| **Supplementary Fig. 7** (XOR ladder, SWR vs MLP vs CNN) | `cd experiments/xor_checkerboard && python make_xor_comparison_figure.py` | `figures/figS7_xor_ladder_comparison.png` |
| **Section S9.1** (kernel rank, d95, PR vs spatial CG — display panel) | `cd experiments/kernel_rank_cg && python make_fig_s91.py` | `figures/figS91_kr_pr.png` |
| **Fig. S9.2** (per-frame spatial rank) | `cd experiments/kernel_rank_cg && python make_fig_spatial.py` | `figures/figS92_spatial_rank.png` |
| Supporting analysis figs 1–5 (rank, SV spectra, cum. variance, multi-metric rank, CG montage) | `cd experiments/kernel_rank_cg && python make_figures.py` | `figures/fig{1..5}_*.png` |
| **Table S9.1** / **Supplementary Table 2** (.docx, Windows + MS Office only) | `cd experiments/kernel_rank_cg && python build_docx_s91.py` | `SI_S9_1_Kernel_Rank.docx` |

End-to-end regeneration *including* the simulator step (for one experiment at a
time) follows the recipe in each experiment's `README.md`. The simulator stage
requires mumax3 + a GPU and takes hours on multiple GPUs for the production
sweeps; see [`simulator/README.md`](simulator/README.md).

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

## Experiments

The two subpackages in [`experiments/`](experiments/) are figure-reproducibility
bundles — they sit on top of the simulator. Each folder is self-contained:
bundled `.npz` caches, scripts to regenerate the figures, and a README
documenting the analysis.

### [`experiments/xor_checkerboard/`](experiments/xor_checkerboard/) — SWR + MLP + CNN on the XOR checkerboard
Compares three classifiers on a ladder of XOR-checkerboard tasks of increasing
spatial frequency (period `P = 16…1` px, 32×32 grid): a **spin-wave-disk
reservoir** read out by a top-`K` mutual-information `LinearSVC`, an
**MLP-on-coords**, and a **CNN-on-coords** — on a shared
balanced-accuracy-vs-FLOPs axis with identical 15-fold cross-validation. Shows
the substrate performs a nonlinear feature lift that solves fine-scale parity
where the feed-forward baselines fail at any swept compute budget. Generates
**Supplementary Fig. 7**.

### [`experiments/kernel_rank_cg/`](experiments/kernel_rank_cg/) — kernel rank, d95, participation ratio vs spatial coarse-graining
Computes the numerical rank, d95 (PCs for 95 % variance), and participation
ratio of the spin-wave reservoir feature matrix Ψc as the `m_z` field is
spatially coarse-grained from 50×50 down to 4×4. Generates **Supplementary
Table 2 / Table S9.1** and **Fig. S9.2** (plus a display panel for the
narrative discussion in Section S9.1).

## Datasets

### [`datasets/water_wave_obstacle/`](datasets/water_wave_obstacle/) — water-wave reservoir obstacle-distance generators
Balanced random-dataset generators for the water-wave reservoir's
obstacle-classification task (Fig. 2e and Fig. 5b in the main text). Three
variants — 2-, 3- and 5-class — share the same six-sensor encoding: each
sample is a 6-tuple of integer distance readings, mapped (in the physical
experiment) to the wave initiation times that drive the six actuators. See
the folder README for the encoding, class definitions, and usage.

## Code description / pseudocode

A formal, language-agnostic specification of the four pipeline stages — the
substrate simulation sweep, the cube assembler, the top-`K` mutual-information
+ LinearSVC readout, and the kernel-rank / d95 / participation-ratio analysis —
is in [`PSEUDOCODE.md`](PSEUDOCODE.md). Each algorithm block is paired with a
pointer to the canonical Python/mumax³ implementation in this repository.

## License

This software is released under the MIT License — see [`LICENSE`](LICENSE) for
the full text. The MIT License is an
[Open Source Initiative–approved](https://opensource.org/licenses/MIT)
permissive licence: users may use, copy, modify, redistribute, and incorporate
the code in derivative works (including closed-source ones), subject only to
preservation of the copyright notice and the licence text.

---
