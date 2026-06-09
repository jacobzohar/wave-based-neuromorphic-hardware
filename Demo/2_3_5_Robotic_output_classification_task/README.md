# 2 / 3 / 5 robotic output classification task — full workflow

End-to-end reproduction of the robotic-vehicle obstacle-distance
classification task that the water-wave reservoir and the spin-wave
reservoir are evaluated on in the manuscript (Fig. 2e). This folder
contains every piece of code and data needed to go from input-dataset
generation, through micromagnetic simulation, to the trained linear-SVM
recognition-rate curves.

The task: a robotic vehicle reads six ultrasonic distance readings (one
per sensor line). Each reading is a small integer in {1, …, 6} or
{1, …, 7} representing a discretised distance band. The reservoir
classifies the six-tuple of readings into one of 2, 3, or 5 output
classes — the three variants are kept separate so the manuscript can
report how the recognition rate scales with task difficulty.

## Folder contents

| File / folder | Role |
|---|---|
| `2_output_unique_data_generation.ipynb` | Step 1 — generate the 2-class dataset |
| `3_output_unique_data_generation.ipynb` | Step 1 — generate the 3-class dataset |
| `5_output_unique_data_generation.ipynb` | Step 1 — generate the 5-class dataset |
| `data/Train{2,3,5}X.csv` | Step 1 output — canonical published datasets (one row per sample, six integer columns) |
| `data/Train{2,3,5}Y.csv` | Step 1 output — labels (one row per sample) |
| `data/swrc_features.npz` | Step 2/3 output — spin-wave-disk reservoir feature cache (`(3, 1000, 2500)` float32, m\_z field at frame 100, z-scored and flattened) |
| `data/training_frame3_results.csv` | Water-wave reservoir lab-measurement baseline scores |
| `train_test_svm.py` | Step 4 — train and test linear SVMs, write the comparison figures |
| `figures/FinalResultsMain.png` | Step 4 output — SWRC recognition rate vs training-set size (with error bars) |
| `figures/SWvsSVMW.png` | Step 4 output — Raw SVM baseline vs SWRC, per output-class count |
| `figures/WWvsSVMvsSW.png` | Step 4 output — three-way comparison (water-wave vs raw SVM vs SWRC) |

The bundled `data/` and `figures/` are sufficient to regenerate the
manuscript figure without running any simulator. Steps 1–3 are only
needed if you want to regenerate the inputs from scratch.

## Workflow

### Step 1 — Generate the dataset

Run any of the three `*_unique_data_generation.ipynb` notebooks. Each
exposes the same two-function interface:

```python
# (cell body of N_output_unique_data_generation.ipynb)
import random

def determine_label(sample):
    ...

def generate_balanced_dataset(n_total):
    ...
    return [data, labels]

DatasTrain = generate_balanced_dataset(1000)
```

The class definitions and distance ranges per variant:

| Variant | Distance values | Classes | Class definition |
|---|---|---|---|
| 2-class | 1–6 | 0 / 1 | label = 1 if any sensor reads in {1, 2, 3}, else 0 |
| 3-class | 1–6 | 0 / 1 / 2 | left-half (sensors 0–2) vs right-half (sensors 3–5) split; side-exclusive |
| 5-class | 1–7 | 0 / 1 / 2 / 3 / 4 | left/right × near (1–2) / mid (3–4); side-exclusive |

The 5-class generator uses an extended distance range (1–7 rather than
1–6) because the near/mid split halves the resolution available per
side; the extra level restores enough dispersion to keep the classes
well separated.

If you set `random.seed(...)` before calling
`generate_balanced_dataset`, the run is deterministic. The bundled
`data/Train{2,3,5}{X,Y}.csv` are the canonical seeds used to produce
the published figures.

### Step 2 — Use the generated dataset in the simulator

In the physical experiment each six-tuple of distance readings drives
six actuators that emit waves at coordinate-encoded initiation times.
The micromagnetic analogue is the spin-wave-disk simulator in
[`../../simulator/`](../../simulator/): the disk is excited by a Ku1
modulation pulse at every input transducer, with the pulse centre time
of transducer *k* set to `distance_reading[k] * T_step + t_off`.

For one sample, the call into the simulator is the same as for the
XOR-chequerboard example, but with `num_inputs = 6` (one input per
sensor reading) instead of the default `num_inputs = 2`. The
parameters used for the manuscript runs (RCT9 sweep, 1000 samples
per variant):

| Parameter | Value |
|---|---|
| `B_ext` (in-plane field) | 45 mT |
| `f` (drive carrier) | 800 MHz |
| `M_s` (saturation magnetisation) | 1 × 10⁶ A/m |
| Disk radius | 500 nm |
| Substrate grid | 512 × 512 × 1 cells (2 nm × 2 nm × 1.5 nm) |
| Number of input transducers | 6 |
| `T_step` (coordinate → time) | 0.05 ns |
| `T_max` | 4 ns |
| Frames saved | every 20 ps (≈ 201 frames per sample) |

See [`../../simulator/GEOMETRY.md`](../../simulator/GEOMETRY.md) for
the full physics parameter table and
[`../../PSEUDOCODE.md`](../../PSEUDOCODE.md) (Algorithm 1) for the
substrate sweep specification.

A full per-variant sweep is `1000 × 4 ns` of simulation time and takes
hours on a single GPU (or minutes on multi-GPU); the bundled
`data/swrc_features.npz` is the pre-extracted readout from that
sweep so the training/testing figures regenerate without rerunning the
simulator.

### Step 3 — Expected output of the simulator

The simulator stage emits, per sample:

- A run directory `RCT9_{N}_X{i}_45mT_800MHz_1e6.out/` containing a
  table of bulk integrals (`table.txt`) and 201 frames of the
  out-of-plane magnetisation `m_z` as 50 × 50 CSV files
  (`m_z000000.csv` … `m_z000200.csv`), each ≈ 30 KB.
- After running [`../../simulator/build_cube.py`](../../simulator/build_cube.py),
  the per-sample CSVs are assembled into an `(N, F, H, W)` float32 cube
  (here `N = 1000`, `F = 201`, `H = W = 50`) plus a `run_index.json`
  mapping cube rows to input-coordinate tuples.

The bundled SWRC feature cache `data/swrc_features.npz` was produced
from that cube by, for each variant and each sample, taking the
`m_z(t = frame 100, ·, ·)` snapshot, z-scoring it across the field
mean and std, and flattening to a 2500-dim feature vector. Frame 100
sits roughly at the substrate's ringdown peak and was the readout
window used in the manuscript.

### Step 4 — Train and test linear SVMs

From this folder:

```bash
python train_test_svm.py            # full 500-shuffle sweep (~20–25 min CPU)
python train_test_svm.py --quick    # 20-shuffle smoke test (~2 min CPU)
python train_test_svm.py --seed 1   # change RNG seed
```

For each variant (2, 3, 5) and each training-set size on a sweep
matching `data/training_frame3_results.csv`, the script:

1. Randomly shuffles the 1000-sample dataset.
2. Trains a `sklearn.svm.SVC(kernel='linear')` on the first `f` samples.
3. Tests on the next 100 samples and records the recognition rate.
4. Repeats 1–3 `N_SHUFFLES` times (default 500) and reports mean ± std.

Two feature spaces are evaluated independently:

- **Raw distance vector (6-d)** — the six sensor distance readings
  themselves, used directly. This is the *Raw SVM* baseline curve.
- **SWRC feature vector (2500-d)** — the spin-wave reservoir
  `m_z`-at-frame-100 readout from `data/swrc_features.npz`. This is
  the *SWRC* curve.

The water-wave reservoir curve comes from
`data/training_frame3_results.csv` (lab measurement, not recomputed).

Expected output (written to `figures/`):

| File | What it shows |
|---|---|
| `FinalResultsMain.png` | SWRC recognition rate vs training-set size with error bars, all three variants |
| `SWvsSVMW.png` | SWRC vs Raw SVM, side-by-side per variant |
| `WWvsSVMvsSW.png` | Three-way comparison (water-wave reservoir vs Raw SVM vs SWRC) — the published Fig. 2e |

Plus three intermediate score caches:

| File | Content |
|---|---|
| `figures/SWvsSVM_avs.csv` | SWRC mean recognition rate, shape `(3, len(fs))` |
| `figures/SWvsSVM_std.csv` | SWRC std across shuffles, shape `(3, len(fs))` |
| `figures/SWvsSVM_svm.csv` | Raw SVM mean recognition rate, shape `(3, len(fs))` |

For the published seed (`--seed 0`, `--n-shuffles 500`), the curves
plateau at:

- 2-class Raw SVM ≈ 92 %; SWRC reaches 100 %.
- 3-class Raw SVM ≈ 87 %; SWRC reaches 100 %.
- 5-class Raw SVM ≈ 68 %; SWRC reaches 100 % by `f ≈ 500`.

These match the published Fig. 2e numbers to within shuffle noise.
