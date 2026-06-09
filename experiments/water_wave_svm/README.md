# Water-wave reservoir vs. SVM vs. spin-wave reservoir — Fig. 2e

Trains and tests linear support vector machines on the
obstacle-distance classification task and produces three of the
manuscript's main-text comparison figures:

| Figure file | Manuscript role |
|---|---|
| `figures/FinalResultsMain.png` | SWRC alone, recognition rate vs training-set size, with error bars (3-, 2-, 5-class) |
| `figures/SWvsSVMW.png` | Raw linear-SVM baseline vs SWRC, side-by-side, per output-class count |
| `figures/WWvsSVMvsSW.png` | Three-way comparison: water-wave reservoir (lab measurement) vs raw SVM vs SWRC |

These are the SVM-trained panels referenced from the manuscript's
Methods section ("training on output information") and used to support
the Fig. 2e / Fig. 5b comparison narrative.

## What the script does

For each output-class variant (2, 3, 5) and each training-set size on a
sweep determined by `data/training_frame3_results.csv`:

1. Randomly shuffles the 1000-sample dataset (the 3-class variant has 999
   real samples; the 1000th row is zero-padded — see the *Inputs* table).
2. Trains a `sklearn.svm.SVC(kernel='linear')` on the first `f` samples.
3. Tests on the next 100 samples and records the recognition rate.
4. Repeats steps 1–3 `N_SHUFFLES` times (default 500) and reports mean ± std.

Two feature spaces are evaluated independently:

- **Raw distance vector** (6-d) — the six sensor distance readings
  themselves, used directly by the linear SVM. This is the
  *Raw SVM* baseline curve.
- **SWRC feature vector** (2500-d) — the spin-wave-disk reservoir's
  `m_z` field at frame 100, z-scored per sample and flattened, then read
  out by the same linear SVM. This is the *SWRC* curve.

The water-wave reservoir scores are not recomputed; they are loaded
verbatim from the lab-measurement CSV at `data/training_frame3_results.csv`.

The SVM training/testing logic is `sklearn.svm.SVC(kernel='linear')`
fitted on the training subset and scored on the test subset — no kernel
trick, no class weights, no regularisation tuning. The classification
work is therefore entirely linear; any non-linearity in the
recognition-rate curves comes from the reservoir feature lift, not the
classifier.

## Inputs (`data/`)

| File | Shape / size | Source |
|---|---|---|
| `Train2X.csv` | (1000, 6) ints in [1, 6] | `datasets/water_wave_obstacle/2_output_unique_data_generation.ipynb` |
| `Train2Y.csv` | (1000,) ints in {0, 1} | same |
| `Train3X.csv` | (999, 6) ints in [1, 6] | `datasets/water_wave_obstacle/3_output_unique_data_generation.ipynb` |
| `Train3Y.csv` | (999,) ints in {0, 1, 2} | same |
| `Train5X.csv` | (1000, 6) ints in [1, 7] | `datasets/water_wave_obstacle/5_output_unique_data_generation.ipynb` |
| `Train5Y.csv` | (1000,) ints in {0, 1, 2, 3, 4} | same |
| `training_frame3_results.csv` | (33, 4) cols: `[training_size, score2, score3, score5]` (lab-measurement recognition rates) | water-wave reservoir lab measurement |
| `swrc_features.npz` | `psi` = (3, 1000, 2500) float32 | spin-wave-disk reservoir cube at frame 100; z-scored per sample and flattened. Extracted from the RCT9 sweep produced by `simulator/run_sweep.py` at `B_ext = 45 mT`, `f = 800 MHz`, `1 µm` disk. |

The CSV inputs match the bundles in
[`../../datasets/water_wave_obstacle/`](../../datasets/water_wave_obstacle/),
but are committed here verbatim so the figure regenerates from one folder
without requiring users to first generate the datasets themselves. Set
`random.seed` before calling `generate_balanced_dataset` if you regenerate
them; the published numbers come from the bundled CSVs.

## How to run

```bash
# from this folder, with the top-level requirements.txt installed
python train_test_svm.py
# -> figures/FinalResultsMain.png, SWvsSVMW.png, WWvsSVMvsSW.png
```

CLI:

| Flag | Default | Notes |
|---|---|---|
| `--data-dir PATH` | `./data` | location of the input CSV and `.npz` files |
| `--out-dir PATH` | `./figures` | where to write the output PNGs and intermediate score caches |
| `--seed N` | 0 | numpy RNG seed (controls shuffle order; published curves used seed 0) |
| `--n-shuffles N` | 500 | random splits per `(variant, training-size)` cell |
| `--quick` | — | shorthand for `--n-shuffles 20`, ≈ 2 min total runtime — useful smoke test |

The script also writes three intermediate score caches into
`figures/`:

- `SWvsSVM_avs.csv` — `(3, len(fs))` SWRC mean recognition rate
- `SWvsSVM_std.csv` — `(3, len(fs))` SWRC std across shuffles
- `SWvsSVM_svm.csv` — `(3, len(fs))` Raw SVM mean recognition rate

These are the raw numbers that go into the three figures.

## Expected run time

| Mode | Hardware | Time |
|---|---|---|
| `--quick` (20 shuffles) | normal laptop CPU | ~2 minutes |
| Full (500 shuffles, default) | normal laptop CPU | ~20–25 minutes |

The Raw-SVM sweep is essentially free (6-d inputs); almost all the
runtime is in the SWRC sweep (2500-d inputs).

## Expected output

The full run writes (to `figures/`):

```
FinalResultsMain.png       (SWRC alone, errorbar)
SWvsSVMW.png               (SWRC vs Raw SVM)
WWvsSVMvsSW.png            (WW vs SVM vs SWRC)
SWvsSVM_avs.csv            (SWRC mean recognition rate)
SWvsSVM_std.csv            (SWRC std across shuffles)
SWvsSVM_svm.csv            (Raw SVM mean recognition rate)
```

For the published seed (`--seed 0`, `--n-shuffles 500`), the curves
plateau at:

- 2-class Raw SVM ≈ 92 %; SWRC reaches 100 %.
- 3-class Raw SVM ≈ 87 %; SWRC reaches 100 %.
- 5-class Raw SVM ≈ 68 %; SWRC reaches 100 % by `f ≈ 500`.

These match the published numbers to within shuffle-noise.
