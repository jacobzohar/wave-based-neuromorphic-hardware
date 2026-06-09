# `simulator/` — the SWRC spin-wave-disk simulator

This is the core of the repository: the **mumax³ code that simulates the
spin-wave-disk reservoir** used throughout the study. The simulator drives the
disk with a pair of Gaussian-windowed RF pulses at coordinate-encoded times,
records the out-of-plane magnetisation `m_z(t)` of the disk, and produces a
`(N_samples, T, H, W)` float32 cube that the downstream analyses consume.

The substrate is **task-agnostic** — the same cube is reused for every
classification / dimensionality experiment in `../Demo/`. If you only
want a 200-line introduction to the simulator before reading the production
code, start with [`../tutorial/SWRC_MuMax3.ipynb`](../tutorial/SWRC_MuMax3.ipynb).

## Files

| File | Role |
|------|------|
| [`run_sweep.py`](run_sweep.py) | **The production generator.** Self-contained: embeds the mumax³ templates, builds the relax cache once, runs all `GRID²` `(X1, X2)` samples (one GPU each, `N_GPUS` in parallel within a single job), reads each OVF, crops + block-means to the chosen coarse grid, and writes per-sample CSV directories + `grid_int.npy` + `MANIFEST.txt`. The default is the 32×32 / `T_step = 0.05 ns` / 4-GPU sweep used for the manuscript's Fig. 5 and Supplementary Fig. 7; every parameter is overridable via env vars (see below). |
| [`build_relax_cache.mx3`](build_relax_cache.mx3) | Standalone, human-readable mumax³ source for the one-time relaxed equilibrium state (identical to the script's embedded `relax_script()`). Read this first if you want to see the substrate physics without parsing the Python templates. |
| [`test_sample.mx3`](test_sample.mx3) | Standalone, human-readable mumax³ source for one swept sample (identical structure to the script's embedded `sample_script()`). The header documents the multi-sample sweep recipe. |
| [`build_cube.py`](build_cube.py) | Stacks the per-sample CSV run-directories into the `(N, T, H, W)` float32 `.npy` cube the analysis loads. Takes the source directory and the output name as args, filters to the `GRID²` paired `(X1, X2)` runs, and prints `m_z` stats + per-pixel std + dead-cell count. |

## Physical model (from the `.mx3`)

- **Geometry:** 1 µm circular disk on a 512×512×1 grid, 2 nm × 2 nm × 1.5 nm cells, single z-layer.
- **Material:** `Msat = 1e6 A/m`, `Aex = 1.5e-11 J/m`, `alpha = 0.012`, uniaxial `Ku1 = 6.3e5 J/m³` (just above the spin-reorientation transition), out-of-plane easy axis.
- **Bias:** `B_ext = 45 mT` in-plane (`+x`).
- **Transducers:** `(num_inputs + 1) · 2 = 6` actuator regions (50 nm circles at radius 300 nm, evenly spaced); 2 are driven inputs.
- **Drive:** each input fires one Gaussian-windowed `800 MHz` pulse, `σ = 1.25 ns`, amplitude `8e5` on `Ku1`, centred at `coord · T_step + 0.2 ns`.

## Encoding scale (the "matched window")

| Variant | `T_step` | `Tmax` | frames | cube shape |
|---------|---------:|-------:|-------:|------------|
| **default (matched window)** | 0.05 ns | 4 ns | 201 | `(1024, 201, 64, 64)` |
| coarse-step | 0.10 ns | 6 ns | 300 | `(1024, 300, 64, 64)` |

The matched-window default keeps the encoding window the same ~1.5 ns the
earlier 16×16 sweep used, at doubled input resolution.

## Regenerate a cube

```bash
# needs a GPU + the mumax3 binary
export MUMAX3_PATH=/path/to/mumax3                # or put mumax3 on PATH
export SWRC_OUTPUT_DIR=/scratch/sweep_out
export SWRC_N_GPUS=4                              # GPUs to use in parallel

python run_sweep.py                               # -> 1024 sample_*.out/ CSV dirs + grid_int.npy + MANIFEST.txt
                                                  #    ~250 min wall on 4 GPUs (per the published MANIFEST)

python build_cube.py --src $SWRC_OUTPUT_DIR \
       --out sw_disk_mz_cube.npy --frames 201     # stack CSV dirs -> the .npy cube
```

All other sweep parameters (grid, Tmax, T_step, crop window, coarse-grain size)
are exposed as `SWRC_*` environment variables — see the `run_sweep.py`
docstring for the full list.

**Requirements:** `mumax3` (GPU micromagnetics, <https://mumax.github.io/>) +
Python `numpy` (and `pandas` for faster CSV reads in `build_cube.py`).

## Where the cubes go

The cubes produced here feed the analyses in `../Demo/`:

- [`../Demo/reservoir_metrics/xor_chequerboard/`](../Demo/reservoir_metrics/xor_chequerboard/) — top-K MI feature selection + LinearSVC readout, compared against MLP / CNN baselines on an XOR-chequerboard task ladder.
- [`../Demo/reservoir_metrics/kernel_rank_cg/`](../Demo/reservoir_metrics/kernel_rank_cg/) — kernel rank, d95 and participation ratio of the reservoir feature matrix vs spatial coarse-graining.

The cubes themselves are multi-GB and are not bundled — see *Data
availability* in the top-level [`../README.md`](../README.md).
