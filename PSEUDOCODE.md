# Pseudocode — algorithmic description of the SWRC pipeline

Pseudocode for the four stages of the spin-wave reservoir computer (SWRC)
pipeline. Each algorithm block points at the corresponding Python/mumax³
source file in this repository.

The four stages and their concrete implementations:

| Stage | Pseudocode block | Implementation |
|---|---|---|
| 1. Substrate simulation sweep | [Algorithm 1](#algorithm-1--substrate-simulation-sweep-sw_disk_runsweep) | [`simulator/run_sweep.py`](simulator/run_sweep.py) + [`simulator/build_relax_cache.mx3`](simulator/build_relax_cache.mx3) + the embedded mumax³ template |
| 2. Cube assembly | [Algorithm 2](#algorithm-2--cube-assembly-build_cube) | [`simulator/build_cube.py`](simulator/build_cube.py) |
| 3. Top-K mutual-information feature readout | [Algorithm 3](#algorithm-3--top-k-mi--linearsvc-readout) | [`examples/minimal_classification/train_minimal.py`](examples/minimal_classification/train_minimal.py) (LOO-CV, N = 4) and [`experiments/xor_checkerboard/top_k_svm_sweep.py`](experiments/xor_checkerboard/top_k_svm_sweep.py) (15-fold CV, N = 1024) |
| 4. Reservoir dimensionality analysis (kernel rank / d95 / participation ratio) | [Algorithm 4](#algorithm-4--kernel-rank-d95-and-participation-ratio-vs-spatial-coarse-graining) | [`experiments/kernel_rank_cg/kernel_rank_cg.py`](experiments/kernel_rank_cg/kernel_rank_cg.py) |

All numerical constants in the algorithms below are the production defaults
used to generate the manuscript figures; the implementations expose every
constant as an environment variable or CLI argument so the same code reproduces
both the minimal worked example and the full production sweeps.

---

## Algorithm 1 — Substrate simulation sweep (`sw_disk_run_sweep`)

Simulates the spin-wave-disk reservoir under every input pair of a
`G × G` coordinate-encoded input grid, producing per-sample time series of the
out-of-plane magnetisation `m_z(t, y, x)` on a coarse-grained spatial grid.

**Input**
- `G ∈ ℤ⁺` — input-grid side (production: 32; minimal example: 2)
- `T_step ∈ ℝ⁺` — coordinate→time scale (production: 0.05 ns)
- `T_max ∈ ℝ⁺` — simulation window (production: 4 ns; minimal: 2 ns)
- `Δt_fix ∈ ℝ⁺` — fixed integrator time step (default: 0.2 ps)
- `t_off ∈ ℝ⁺` — pre-pulse offset to give the substrate one Gaussian half-width
  of pre-roll before the first input (default: 0.2 ns)
- `σ_p ∈ ℝ⁺` — Gaussian pulse half-width (default: 1.25 ns)
- `f_p ∈ ℝ⁺` — pulse carrier frequency (default: 800 MHz)
- `A_p ∈ ℝ⁺` — pulse drive amplitude in Ku1 units (default: 8 × 10⁵ J/m³)
- `M_cg ∈ ℤ⁺` — spatial coarse-graining target (default: 64 ⇒ 12 nm/cell)
- substrate physical parameters: `M_s = 10⁶ A/m`, `A_ex = 1.5 × 10⁻¹¹ J/m`,
  `α = 0.012`, `K_u1 = 6.3 × 10⁵ J/m³`, `B_ext = 45 mT` (in-plane),
  disk radius `R = 500 nm` on a `512 × 512 × 1` cell mesh (2 nm xy, 1.5 nm z)

**Output**
- `cube ∈ ℝ^{G²×F×M_cg×M_cg}`, where `F = ⌊T_max / (100 Δt_fix)⌋ + 1` is the
  frame count (production: F = 201).

**Steps**

```
procedure sw_disk_run_sweep(G, T_step, T_max, M_cg, …):

  # ----- One-time substrate equilibration -----
  if RELAX_OVF not on disk:
      script := build_geometry()
                  + "m = uniform(0, 0.01, 1)"     # break z-symmetry
                  + "relax()"                     # mumax3 conjugate-gradient relax
                  + "saveas(m, M_Relax_Initial)"
      run mumax3 on script
      cache m_z OVF as RELAX_OVF

  # ----- Per-sample sweep over the G×G input grid -----
  for (X1, X2) in {0,1,…,G−1}² in parallel across N_GPUS workers:
      t_c1 := X1 · T_step                          # coordinate→time encoding
      t_c2 := X2 · T_step
      script := build_geometry()
                  + load_relax_cache(RELAX_OVF)
                  + set FixDt := Δt_fix, Tmax := T_max
                  + autosave(m_z, every 100·Δt_fix)
                  + assign region 1 a Ku1 modulation pulse(t; t_c1)
                  + assign region 2 a Ku1 modulation pulse(t; t_c2)
                  + run(T_max)
      run mumax3 on script
      for each emitted OVF frame f_t:
          mz_t := read_ovf2(f_t)              # 512 × 512 float32 m_z map
          mz_t := mz_t[64:448, 64:448]        # central 384×384 crop (whole disk)
          mz_t := block_mean(mz_t, M_cg)      # area-average 6×6 → 64×64
          write mz_t.csv
      remove the raw OVF + .mx3 script (CSV is the persistent artefact)

  # The drive pulse for region i at coordinate-encoded time t_ci:
  function pulse(t; t_c) :=
      A_p · exp( −((t − t_c − t_off) / σ_p)² )
            · sin( 2π f_p (t − t_c − t_off) )

  return the (G², F, M_cg, M_cg) ensemble of CSV directories.
```

The sweep is **resumable**: a sample whose output directory already contains
`F` CSV frames is skipped. The script `simulator/build_relax_cache.mx3` is
the standalone mumax³ source used by the in-process relax step;
`simulator/test_sample.mx3` is the standalone mumax³ source for one sample run
(both human-readable for substrate verification). The `Ku1` modulation is the
voltage-controlled magnetic anisotropy (VCMA) drive; see
`simulator/GEOMETRY.md` for the experimental-voltage conversion convention.

---

## Algorithm 2 — Cube assembly (`build_cube`)

Stacks the per-sample CSV directories produced by Algorithm 1 into a single
dense `(N, F, H, W)` `float32` `m_z` cube for downstream analysis.

**Input**
- `src` — directory containing `sample_X1{i}X2{j}.out/` subdirectories of CSV
  frames produced by Algorithm 1.
- `F ∈ ℤ⁺` — expected frame count per sample (must match Algorithm 1's
  emission rate; production: 201, minimal example: 101).

**Output**
- `cube ∈ ℝ^{N × F × H × W}`, where `N = number of paired (X1, X2).out dirs`
  and `H = W` is the spatial side of the first emitted CSV frame.
- `run_index.json` — a `{cube_row_index → (X1, X2, dir_name)}` map preserving
  the substrate-grid ↔ cube-row correspondence.

**Steps**

```
procedure build_cube(src, F):
  D := sort by dir name { d in listdir(src) : d matches "sample_X1*X2*.out" }
  N := |D|
  parse first frame from D[0] to autodetect (H, W); assert square
  allocate cube ∈ ℝ^{N × F × H × W}
  for i, d in enumerate(D):                       # rows are sorted (X1, X2)
      frames := sort glob(src / d / "m_z*.csv")
      assert |frames| == F
      for f_idx, fp in enumerate(frames):
          cube[i, f_idx] := read_csv(fp)          # (H, W) float32
      run_index[i] := { "X1": X1(d), "X2": X2(d), "dir": d }
  write cube to .npy
  write run_index to .json
  report: m_z min/max/mean/std, per-pixel std-across-runs,
          dead-cell count (pixels with std < 1e-5 in a centred 96% crop)
  return cube, run_index
```

---

## Algorithm 3 — Top-K MI + LinearSVC readout

The reservoir's only trainable layer. Reduces the high-dimensional reservoir
state to a `K`-feature vector by ranking voxels (frame × y × x) by their
mutual information with the task label, then trains a linear support vector
classifier on those features under leave-one-out (minimal example) or
repeated stratified k-fold (production) cross-validation.

**Input**
- `cube ∈ ℝ^{N × F × H × W}` — the assembled `m_z` cube from Algorithm 2.
- `y ∈ {0, 1}^N` — binary task labels (XOR labels in the included experiments).
- `K ∈ ℤ⁺` — number of top-MI features to retain (e.g. K = 4 for the minimal
  example; swept from 1 to ~1000 in the production XOR-checkerboard study).
- `B ∈ ℤ⁺` — number of equal-frequency MI bins (default: 2 for binary tasks).

**Output**
- `bal_acc ∈ [0, 1]` — cross-validated balanced accuracy.
- The `K`-tuple of selected voxel coordinates `(t, y, x)` and the per-voxel MI
  scores (in bits, with Miller–Madow bias correction).

**Steps**

```
procedure top_k_mi_linearsvc(cube, y, K, B):

  # ----- Per-voxel mutual information I(m_z; y) -----
  flatten cube to X ∈ ℝ^{N × (F·H·W)}                       # one column per voxel
  for each voxel column v:
      θ := equal-frequency quantile thresholds of X[:, v] into B bins
      codes[:, v] := bin index of X[i, v] under θ           # ∈ {0, …, B−1}
      build joint count tensor jc[b, c] := #{ i : codes[i, v] = b, y[i] = c }
      p_xy := jc / N
      p_x := Σ_c p_xy[:, c],  p_y := Σ_b p_xy[b, :]
      MI[v] := Σ_{b, c} p_xy[b, c] · log₂( p_xy[b, c] / (p_x[b] · p_y[c]) )
      apply Miller–Madow bias correction:
          bias := (m_xy − m_x − m_y + 1) / (2 N ln 2)
                  where m_* are counts of non-empty marginal/joint cells
      MI[v] := max( MI[v] − bias, 0 )

  # ----- Top-K feature selection -----
  Π := argsort(MI) descending
  top_K := Π[0 : K]
  features ∈ ℝ^{N × K} := cube columns at top_K

  # ----- Cross-validated LinearSVC -----
  pipe := StandardScaler → LinearSVC(C = 1.0, squared-hinge loss, dual=auto)
  cv   := LeaveOneOut    (minimal example, N = 4)
       OR RepeatedStratifiedKFold(n_splits = k, n_repeats = r,
                                   random_state seeded by job-id)
                                                            # production
  for each (train_idx, test_idx) in cv.split(features, y):
      fit pipe on (features[train_idx], y[train_idx])
      ŷ[test_idx] := pipe.predict(features[test_idx])
  bal_acc := balanced_accuracy_score(y, ŷ)

  return bal_acc, top_K, MI
```

The Miller–Madow correction `(m_xy − m_x − m_y + 1) / (2N ln 2)` is the
small-sample bias correction for the plug-in MI estimator. At very small `N`
(= 4 in the minimal example) it can transiently produce MI values above the
hard-bound `H(Y)` of the label; this is estimator noise, not a violation of
information theory, and is documented in
[`examples/minimal_classification/train_minimal.py`](examples/minimal_classification/train_minimal.py).
The production pipeline runs at `N = 1024`, where the bias is negligible.

---

## Algorithm 4 — Kernel rank, d95, and participation ratio vs spatial coarse-graining

Quantifies how the reservoir's effective dimensionality scales with the
spatial resolution at which the substrate field is read out. Reproduces
**Supplementary Table 2 / Table S9.1** and **Supplementary Figure S9.2** of
the manuscript.

**Input**
- `cube ∈ ℝ^{N × F × H × W}` (production: `(256, 201, 50, 50)` — the 16 × 16
  / 256-run / 50 × 50 spin-wave-disk cube at 45 mT, 800 MHz).
- `G = (g₁, g₂, …)` — ladder of target coarse-grained grids
  (production: `{50, 40, 30, 20, 10, 8, 6, 4}`).
- `E = (ε₁, ε₂, …)` — singular-value threshold ratios for kernel rank
  (production: `{10⁻³, 10⁻², 5 × 10⁻²}`; 10⁻³ is the headline threshold).

**Output**
For each grid `g ∈ G`:
- **kernel rank** `KR(g, ε) := |{i : σᵢ > ε · σ₁}|` for each `ε ∈ E`
- **dₚ** for `p ∈ {50, 95, 99}`%: the smallest number of principal components
  whose cumulative explained variance reaches `p`
- **participation ratio** `PR(g) := (Σ σᵢ²)² / Σ σᵢ⁴`
- **raw numerical rank** with NumPy's machine-eps tolerance
- the normalised singular-value spectrum and cumulative variance curve

**Steps**

```
procedure kernel_rank_cg(cube, G, E):
  N, F, H, W := shape(cube)
  assert (H, W) = (50, 50)

  for g in G:
      # 1. Spatial coarse-graining (Nyquist-respecting area-average)
      if g == 50:
          X_cg := cube                                       # identity
      else:
          allocate X_cg ∈ ℝ^{N × F × g × g}
          for r in 0..N−1, t in 0..F−1:
              X_cg[r, t] := cv2.resize(cube[r, t], (g, g), INTER_AREA)

      # 2. Build the feature matrix Ψc and mean-centre across samples
      X := reshape X_cg to (N, F·g²) as float64
      X := X − mean(X, axis = sample)                         # column-centring

      # 3. Economy SVD on a (N × F·g²) matrix (skip U, V for memory)
      σ := svd(X, full_matrices = false, compute_uv = false)  # σ₁ ≥ σ₂ ≥ …

      # 4a. Kernel rank at each threshold
      for ε in E:
          KR(g, ε) := |{ i : σᵢ > ε · σ₁ }|

      # 4b. Raw numerical rank (NumPy convention)
      tol := σ₁ · max(N, F·g²) · machine_eps(float64)
      raw_rank(g) := |{ i : σᵢ > tol }|

      # 5. Variance-based dimensionalities
      EV := σ²
      cumvar := cumulative_sum(EV / Σ EV)
      d_p(g) := min{ k : cumvar[k] ≥ p / 100 }      for p in {50, 95, 99}
      PR(g)  := (Σ EV)² / Σ EV²

  return tables KR, d_p, PR, raw_rank, σ-spectra, cumvar curves
```

The companion script
[`experiments/kernel_rank_cg/spatial_rank_per_frame.py`](experiments/kernel_rank_cg/spatial_rank_per_frame.py)
runs the same SVD step on each frame independently (a `(N × g²)` matrix per
frame) to test whether the `g²` spatial features remain independent past the
spatial-Nyquist scale of the disk's spin-wave dispersion.

---

## Verifying these algorithms against the implementation

Each pseudocode block above tracks the corresponding source file
line-for-line. Constants and small bookkeeping details (resumability,
progress reporting, MANIFEST writing, dead-cell diagnostics, dtype
promotions) are documented inline in the implementation but omitted here.

When the implementation and this document disagree, the implementation is the
source of truth — please open an issue on the public repository so this
document can be brought back into sync.
