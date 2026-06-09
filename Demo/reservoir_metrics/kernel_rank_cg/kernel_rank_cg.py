"""kernel_rank_cg.py

Spatial kernel-rank (eps=1e-3) and d95 vs spatial
coarse-graining scale on the 45 mT / 800 MHz SW-disk reservoir cube.

Run on the workstation that holds the input cube. The input cube has shape
(256,201,50,50) float32 = 256 runs (the 16x16 direct-temporal input space)
x 201 frames x 50x50 m_z field.

Per grid g in {50,40,30,20,10,8,6,4}: spatially coarse-grain every (run,frame)
2D m_z map to g x g via cv2 INTER_AREA (area-averaging), flatten the full
spatiotemporal volume per run -> 256 x (201*g^2) Samples matrix, mean-centre
across the 256 samples, one economy SVD -> kernel rank + d95 + companions.

Outputs (this dir): kernel_rank_results.npz, montage_data.npz
"""
import os, time, gc
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
# CUBE_PATH points at the (256,201,50,50) float32 .npy of the SW-disk m_z
# field — too large (~5 GB) to bundle; supply via env var.
CUBE = os.environ.get(
    "CUBE_PATH",
    "/path/to/spin_wave_cube.npy")

# Coarse-graining ladder: native 50x50 down to 4x4 (below the spatial-Nyquist
# scale of the disk's spin-wave dispersion at the 800 MHz drive).
GRIDS = [50, 40, 30, 20, 10, 8, 6, 4]
# Singular-value thresholds for kernel rank: rank_eps(X) = #{ sigma_i / sigma_1 > eps }.
# 1e-3 is the primary threshold; the other two are reported as robustness
# checks against the choice of threshold.
EPS_LIST = [1e-3, 1e-2, 5e-2]
# Single frame from the late ringdown used for the coarse-grain montage panel
# in Fig. 5; ~180 / 201 frames -> near the end of the simulation window where
# the disk's response is the most spatially structured.
MONTAGE_FRAME = 180


def _progress(i, n, t0, label=""):
    pct = (i + 1) / n
    el = time.time() - t0
    eta = el / pct * (1 - pct) if pct > 0 else 0
    bar = ("#" * int(30 * pct)).ljust(30, ".")
    print(f"\r[{bar}] {i+1}/{n} {pct*100:.0f}% | {el:.0f}s | ETA {eta:.0f}s | {label}",
          end="", flush=True)
    if i + 1 == n:
        print()


print("=" * 78)
print("kernel_rank_cg.py  started", time.strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 78)

cube = np.load(CUBE)                                    # (256,201,50,50) f32
N, T, H, W = cube.shape
print(f"cube: {CUBE}")
print(f"  shape={cube.shape} dtype={cube.dtype} "
      f"min={cube.min():.4f} max={cube.max():.4f} mean={cube.mean():.6f}")
assert (H, W) == (50, 50), f"expected 50x50 spatial, got {H}x{W}"
CUBE_MEAN = float(cube.mean())


def cg_cube(cube, g):
    """Spatially coarse-grain (N,T,50,50) -> (N,T,g,g) via cv2 INTER_AREA.

    INTER_AREA is the Nyquist-respecting low-pass (area-average): each output
    cell is the mean of the input cells that overlap it. This is the correct
    spatial CG for a continuous physical field; bilinear/bicubic would alias.
    """
    if g == 50:
        return cube.astype(np.float32, copy=True)       # identity (no CG)
    out = np.empty((N, T, g, g), dtype=np.float32)
    t0 = time.time()
    for r in range(N):
        for t in range(T):
            out[r, t] = cv2.resize(cube[r, t], (g, g),
                                   interpolation=cv2.INTER_AREA)
        _progress(r, N, t0, label=f"CG g={g}")
    return out


n_g = len(GRIDS)
kernel_rank = np.zeros((n_g, len(EPS_LIST)), dtype=int)
d50 = np.zeros(n_g, int); d95 = np.zeros(n_g, int); d99 = np.zeros(n_g, int)
raw_rank = np.zeros(n_g, int)
part_ratio = np.zeros(n_g)
n_feat = np.zeros(n_g, int)
mean_check = np.zeros(n_g)
sv_spectra = np.full((n_g, N), np.nan)        # normalised singular values
cumvar = np.full((n_g, N), np.nan)            # cumulative explained variance
montage = {}

for gi, g in enumerate(GRIDS):
    # 1. Spatially coarse-grain the whole cube to g x g.
    xcg = cg_cube(cube, g)
    # Stash one snapshot per grid for the Fig. 5 montage (visual sanity check).
    montage[f"montage_g{g}"] = xcg[0, MONTAGE_FRAME].copy()
    mean_check[gi] = float(xcg.mean())
    if g == 50:                                         # identity validation
        assert np.array_equal(xcg, cube), "g=50 CG is not the identity"

    # 2. Flatten per run -> (N, T*g*g) feature matrix; promote to float64 for
    #    SVD numerical stability. Mean-centre across the N samples (column-wise)
    #    so the SVD captures across-sample variance, not the static field.
    X = xcg.reshape(N, -1).astype(np.float64)            # 256 x (201*g^2)
    del xcg; gc.collect()
    n_feat[gi] = X.shape[1]
    X -= X.mean(axis=0, keepdims=True)

    # 3. Economy SVD: at the largest grid, X is 256 x 502500 -- compute_uv=False
    #    skips materialising U/V (each would be GB) and returns only the
    #    min(N, D) singular values.
    S = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    del X; gc.collect()

    # 4a. Kernel rank: count singular values above eps * sigma_max for each
    #     threshold. Scale-blind and noise-floor-aware (project convention).
    for ei, eps in enumerate(EPS_LIST):
        kernel_rank[gi, ei] = int((S > S[0] * eps).sum())
    # 4b. Raw numerical rank: numpy's matrix_rank tolerance (machine-eps scaled).
    tol = S[0] * max(N, n_feat[gi]) * np.finfo(np.float64).eps
    raw_rank[gi] = int((S > tol).sum())

    # 5. Variance-based dimensionality summaries:
    #    d_p = #PCs needed to explain p% of variance, and the participation
    #    ratio PR = (sum sigma^2)^2 / sum sigma^4 (effective number of
    #    contributing modes; PR <= min(N, D)).
    ev = S ** 2
    evr = ev / ev.sum()
    cs = np.cumsum(evr)
    d50[gi] = int(np.searchsorted(cs, 0.50) + 1)
    d95[gi] = int(np.searchsorted(cs, 0.95) + 1)
    d99[gi] = int(np.searchsorted(cs, 0.99) + 1)
    part_ratio[gi] = float(ev.sum() ** 2 / (ev ** 2).sum())
    # 6. Keep the normalised singular-value spectrum + cumulative variance for
    #    the figure-stage plots.
    ns = S.shape[0]
    sv_spectra[gi, :ns] = S / S[0]
    cumvar[gi, :ns] = cs
    print(f"  g={g:2d}  D={n_feat[gi]:7d}  KR(1e-3)={kernel_rank[gi,0]:3d}  "
          f"d95={d95[gi]:3d}  d99={d99[gi]:3d}  raw_rank={raw_rank[gi]:3d}  "
          f"PR={part_ratio[gi]:7.1f}  mean={mean_check[gi]:.6f}")
    gc.collect()

np.savez_compressed(
    os.path.join(HERE, "kernel_rank_results.npz"),
    grids=np.array(GRIDS), eps_list=np.array(EPS_LIST),
    kernel_rank=kernel_rank, d50=d50, d95=d95, d99=d99, raw_rank=raw_rank,
    participation_ratio=part_ratio, n_feat=n_feat, mean_check=mean_check,
    sv_spectra=sv_spectra, cumvar=cumvar, n_samples=N,
    cube_mean=CUBE_MEAN, montage_frame=MONTAGE_FRAME)
np.savez_compressed(
    os.path.join(HERE, "montage_data.npz"),
    grids=np.array(GRIDS), montage_frame=MONTAGE_FRAME, **montage)
print("saved kernel_rank_results.npz, montage_data.npz")

print("\n--- SUMMARY (sample count N = 256, rank ceiling = 255) ---")
print(f"{'g':>4} {'D':>9} {'KR1e-3':>7} {'KR1e-2':>7} {'KR5e-2':>7} "
      f"{'d50':>5} {'d95':>5} {'d99':>5} {'raw':>5} {'PR':>9}")
for gi, g in enumerate(GRIDS):
    print(f"{g:>4} {n_feat[gi]:>9} {kernel_rank[gi,0]:>7} {kernel_rank[gi,1]:>7} "
          f"{kernel_rank[gi,2]:>7} {d50[gi]:>5} {d95[gi]:>5} {d99[gi]:>5} "
          f"{raw_rank[gi]:>5} {part_ratio[gi]:>9.1f}")
print("\ncube native mean =", f"{CUBE_MEAN:.6f}",
      "  (INTER_AREA preserves the spatial mean -> mean_check should match)")
print("done", time.strftime("%Y-%m-%d %H:%M:%S"))
