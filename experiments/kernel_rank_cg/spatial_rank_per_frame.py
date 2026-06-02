"""spatial_rank_per_frame.py

Spatial-only kernel rank. Instead of the full spatiotemporal volume (X*T
features per sample), treat each of the 201 frames as a SEPARATE feature set:
the Samples matrix for frame t is 256 x g^2 (the coarse-grained spatial field
at that frame). Compute the rank per frame, then report, per coarse-graining
grid, the frame with the HIGHEST rank.

Now the feature dimension is g^2, so for the coarse grids (g <= 15) the rank
ceiling is g^2 (< N), not the sample count -- this is the clean test of
whether the g^2 spatial features stay independent (full rank = g^2) or become
redundant past the spatial-Nyquist scale.

Outputs (this dir): spatial_rank_results.npz
"""
import os, time, gc
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
CUBE = os.environ.get(
    "CUBE_PATH",
    "/path/to/spin_wave_cube.npy")  # set CUBE_PATH env var
GRIDS = [50, 40, 30, 20, 10, 8, 6, 4]
EPS = 1e-3


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
print("spatial_rank_per_frame.py  started  ",
      time.strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 78)

cube = np.load(CUBE)                                # (256,201,50,50) f32
N, T, H, W = cube.shape
print(f"cube {cube.shape} dtype={cube.dtype}")
assert (H, W) == (50, 50)


def cg_cube(cube, g):
    if g == 50:
        return cube.astype(np.float32, copy=True)
    out = np.empty((N, T, g, g), dtype=np.float32)
    t0 = time.time()
    for r in range(N):
        for t in range(T):
            out[r, t] = cv2.resize(cube[r, t], (g, g),
                                   interpolation=cv2.INTER_AREA)
        _progress(r, N, t0, label=f"CG g={g}")
    return out


n_g = len(GRIDS)
d95_gt   = np.zeros((n_g, T), dtype=int)        # 95%-variance kernel rank
kr_gt    = np.zeros((n_g, T), dtype=int)        # eps=1e-3 kernel rank
raw_gt   = np.zeros((n_g, T), dtype=int)        # numerical rank
pr_gt    = np.zeros((n_g, T), dtype=float)      # participation ratio
bound_g  = np.zeros(n_g, dtype=int)             # min(g^2, N-1)

for gi, g in enumerate(GRIDS):
    xcg = cg_cube(cube, g)
    D = g * g
    bound_g[gi] = min(D, N - 1)
    for t in range(T):
        X = xcg[:, t, :, :].reshape(N, D).astype(np.float64)
        X -= X.mean(axis=0, keepdims=True)
        S = np.linalg.svd(X, full_matrices=False, compute_uv=False)
        kr_gt[gi, t]  = int((S > S[0] * EPS).sum())
        tol = S[0] * max(N, D) * np.finfo(np.float64).eps
        raw_gt[gi, t] = int((S > tol).sum())
        ev = S ** 2
        cs = np.cumsum(ev / ev.sum())
        d95_gt[gi, t] = int(np.searchsorted(cs, 0.95) + 1)
        pr_gt[gi, t]  = float(ev.sum() ** 2 / (ev ** 2).sum())
    del xcg; gc.collect()
    tb = int(d95_gt[gi].argmax())
    print(f"  g={g:2d}  D={D:5d}  bound={bound_g[gi]:3d}  | best frame "
          f"t*={tb:3d}: d95={d95_gt[gi,tb]:3d}  kr(1e-3)={kr_gt[gi,tb]:3d}  "
          f"raw={raw_gt[gi,tb]:3d}  PR={pr_gt[gi,tb]:6.1f}  | "
          f"d95 over frames: min={d95_gt[gi].min()} max={d95_gt[gi].max()}")

np.savez_compressed(
    os.path.join(HERE, "spatial_rank_results.npz"),
    grids=np.array(GRIDS), n_frames=T, n_samples=N, eps=EPS,
    d95_gt=d95_gt, kr_gt=kr_gt, raw_gt=raw_gt, pr_gt=pr_gt, bound_g=bound_g)
print("saved spatial_rank_results.npz")

print("\n--- BEST-FRAME SPATIAL RANK per grid (frame chosen = argmax d95) ---")
print(f"{'g':>4} {'D=g^2':>7} {'bound':>6} {'t*':>4} {'d95':>5} {'KR_norm':>8} "
      f"{'kr1e-3':>7} {'raw':>5} {'PR':>7}  {'d95 min-max':>12}")
for gi, g in enumerate(GRIDS):
    tb = int(d95_gt[gi].argmax())
    D = g * g
    print(f"{g:>4} {D:>7} {bound_g[gi]:>6} {tb:>4} {d95_gt[gi,tb]:>5} "
          f"{d95_gt[gi,tb]/bound_g[gi]:>8.3f} {kr_gt[gi,tb]:>7} "
          f"{raw_gt[gi,tb]:>5} {pr_gt[gi,tb]:>7.1f}  "
          f"{d95_gt[gi].min():>5}-{d95_gt[gi].max():<5}")
print("done", time.strftime("%Y-%m-%d %H:%M:%S"))
