"""make_fig_spatial.py
Spatial-only (per-frame) kernel rank: d95 vs frame for each grid + best-frame
d95 vs grid compared to the full-volume KR.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
S = np.load(os.path.join(HERE, "spatial_rank_results.npz"))
R = np.load(os.path.join(HERE, "kernel_rank_results.npz"))   # full-volume

grids = S["grids"]
d95_gt = S["d95_gt"]            # (8, 201)
bound_g = S["bound_g"]
T = int(S["n_frames"])
kr_full = R["d95"]             # full-volume KR per grid

ng = len(grids)
cols = plt.cm.viridis(np.linspace(0, 0.92, ng))
x = np.arange(ng)
xlab = [f"{g}×{g}" for g in grids]
best_t = d95_gt.argmax(axis=1)
best_d95 = d95_gt[np.arange(ng), best_t]

fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.9))

# (a) d95 vs frame, one curve per grid
a = ax[0]
for i in range(ng):
    a.plot(np.arange(T), d95_gt[i], lw=1.8, color=cols[i],
           label=f"{grids[i]}×{grids[i]}")
    a.plot(best_t[i], best_d95[i], "o", color=cols[i], ms=8,
           mec="k", mew=0.8, zorder=5)
a.set_xlabel("frame  t")
a.set_ylabel("spatial kernel rank  d95  (256 × g² matrix)")
a.set_title("(a)  Per-frame spatial rank — peak shifts early→late as g coarsens",
            fontsize=10.5, loc="left")
a.set_xlim(0, T)
a.grid(alpha=0.3)
a.legend(fontsize=8.5, ncol=2, title="grid")

# (b) best-frame spatial d95 vs grid, vs full-volume KR
b = ax[1]
b.plot(x, best_d95, "-s", color="navy", lw=2.4, ms=9,
       label="best-frame spatial KR (d95)")
b.plot(x, kr_full, "-^", color="crimson", lw=2.2, ms=8,
       label="full spatiotemporal-volume KR")
b.plot(x, bound_g, ":", color="0.5", lw=1.8, label="rank ceiling min(g², N−1)")
for i in range(ng):
    b.annotate(f"t*={best_t[i]}", (x[i], best_d95[i]),
               textcoords="offset points", xytext=(0, 9), ha="center",
               fontsize=8, color="navy")
b.set_xticks(x); b.set_xticklabels(xlab, rotation=45, ha="right")
b.set_xlabel("coarse-grained spatial grid")
b.set_ylabel("kernel rank  d95")
b.set_title("(b)  Best-frame spatial rank vs full-volume KR", fontsize=10.5,
            loc="left")
b.set_yscale("log")
b.set_ylim(5, 300)
b.grid(alpha=0.3, which="both")
b.legend(fontsize=9)

fig.tight_layout()
out = os.path.join(FIGDIR, "figS92_spatial_rank.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved", out)
