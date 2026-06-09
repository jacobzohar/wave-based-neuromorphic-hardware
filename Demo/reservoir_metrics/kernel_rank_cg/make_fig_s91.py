"""make_fig_s91.py
Supplementary Fig. S9.1: KR (d95) + participation ratio vs spatial CG grid,
and cumulative explained variance curves. Reads kernel_rank_results.npz.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
R = np.load(os.path.join(HERE, "kernel_rank_results.npz"))

grids = R["grids"]
d95 = R["d95"]
pr = R["participation_ratio"]
raw_rank = R["raw_rank"]
cv = R["cumvar"]
N = int(R["n_samples"])
CEIL = N - 1

ng = len(grids)
x = np.arange(ng)
xlab = [f"{g}×{g}" for g in grids]
NYQ = (3.5, 5.5)                       # g~10-8 index band

fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.8))

# (a) KR (d95) + participation ratio vs grid
a = ax[0]
a.axvspan(NYQ[0], NYQ[1], color="gold", alpha=0.20,
          label="spatial-Nyquist band")
a.plot(x, d95, "-s", color="navy", lw=2.4, ms=9, label="kernel rank KR (d95)")
a.plot(x, pr, "-o", color="purple", lw=2.4, ms=9, label="participation ratio PR")
a.set_xticks(x); a.set_xticklabels(xlab, rotation=45, ha="right")
a.set_xlabel("coarse-grained spatial grid")
a.set_ylabel("effective dimension")
a.set_ylim(0, 30)
a.set_title("(a)  Kernel rank and participation ratio", fontsize=11, loc="left")
a.grid(alpha=0.3)
a.legend(fontsize=9, loc="lower left")
a.text(0.97, 0.95, f"rank(Ψ$_c$) = {CEIL} (max) at every grid",
       transform=a.transAxes, ha="right", va="top", fontsize=9,
       bbox=dict(boxstyle="round", fc="white", ec="0.7"))

# (b) cumulative explained variance curves
b = ax[1]
cols = plt.cm.viridis(np.linspace(0, 0.92, ng))
idx = np.arange(1, N + 1)
for i in range(ng):
    b.plot(idx, cv[i], lw=2.0, color=cols[i],
           label=f"{grids[i]}×{grids[i]}  (KR={d95[i]})")
b.axhline(0.95, color="crimson", ls="--", lw=1.5, label="95% threshold")
b.set_xlim(0, 60); b.set_ylim(0, 1.02)
b.set_xlabel("number of singular components  r")
b.set_ylabel("cumulative explained variance  C(r)")
b.set_title("(b)  Cumulative explained variance", fontsize=11, loc="left")
b.grid(alpha=0.3)
b.legend(fontsize=8, ncol=2, loc="lower right")

fig.tight_layout()
out = os.path.join(FIGDIR, "figS91_kr_pr.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print("saved", out)
