"""make_figures.py

Figures for the spatial kernel-rank / d95 vs spatial-CG-scale experiment.
Reads kernel_rank_results.npz + montage_data.npz (the cached outputs of
kernel_rank_cg.py) and writes the rendered PNGs to figures/.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

R = np.load(os.path.join(HERE, "kernel_rank_results.npz"))
M = np.load(os.path.join(HERE, "montage_data.npz"))

grids = R["grids"]                       # [50,40,30,20,10,8,6,4]
eps = R["eps_list"]                      # [1e-3,1e-2,5e-2]
kr = R["kernel_rank"]                    # (8,3)
d50, d95, d99 = R["d50"], R["d95"], R["d99"]
raw_rank = R["raw_rank"]
pr = R["participation_ratio"]
n_feat = R["n_feat"]
sv = R["sv_spectra"]                     # (8,256) normalised
cv = R["cumvar"]                         # (8,256)
N = int(R["n_samples"])                  # 256
CEIL = N - 1                             # 255 (mean-centred)
mframe = int(R["montage_frame"])

ng = len(grids)
x = np.arange(ng)                        # categorical (grids unevenly spaced)
xlab = [str(g) for g in grids]
NYQ = (3.5, 5.5)                         # g~10-8 band index range for shading
cols = plt.cm.viridis(np.linspace(0, 0.92, ng))

saved = []
def finish(fig, name):
    p = os.path.join(FIGDIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(p)
    print("saved", name)

# ------------------------------------------------------------------ fig1
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
a = ax[0]
a.axhspan(NYQ[0], NYQ[1], color="gold", alpha=0.18,
          label="spatial-Nyquist band (g~10-8)")
a.axhline(CEIL, color="0.4", ls="--", lw=1.3, label=f"rank ceiling = {CEIL}")
a.plot(x, kr[:, 0], "-o", color="crimson", lw=2.6, ms=9,
       label="kernel rank ($\\epsilon=10^{-3}$)")
a.set_ylim(200, 262)
a.set_xticks(x); a.set_xticklabels(xlab)
a.set_xlabel("coarse-grained grid  $g \\times g$")
a.set_ylabel("kernel rank  (of 255)")
a.set_title("Kernel rank vs spatial CG scale")
a.grid(alpha=0.3); a.legend(fontsize=9, loc="lower left")
a2 = a.twinx()
a2.plot(x, d95, "-s", color="navy", lw=2.2, ms=8, label="d95")
a2.set_ylabel("d95  (PCs for 95% variance)", color="navy", labelpad=2)
a2.tick_params(axis="y", colors="navy")
a2.set_ylim(0, 40)
a2.legend(fontsize=9, loc="upper right")
for i in range(ng):
    a.annotate(str(kr[i, 0]), (x[i], kr[i, 0]), textcoords="offset points",
               xytext=(0, 8), ha="center", fontsize=8, color="crimson")

b = ax[1]
b.axvspan(NYQ[0], NYQ[1], color="gold", alpha=0.18)
b.axhline(1.0, color="0.4", ls="--", lw=1.3, label="fraction = 1")
b.plot(x, kr[:, 0] / CEIL, "-o", color="crimson", lw=2.6, ms=9,
       label="kernel rank / 255")
b.plot(x, raw_rank / CEIL, "-^", color="seagreen", lw=2.0, ms=8,
       label="raw numerical rank / 255")
b.plot(x, d95 / CEIL, "-s", color="navy", lw=2.0, ms=7, label="d95 / 255")
b.set_xticks(x); b.set_xticklabels(xlab)
b.set_xlabel("coarse-grained grid  $g \\times g$")
b.set_ylabel("fraction of rank ceiling (255)")
b.set_title("Rank utilisation fraction")
b.set_ylim(0, 1.08)
b.grid(alpha=0.3); b.legend(fontsize=9, loc="center left")
fig.suptitle("45 mT / 800 MHz SW-disk: spatial kernel rank "
             "(256 runs, full spatiotemporal volume)", fontsize=12)
fig.subplots_adjust(wspace=0.42)
finish(fig, "fig1_rank_vs_grid.png")

# ------------------------------------------------------------------ fig2
fig, a = plt.subplots(figsize=(8.5, 6))
idx = np.arange(1, N + 1)
for i in range(ng):
    a.semilogy(idx, sv[i], lw=1.9, color=cols[i], label=f"g={grids[i]}")
for e in eps:
    a.axhline(e, color="0.55", ls=":", lw=1.1)
    a.text(N * 0.97, e * 1.25, f"$\\epsilon={e:g}$", fontsize=8,
           ha="right", color="0.4")
a.set_xlabel("singular-value index")
a.set_ylabel("normalised singular value  $\\sigma_i/\\sigma_1$")
a.set_title("Singular-value spectra across spatial CG scales")
a.set_xlim(0, N); a.set_ylim(1e-7, 1.5)
a.grid(alpha=0.3, which="both"); a.legend(fontsize=9, ncol=2)
finish(fig, "fig2_sv_spectra.png")

# ------------------------------------------------------------------ fig3
fig, a = plt.subplots(figsize=(8.5, 6))
idx = np.arange(1, N + 1)
for i in range(ng):
    a.plot(idx, cv[i], lw=2.0, color=cols[i], label=f"g={grids[i]}  (d95={d95[i]})")
a.axhline(0.95, color="crimson", ls="--", lw=1.4, label="95% variance")
a.axhline(0.99, color="darkorange", ls=":", lw=1.3, label="99% variance")
a.set_xlabel("number of principal components")
a.set_ylabel("cumulative explained variance")
a.set_title("Cumulative explained variance across spatial CG scales")
a.set_xlim(0, 70); a.set_ylim(0, 1.02)
a.grid(alpha=0.3); a.legend(fontsize=8.5, ncol=2, loc="lower right")
finish(fig, "fig3_cumvar.png")

# ------------------------------------------------------------------ fig4
fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
a = ax[0]
a.axvspan(NYQ[0], NYQ[1], color="gold", alpha=0.18)
a.axhline(CEIL, color="0.4", ls="--", lw=1.2, label=f"ceiling = {CEIL}")
mk = ["o", "D", "v"]
for ei, e in enumerate(eps):
    a.plot(x, kr[:, ei], "-" + mk[ei], lw=2.3, ms=8,
           label=f"kernel rank $\\epsilon={e:g}$")
a.plot(x, d99, "-s", color="0.35", lw=1.8, ms=7, label="d99")
a.plot(x, d95, "-^", color="navy", lw=1.8, ms=7, label="d95")
a.plot(x, d50, "-p", color="teal", lw=1.8, ms=7, label="d50")
a.set_xticks(x); a.set_xticklabels(xlab)
a.set_xlabel("coarse-grained grid  $g \\times g$")
a.set_ylabel("effective dimension / rank")
a.set_title("Threshold sweep: kernel rank + variance dimensions")
a.set_yscale("log"); a.grid(alpha=0.3, which="both")
a.legend(fontsize=8.5, ncol=2)

b = ax[1]
b.axvspan(NYQ[0], NYQ[1], color="gold", alpha=0.18,
          label="Nyquist band")
b.plot(x, pr, "-o", color="purple", lw=2.6, ms=9, label="participation ratio")
b.set_xticks(x); b.set_xticklabels(xlab)
b.set_xlabel("coarse-grained grid  $g \\times g$")
b.set_ylabel("participation ratio  $(\\Sigma\\lambda)^2/\\Sigma\\lambda^2$")
b.set_title("Smooth effective rank (variance spread)")
b.set_ylim(0, 28)
b.grid(alpha=0.3); b.legend(fontsize=9)
for i in range(ng):
    b.annotate(f"{pr[i]:.1f}", (x[i], pr[i]), textcoords="offset points",
               xytext=(0, 8), ha="center", fontsize=8, color="purple")
fig.suptitle("Multi-metric effective rank vs spatial CG scale",
             fontsize=12)
finish(fig, "fig4_multimetric.png")

# ------------------------------------------------------------------ fig5
vmax = max(abs(float(M[f"montage_g{g}"].min())) for g in grids)
vmax = max(vmax, max(abs(float(M[f"montage_g{g}"].max())) for g in grids))
fig, ax = plt.subplots(1, ng, figsize=(2.0 * ng, 2.7))
for i, g in enumerate(grids):
    im = ax[i].imshow(M[f"montage_g{g}"], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      origin="lower")
    ax[i].set_title(f"{g}x{g}", fontsize=10)
    ax[i].set_xticks([]); ax[i].set_yticks([])
fig.suptitle(f"Spatial coarse-graining of the $m_z$ field "
             f"(run 0, frame {mframe})", fontsize=11, y=1.02)
cb = fig.colorbar(im, ax=list(ax), fraction=0.018, pad=0.01)
cb.set_label("$m_z$")
finish(fig, "fig5_cg_montage.png")

print("done -", len(saved), "figures")
