"""Comparison figure for the 32x32 XOR ladder (Supplementary Fig. 7) with a
summary table of the smallest-parameter feed-forward baseline matching the
spin-wave reservoir.

Output: figures/figS7_xor_ladder_comparison.png

Panels P=16..1 left->right with 'SW' labels, plus a third row: a table listing,
per rung, the SW target accuracy and the smallest-parameter MLP/CNN-on-coords
architecture that matches or beats it.
"""
import os, sys, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FixedLocator

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
FIGD = os.path.join(HERE, 'figures')

DATA_SCALES = [1, 2, 4, 8, 16]      # scale axis order in the swd .npz arrays
SCALES = [16, 8, 4, 2, 1]           # panel order, left -> right
GRID = 32
NPIX = GRID * GRID
C_SW = '#e0533a'   # red
C_MLP = '#2e7d32'  # forest green
C_CNN = '#1f4f9e'  # blue


def xor_pattern(N, P):
    ii = np.arange(N)[:, None]
    jj = np.arange(N)[None, :]
    return ((ii // P) + (jj // P)) & 1


def pareto(F, H):
    """Upper-left Pareto frontier: max accuracy at each ascending FLOPs."""
    order = np.argsort(F)
    best = -np.inf
    px, py = [], []
    for fv, hv in zip(F[order], H[order]):
        if hv > best:
            px.append(fv); py.append(hv); best = hv
    return px, py


def load_ext(name):
    fp = os.path.join(DATA, name)
    return np.load(fp, allow_pickle=True) if os.path.exists(fp) else None


mlp = np.load(os.path.join(DATA, 'mlp_sweep_xor.npz'), allow_pickle=True)
cnn = np.load(os.path.join(DATA, 'cnn_sweep_xor.npz'), allow_pickle=True)
swd = np.load(os.path.join(DATA, 'topk_svm_32_45mT_t05.npz'), allow_pickle=True)
mlp_ext = load_ext('mlp_ext_xor.npz')
cnn_ext = load_ext('cnn_ext_xor.npz')

mlp_idx = {int(p): i for i, (g, p) in enumerate(zip(mlp['task_grid'], mlp['task_P']))
           if int(g) == GRID}
cnn_idx = {int(p): i for i, (g, p) in enumerate(zip(cnn['task_grid'], cnn['task_P']))
           if int(g) == GRID}

BIN = int(np.where(swd['readouts'] == 'binary')[0][0])
swd_ba = swd['bal_acc']                       # (scale, K, readout, fold)
swd_fit = swd['flops_fit']                    # (scale, K, readout, fold)
swd_pred = swd['flops_predict_sample'].astype(float)   # 2K, per K
swd_nte = float(np.mean(swd['n_test']))       # mean test-fold size
mlp_ep = int(mlp['epochs'])
cnn_ep = int(cnn['epochs'])


def mlp_total_flops(fwd, n_train, epochs):
    return 3.0 * fwd * n_train * epochs + fwd * (NPIX - n_train)


def cnn_total_flops(fwd_img, epochs):
    return fwd_img * (3.0 * epochs + 1.0)


# ---- smallest-parameter model matching the SW target, per rung ----
m_depths, m_widths = list(mlp['depths']), list(mlp['widths'])
c_depths, c_widths = list(cnn['depths']), list(cnn['widths'])
m_params = np.asarray(mlp['params'], float)
c_params = np.asarray(cnn['params'], float)


def collect_mlp(P):
    out = []
    mi = mlp_idx[P]
    mf = np.asarray(mlp['mean_final'][mi])
    for di, d in enumerate(m_depths):
        for wi, w in enumerate(m_widths):
            out.append((float(m_params[di, wi]), float(mf[di, wi]), f'd{d}w{w}'))
    if mlp_ext is not None and P in list(mlp_ext['task_P']):
        col = list(mlp_ext['task_P']).index(P)
        accs = np.asarray(mlp_ext['mean_final'])[:, col]
        pr = np.asarray(mlp_ext['params'], float)
        ad, aw = list(mlp_ext['arch_d']), list(mlp_ext['arch_w'])
        for ai in range(len(ad)):
            pv = pr[ai, col] if pr.ndim == 2 else pr[ai]
            out.append((float(pv), float(accs[ai]), f'd{ad[ai]}w{aw[ai]}'))
    return out


def collect_cnn(P):
    out = []
    ck = cnn_idx[P]
    mf = np.asarray(cnn['mean_final'][ck])
    for di, d in enumerate(c_depths):
        for wi, w in enumerate(c_widths):
            out.append((float(c_params[di, wi]), float(mf[di, wi]), f'd{d}w{w}'))
    if cnn_ext is not None and P in list(cnn_ext['task_P']):
        col = list(cnn_ext['task_P']).index(P)
        accs = np.asarray(cnn_ext['mean_final'])[:, col]
        pr = np.asarray(cnn_ext['params'], float)
        ad, aw = list(cnn_ext['arch_d']), list(cnn_ext['arch_w'])
        for ai in range(len(ad)):
            pv = pr[ai, col] if pr.ndim == 2 else pr[ai]
            out.append((float(pv), float(accs[ai]), f'd{ad[ai]}w{aw[ai]}'))
    return out


def smallest_match(cands, target):
    hits = [c for c in cands if c[1] >= target]
    return min(hits, key=lambda c: c[0]) if hits else None


# ---- figure ----
fig = plt.figure(figsize=(19, 10.5))
# Row order (top -> bottom): decision-boundary thumbnails (n labels), then the
# accuracy-vs-FLOPs plots, then the summary table.
# Outer split: top block = thumbnails + plots (+ xlabels + legend gap below it);
# bottom block = table.
outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.42],
                         hspace=0.46, wspace=0.0,
                         left=0.105, right=0.99, top=0.97, bottom=0.06)
# Top block: small thumbnails sit close above the plots (tight inner hspace).
gs_topblk = outer[0].subgridspec(2, 1, height_ratios=[0.42, 1.0], hspace=0.20)
gs_thumbs = gs_topblk[0].subgridspec(1, 5, wspace=0.25)
gs_top = gs_topblk[1].subgridspec(1, 5, wspace=0.25)
# Bottom block: table row touches (wspace=0) so the 5 column-axes visually merge
# into one continuous table.
gs_table = outer[1].subgridspec(1, 5, wspace=0.0)

table_rows = {'P = 16': [], 'P = 8': [], 'P = 4': [], 'P = 2': [], 'P = 1': []}
top = []
thumb_axes = []
for ci, P in enumerate(SCALES):
    ax = fig.add_subplot(gs_top[0, ci],
                         sharex=top[0] if top else None,
                         sharey=top[0] if top else None)
    top.append(ax)
    si = DATA_SCALES.index(P)

    # --- SW reservoir, top-K sweep ---
    ba = np.nanmean(swd_ba[si, :, BIN, :], axis=1)
    sd = np.nanstd(swd_ba[si, :, BIN, :], axis=1)
    swd_x = np.nanmean(swd_fit[si, :, BIN, :], axis=1) + swd_pred * swd_nte
    ax.fill_between(swd_x, ba - sd, ba + sd, color=C_SW, alpha=0.15, zorder=2)
    ax.plot(swd_x, ba, color=C_SW, lw=2.5, marker='s', ms=7, mec='none',
            zorder=6, label='SW reservoir')
    swd_target = float(ba[-1])
    ax.axhline(swd_target, color=C_SW, lw=0.9, ls=':', alpha=0.65, zorder=1)

    # --- MLP-on-coords (Pareto front only) ---
    mi = mlp_idx[P]
    n_tr_mlp = int(mlp['n_train_per_task'][mi])
    Hm = list(mlp['mean_final'][mi].ravel())
    Fm = list(mlp_total_flops(mlp['fwd_flops'].astype(float).ravel(),
                              n_tr_mlp, mlp_ep))
    if mlp_ext is not None and P in list(mlp_ext['task_P']):
        col = list(mlp_ext['task_P']).index(P)
        n_tr_e = int(mlp_ext['n_train_per_task'][col])
        ep_e = int(mlp_ext['epochs'])
        Fm += list(mlp_total_flops(mlp_ext['fwd_flops'].astype(float),
                                   n_tr_e, ep_e))
        Hm += list(mlp_ext['mean_final'][:, col])
    Fm, Hm = np.array(Fm), np.array(Hm)
    ax.scatter(Fm, Hm, s=27, c=C_MLP, marker='o', alpha=0.45, edgecolors='none',
               zorder=3)
    px, py = pareto(Fm, Hm)
    ax.plot(px, py, color=C_MLP, lw=2.5, zorder=4, label='MLP-on-coords')

    # --- CNN-on-coords (Pareto front only) ---
    ck = cnn_idx[P]
    Hc = list(cnn['mean_final'][ck].ravel())
    Fc = list(cnn_total_flops(cnn['fwd_flops_img_per_task'][ck].astype(float).ravel(),
                              cnn_ep))
    if cnn_ext is not None and P in list(cnn_ext['task_P']):
        col = list(cnn_ext['task_P']).index(P)
        ep_e = int(cnn_ext['epochs'])
        fwd_img_e = cnn_ext['fwd_flops_per_pixel'].astype(float) * NPIX
        Fc += list(cnn_total_flops(fwd_img_e, ep_e))
        Hc += list(cnn_ext['mean_final'][:, col])
    Fc, Hc = np.array(Fc), np.array(Hc)
    ax.scatter(Fc, Hc, s=32, c=C_CNN, marker='D', alpha=0.55, edgecolors='none',
               zorder=3)
    px, py = pareto(Fc, Hc)
    ax.plot(px, py, color=C_CNN, lw=2.5, zorder=4, label='CNN-on-coords')

    ax.set_xscale('log')
    ax.set_ylim(0.45, 1.03)
    ax.set_box_aspect(1)
    ax.axhline(0.5, color='k', lw=0.5, ls=':', alpha=0.35, zorder=1)
    ax.grid(True, which='both', alpha=0.22)
    ax.xaxis.set_major_locator(FixedLocator([1e6, 1e8, 1e10, 1e12]))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=[2.0, 3.0, 5.0, 7.0], numticks=24))
    ax.tick_params(axis='x', which='minor', length=3)
    ax.tick_params(labelsize=20)
    ax.set_xlabel('total FLOPs', fontsize=24)
    # y quantity is carried by the 'Accuracy vs Compute' row title (left margin);
    # keep tick numbers on the leftmost panel only, drop the inline ylabel.
    if ci != 0:
        plt.setp(ax.get_yticklabels(), visible=False)

    # --- XOR label thumbnail ---
    axt = fig.add_subplot(gs_thumbs[0, ci])
    thumb_axes.append(axt)
    axt.imshow(xor_pattern(GRID, P), cmap='gray_r', vmin=0, vmax=1, aspect='equal')
    axt.set_xticks([]); axt.set_yticks([])
    for sp in axt.spines.values():
        sp.set_linewidth(0.8)
    axt.set_title(f'n = {ci + 1}', fontsize=26, pad=8)

    # --- smallest matching model for this rung ---
    mhit = smallest_match(collect_mlp(P), swd_target)
    chit = smallest_match(collect_cnn(P), swd_target)
    cand = []
    if mhit is not None:
        cand.append(('MLP', mhit))
    if chit is not None:
        cand.append(('CNN', chit))
    best = min(cand, key=lambda kv: kv[1][0]) if cand else None
    if best is None:
        cells = [f'{swd_target:.3f}', '—', '—', '—']
    else:
        kind, (pv, acc, lab) = best
        cells = [f'{swd_target:.3f}', f'{kind} {lab}',
                 f'{int(pv):,d}', f'{acc:.3f}']
    table_rows[f'P = {P}'] = cells

# ---- table row: one sub-table per column, aligned exactly with the panels ----
row_labels = ['SW target', 'Smallest model', 'Parameters', 'Model accuracy']
n_rows = 4
row_face = ['#ffffff', '#f4f4f4', '#ffffff', '#f4f4f4']
tab_axes = []
for ci, P in enumerate(SCALES):
    axc = fig.add_subplot(gs_table[0, ci])    # 5 axes touch (wspace=0) → unified table
    axc.axis('off')
    cells = [[table_rows[f'P = {P}'][r]] for r in range(n_rows)]
    t = axc.table(cellText=cells, loc='center', cellLoc='center',
                  bbox=[0.0, 0.0, 1.0, 1.0])
    t.auto_set_font_size(False)
    t.set_fontsize(24)
    # Hide the left border on columns 1+ so adjacent column-axes share one
    # vertical rule instead of drawing two stacked ones (which look fuzzy).
    edges = 'BTR' if ci > 0 else 'BLTR'
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor('#bbbbbb')
        cell.set_facecolor(row_face[r])
        cell.set_height(1.0 / n_rows)
        cell.visible_edges = edges
    tab_axes.append(axc)

# row labels for the table rows AND the two macro rows above it, all in one
# left-margin column at a shared x so the whole figure reads as one big table.
pos = tab_axes[0].get_position()
label_x = pos.x0 - 0.045          # left of the leftmost plot's y-tick numbers
for r, lab in enumerate(row_labels):
    yc = pos.y1 - (r + 0.5) / n_rows * (pos.y1 - pos.y0)
    fig.text(label_x, yc, lab, ha='right', va='center',
             fontsize=23, fontweight='bold')
# macro-row titles, vertically centred on the thumbnail and plot rows
tpos = thumb_axes[0].get_position()
fig.text(label_x, (tpos.y0 + tpos.y1) / 2.0, 'Problem geometry',
         ha='right', va='center', fontsize=23, fontweight='bold')
ppos = top[0].get_position()
fig.text(label_x, (ppos.y0 + ppos.y1) / 2.0, 'Accuracy vs Compute',
         ha='right', va='center', fontsize=23, fontweight='bold')

# line-colour legend in the gap below the plot xlabels and above the table
p_bot = top[0].get_position().y0          # bottom of the (middle) plot row
tab_top = tab_axes[0].get_position().y1   # top of the table row
y_leg = (p_bot + tab_top) / 2.0           # centre of the gap, clear of the 'total FLOPs' xlabels
h, l = top[0].get_legend_handles_labels()
fig.legend(h, l, loc='center', ncol=3, fontsize=24,
           frameon=False, bbox_to_anchor=(0.5, y_leg))

out = os.path.join(FIGD, 'figS7_xor_ladder_comparison.png')
fig.savefig(out, dpi=160, bbox_inches='tight')
plt.close(fig)
print('saved', out, f'({os.path.getsize(out)//1024} KB)')
