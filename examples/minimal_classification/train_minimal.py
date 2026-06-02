"""Minimal SWRC classifier: top-K MI features + LinearSVC + LOO-CV on a
4-sample 2x2 XOR task. See ./README.md for the full pipeline.

Inputs (encoded by `run_example.sh`):
    --cube      path to (4, T, H, W) float32 m_z cube
    --out_dir   where to write result.json + response_snapshot.png

Labels are computed inline: for each sample (X1, X2) in {0, 1}^2 the XOR label
is (X1 XOR X2) -> [0, 1, 1, 0] for samples ordered by sorted (X1, X2) tuples.
This matches the ordering build_cube.py uses (sorted by dir name).
"""
import os, json, argparse
import numpy as np


def per_voxel_mi(cube, labels, k_bins=2):
    """Plug-in MI estimator (bits) with Miller-Madow bias correction.

    Per voxel (t, y, x):
      1. Bin m_z into `k_bins` equal-frequency bins across the N samples.
      2. Build the joint count tensor jc[bin, class].
      3. MI = sum pxy * log2(pxy / (px * py)), minus the bias correction
            (m_xy - m_x - m_y + 1) / (2 N ln 2)
         where m_* are counts of non-empty cells.

    With only N = 4 samples and 2 classes, k_bins = 2 is the right call:
    equal-frequency binning forces a 2-vs-2 split per voxel.
    """
    N, T, H, W = cube.shape
    n_cells = T * H * W
    flat = cube.reshape(N, n_cells).astype(np.float32)
    qs = np.arange(1, k_bins) / k_bins
    thr = np.quantile(flat, qs, axis=0)                          # (k_bins-1, n_cells)
    codes = np.zeros((N, n_cells), dtype=np.int8)
    for k in range(k_bins - 1):
        codes += (flat >= thr[k][None, :]).astype(np.int8)

    LN2 = np.log(2.0)
    jc = np.zeros((k_bins, 2, n_cells), dtype=np.float64)
    for c in (0, 1):
        sub = codes[labels == c]
        for k in range(k_bins):
            jc[k, c] = (sub == k).sum(axis=0)
    Nv = jc.sum(axis=(0, 1))
    pxy = jc / Nv
    px = pxy.sum(axis=1); py = pxy.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = (pxy * (np.log(pxy) - np.log(px[:, None, :])
                       - np.log(py[None, :, :]))) / LN2
    mi = np.nansum(term, axis=(0, 1))
    m_xy = (jc > 0).sum(axis=(0, 1))
    m_x  = (px > 0).sum(axis=0)
    m_y  = (py > 0).sum(axis=0)
    bias = (m_xy - m_x - m_y + 1) / (2.0 * Nv * LN2)
    return np.clip(mi - bias, 0.0, None).reshape(T, H, W)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cube', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--K', type=int, default=4,
                    help='top-K MI features for the SVM (default: 4)')
    args = ap.parse_args()

    # ---- Load cube + define labels -----------------------------------------
    cube = np.load(args.cube)                                    # (4, T, H, W)
    N, T, H, W = cube.shape
    if N != 4:
        raise SystemExit(f'expected N=4 samples, got N={N}; check the sweep')

    # Coordinates ordered as build_cube.py sorts them: by dir name -> X1, X2
    # = (0,0), (0,1), (1,0), (1,1). XOR labels = [0, 1, 1, 0].
    coords = [(0, 0), (0, 1), (1, 0), (1, 1)]
    labels = np.array([x1 ^ x2 for x1, x2 in coords], dtype=np.int64)
    print(f'[load] cube {cube.shape} {cube.dtype}')
    print(f'[labels] coords={coords}  XOR labels={labels.tolist()}')

    # ---- Per-voxel MI on the XOR labels ------------------------------------
    mi = per_voxel_mi(cube, labels, k_bins=2)                    # (T, H, W)
    flat_mi = mi.ravel()
    order = np.argsort(-flat_mi, kind='stable')
    top_idx = order[:args.K]
    t_idx, y_idx, x_idx = np.unravel_index(top_idx, mi.shape)
    feats = cube[:, t_idx, y_idx, x_idx].astype(np.float32)      # (4, K)
    # Note: the Miller-Madow bias correction can let the plug-in MI estimator
    # transiently exceed H(Y) = 1 bit at very small N (= 4 here); this is
    # noise in the estimator, not a violation of information theory. The
    # estimator is appropriate for the production pipeline (N >= 256).
    print(f'[mi] peak MI = {float(flat_mi[top_idx[0]]):.3f} bits')
    print(f'[features] top-K coords (t, y, x): {list(zip(t_idx.tolist(), y_idx.tolist(), x_idx.tolist()))}')

    # ---- LinearSVC with leave-one-out CV -----------------------------------
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import balanced_accuracy_score

    loo = LeaveOneOut()
    fold_correct = []
    fold_records = []
    for fi, (tr, te) in enumerate(loo.split(feats, labels)):
        pipe = Pipeline([
            ('sc', StandardScaler()),
            ('clf', LinearSVC(C=1.0, loss='squared_hinge', dual='auto',
                              max_iter=5000, random_state=0)),
        ])
        pipe.fit(feats[tr], labels[tr])
        pr = pipe.predict(feats[te])
        ok = int(pr[0] == labels[te][0])
        fold_correct.append(ok)
        fold_records.append({
            'fold': fi, 'test_sample': int(te[0]),
            'true_label': int(labels[te][0]), 'predicted_label': int(pr[0]),
            'correct': bool(ok),
        })

    bal_acc = balanced_accuracy_score(labels, [r['predicted_label'] for r in
                                               sorted(fold_records, key=lambda r: r['test_sample'])])
    n_correct = sum(fold_correct); n_folds = len(fold_correct)
    print(f'[result] LOO-CV balanced accuracy = {bal_acc:.3f}  '
          f'({n_correct}/{n_folds} folds correct)')
    print(f'         top-K MI features used: K = {args.K}  '
          f'(peak MI = {float(flat_mi[top_idx[0]]):.3f} bits, '
          f'binary-task ceiling H(Y) = 1.000 bit)')

    # ---- Save result.json --------------------------------------------------
    result = {
        'cube_path': os.path.abspath(args.cube),
        'cube_shape': list(cube.shape),
        'N_samples': int(N),
        'coords': coords,
        'labels': labels.tolist(),
        'K': int(args.K),
        'top_K_voxels_tyx': list(zip(t_idx.tolist(), y_idx.tolist(), x_idx.tolist())),
        'peak_MI_bits': float(flat_mi[top_idx[0]]),
        'MI_ceiling_bits': 1.0,
        'loo_balanced_accuracy': float(bal_acc),
        'loo_fold_records': fold_records,
    }
    out_json = os.path.join(args.out_dir, 'result.json')
    with open(out_json, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'         saved {out_json}')

    # ---- Diagnostic figure: m_z snapshot at peak-MI frame for each sample --
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    peak_t = int(t_idx[0])
    fig, axes = plt.subplots(1, N, figsize=(3.2 * N, 3.5),
                             constrained_layout=True)
    vmax = float(np.abs(cube[:, peak_t]).max())
    for i, ax in enumerate(axes):
        im = ax.imshow(cube[i, peak_t], cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                       origin='lower')
        ax.set_title(f'sample {i}: X=({coords[i][0]},{coords[i][1]})  '
                     f'label={labels[i]}', fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'$m_z$ snapshot at peak-MI frame (t = frame {peak_t} of {T}); '
                 f'LOO bal acc = {bal_acc:.3f}', fontsize=11)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label('$m_z$', rotation=0, labelpad=10)
    out_png = os.path.join(args.out_dir, 'response_snapshot.png')
    fig.savefig(out_png, dpi=120)
    print(f'         saved {out_png}')


if __name__ == '__main__':
    main()
