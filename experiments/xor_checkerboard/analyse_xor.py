"""Aggregate top-K SVM + MLP + CNN results into summary tables.

Reads:
  data/topk_svm_16_{drive}.npz        per drive at 16x16
  data/topk_svm_32_{drive_32}.npz     32x32 single drive
  data/mlp_sweep_xor.npz              MLP-on-coords sweep
  data/cnn_sweep_xor.npz              CNN-on-coords sweep (3x3 padding=same)
  data/pixel_mi_xor_summary_{grid}_{drive}.npz  per drive

Writes:
  data/analyse_summary.json
  data/topk_summary.csv              per (grid, drive, scale, K, readout)
  data/topk_winner_matrix.csv        per (grid, drive, scale) — best K*, best readout, ba_max
  data/smallest_matching_mlp.csv     per (grid, scale, threshold) — MLP
  data/smallest_matching_cnn.csv     per (grid, scale, threshold) — CNN
  data/mlp_vs_cnn_side_by_side.csv   per (grid, scale) — train-FLOPs match
  data/wavelength_match_table.csv    per (grid, scale, drive) — ba @ K_max with best readout

Summary tables logged to log.txt.
"""
import os, sys, glob, json, time
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
LOG = os.path.join(HERE, 'log.txt')

# Architecture grids -- must match the run_*_sweep.py constants or the indexing
# into the saved .npz arrays will misalign.
DEPTHS = [1, 2, 3, 4, 6, 8]
WIDTHS = [2, 4, 8, 16, 32, 64, 128, 256]
DEPTHS_CNN = [1, 2, 3, 4, 6]
WIDTHS_CNN = [4, 8, 16, 32, 64]

# Display labels for each of the 5 drives in the 16x16 wavelength-match panel.
# DRIVE_ORDER_16 is the row order used in the summary tables (ascending drive
# frequency so the wavelength-vs-substrate-period axis reads left-to-right).
DRIVE_LABELS_16 = {
    '45mT':     '45 mT / 800 MHz',
    '15mT200':  '15 mT / 200 MHz',
    '15mT1100': '15 mT / 1100 MHz',
    '30mT500':  '30 mT / 500 MHz',
    '60mT':     '60 mT / 1100 MHz',
}
DRIVE_ORDER_16 = ['15mT200', '30mT500', '45mT', '15mT1100', '60mT']
DRIVE_LABEL_32 = {'45mT_t05': '45 mT / 800 MHz (t=0.5, 32x32)'}

_log = open(LOG, 'a', encoding='utf-8')
def log(m=''):
    print(m, flush=True); _log.write(str(m) + '\n'); _log.flush()


def load_topk(grid):
    out = {}
    for fp in sorted(glob.glob(os.path.join(DATA, f'topk_svm_{grid}_*.npz'))):
        drv = os.path.basename(fp).replace(
            f'topk_svm_{grid}_', '').replace('.npz', '')
        out[drv] = np.load(fp, allow_pickle=True)
    return out


def load_mlp():
    fp = os.path.join(DATA, 'mlp_sweep_xor.npz')
    if not os.path.exists(fp):
        return None
    return np.load(fp, allow_pickle=True)


def load_cnn():
    fp = os.path.join(DATA, 'cnn_sweep_xor.npz')
    if not os.path.exists(fp):
        return None
    return np.load(fp, allow_pickle=True)


def main():
    log('\n' + '='*78)
    log(f'analyse_xor.py started {time.strftime("%Y-%m-%d %H:%M:%S")}')
    log('='*78)

    topk16 = load_topk(16)
    topk32 = load_topk(32)
    mlp = load_mlp()
    cnn = load_cnn()

    log(f'top-K SVM 16x16 drives loaded: {list(topk16.keys())}')
    log(f'top-K SVM 32x32 drives loaded: {list(topk32.keys())}')
    log(f'MLP loaded: {mlp is not None}')
    log(f'CNN loaded: {cnn is not None}')

    # ---- per-cell summary CSV ------------------------------------------------
    summary_rows = []   # full table
    winner_rows = []    # (grid, drive, scale) -> best (K, readout, ba_max)
    for grid_size, topk in ((16, topk16), (32, topk32)):
        for drv in (DRIVE_ORDER_16 if grid_size == 16 else sorted(topk.keys())):
            if drv not in topk: continue
            d = topk[drv]
            scales = d['scales']; K_list = d['K_list']
            readouts = [str(x) for x in d['readouts']]
            ba = d['bal_acc']                   # (n_scales, n_K, n_ro, n_folds)
            n_iter = d['n_iter']
            n_train = d['n_train'].astype(int)
            n_test = d['n_test'].astype(int)
            for si, P in enumerate(scales):
                P = int(P)
                best_ba_overall = -1.0; best_K = None; best_ro = None
                for ki, K in enumerate(K_list):
                    K = int(K)
                    for ri, ro in enumerate(readouts):
                        # Mean +/- std over the 15 CV folds for this (P, K, readout).
                        ba_mu = float(np.nanmean(ba[si, ki, ri]))
                        ba_sd = float(np.nanstd(ba[si, ki, ri]))
                        # SVM FLOPs: prediction is a dot product (2K MAC/sample);
                        # fit cost is 3 * n_iter * n_train * K under the LinearSVC
                        # squared-hinge accounting (constant 3 = forward + backward
                        # + parameter update per iteration). Median across folds.
                        f_pred = 2 * K
                        f_fit_med = float(3.0 * np.median(n_iter[si, ki, ri])
                                          * float(n_train.mean()) * K)
                        summary_rows.append(dict(
                            grid=grid_size, drive=drv, P=P, K=K, readout=ro,
                            bal_mean=ba_mu, bal_std=ba_sd,
                            flops_predict_sample=int(f_pred),
                            flops_fit_median=f_fit_med,
                            median_n_iter=int(np.median(n_iter[si, ki, ri])),
                            n_train_mean=int(n_train.mean()),
                            n_test_mean=int(n_test.mean())))
                        if ba_mu > best_ba_overall:
                            best_ba_overall = ba_mu
                            best_K = K; best_ro = ro
                winner_rows.append(dict(
                    grid=grid_size, drive=drv, P=P,
                    best_K=best_K, best_readout=best_ro,
                    bal_acc_best=best_ba_overall))

    # write CSVs
    with open(os.path.join(DATA, 'topk_summary.csv'), 'w', encoding='utf-8') as f:
        cols = ['grid', 'drive', 'P', 'K', 'readout', 'bal_mean', 'bal_std',
                'flops_predict_sample', 'flops_fit_median', 'median_n_iter',
                'n_train_mean', 'n_test_mean']
        f.write(','.join(cols) + '\n')
        for r in summary_rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')

    with open(os.path.join(DATA, 'topk_winner_matrix.csv'), 'w',
              encoding='utf-8') as f:
        cols = ['grid', 'drive', 'P', 'best_K', 'best_readout', 'bal_acc_best']
        f.write(','.join(cols) + '\n')
        for r in winner_rows:
            f.write(','.join(str(r[c]) for c in cols) + '\n')

    # ---- log key tables ------------------------------------------------------
    # Wavelength-match: ba @ K_max with best readout per (drive × P), 16x16
    log('\n' + '-'*78)
    log('Wavelength matching: SVM bal_acc per (drive × XOR-scale)')
    log('  Best-of-readouts at K=K_max (=2048 by default).')
    log('-'*78)
    if topk16:
        drvs = [d for d in DRIVE_ORDER_16 if d in topk16]
        any_d = topk16[drvs[0]]
        scales = any_d['scales']
        K_list = any_d['K_list']
        K_max_idx = int(np.argmax(K_list))
        readouts = [str(x) for x in any_d['readouts']]
        log('  drive          ' + '  '.join(f'P={int(P):2d}' for P in scales)
            + '    [readout, K=' + str(int(K_list[K_max_idx])) + ']')
        wm_rows = []
        for drv in drvs:
            d = topk16[drv]
            row_vals = []
            for si, P in enumerate(scales):
                vals = [(ri, str(d['readouts'][ri]),
                         float(np.nanmean(d['bal_acc'][si, K_max_idx, ri])))
                        for ri in range(len(readouts))]
                ri_best, ro_best, ba_best = max(vals, key=lambda x: x[2])
                row_vals.append(f'{ba_best:.3f}({ro_best[:1]})')
                wm_rows.append(dict(grid=16, drive=drv, P=int(P),
                                    K_max=int(K_list[K_max_idx]),
                                    best_readout=ro_best, bal_acc=ba_best))
            log(f'  {DRIVE_LABELS_16[drv]:18s}'
                + '  '.join(f'{v:>9s}' for v in row_vals))

        with open(os.path.join(DATA, 'wavelength_match_table.csv'), 'w',
                  encoding='utf-8') as f:
            cols = ['grid', 'drive', 'P', 'K_max', 'best_readout', 'bal_acc']
            f.write(','.join(cols) + '\n')
            for r in wm_rows:
                f.write(','.join(str(r[c]) for c in cols) + '\n')

    # 32x32
    if topk32:
        log('\n  32x32:')
        for drv in sorted(topk32.keys()):
            d = topk32[drv]
            scales = d['scales']
            K_list = d['K_list']
            K_max_idx = int(np.argmax(K_list))
            readouts = [str(x) for x in d['readouts']]
            log('  ' + DRIVE_LABEL_32.get(drv, drv) + ': '
                + 'scales=' + str(list(scales))
                + ' K_max=' + str(int(K_list[K_max_idx])))
            for si, P in enumerate(scales):
                vals = [(ri, str(d['readouts'][ri]),
                         float(np.nanmean(d['bal_acc'][si, K_max_idx, ri])))
                        for ri in range(len(readouts))]
                for ri, ro, v in vals:
                    log(f'    P={int(P):2d}  {ro:7s}: {v:.4f}')

    # ---- Smallest K to reach threshold per (drive × scale × readout) --------
    # For each (drive, P, readout), the smallest K at which mean bal_acc first
    # crosses each threshold T. Reports the minimum compute (= 2K FLOPs/predict)
    # that the SWR readout needs to match a given accuracy target.
    log('\n' + '-'*78)
    log('Smallest K for which mean bal_acc ≥ T  (per drive × scale × readout)')
    log('-'*78)
    THR = [0.80, 0.90, 0.95]
    smallest_K_rows = []
    for grid_size, topk in ((16, topk16), (32, topk32)):
        for drv in sorted(topk.keys()):
            d = topk[drv]
            scales = d['scales']; K_list = d['K_list']
            readouts = [str(x) for x in d['readouts']]
            for si, P in enumerate(scales):
                for ri, ro in enumerate(readouts):
                    for T in THR:
                        ba_K = np.array([
                            float(np.nanmean(d['bal_acc'][si, ki, ri]))
                            for ki in range(len(K_list))])
                        hits = np.where(ba_K >= T)[0]
                        if len(hits) > 0:
                            K_star = int(K_list[hits[0]])
                            ba_at = float(ba_K[hits[0]])
                        else:
                            K_star = None; ba_at = float(np.max(ba_K))
                        smallest_K_rows.append(dict(
                            grid=grid_size, drive=drv, P=int(P),
                            readout=ro, T=T,
                            K_star=K_star, bal_at_K_star=ba_at))

    with open(os.path.join(DATA, 'smallest_K_per_threshold.csv'), 'w',
              encoding='utf-8') as f:
        cols = ['grid', 'drive', 'P', 'readout', 'T', 'K_star', 'bal_at_K_star']
        f.write(','.join(cols) + '\n')
        for r in smallest_K_rows:
            f.write(','.join(('' if r[c] is None else str(r[c]))
                             for c in cols) + '\n')

    # ---- MLP analysis (unchanged) -------------------------------------------
    if mlp is not None:
        log('\n' + '-'*78)
        log('MLP-on-coords summary')
        log('-'*78)
        task_grid = mlp['task_grid'].astype(int)
        task_P = mlp['task_P'].astype(int)
        mean_final = mlp['mean_final']
        for ti in range(len(task_P)):
            g = int(task_grid[ti]); P = int(task_P[ti])
            flat = mean_final[ti]
            di_w, wi_w = np.unravel_index(np.nanargmax(flat), flat.shape)
            log(f'  g{g}P={P:2d}  best={float(flat[di_w,wi_w]):.4f}  '
                f'd={DEPTHS[di_w]} w={WIDTHS[wi_w]}  '
                f'params={int(mlp["params"][di_w,wi_w])}')

        # For each (grid, P), find the smallest-train-FLOPs MLP architecture
        # that matches the BEST SVM accuracy (across drives x readouts at K_max).
        # This is the "MLP needs N more compute to match the substrate"
        # comparison underlying the bottom table of Supplementary Fig. 7.
        log('\n  Smallest matching MLP per (grid, P, threshold) — '
            'target = best SVM at K_max:')
        sm_rows = []
        for ti in range(len(task_P)):
            g = int(task_grid[ti]); P = int(task_P[ti])
            # Per (g, P): the SVM target = max bal_acc over all drives and
            # readouts at K=K_max (the substrate's best shot at this task).
            T = None
            topk = topk16 if g == 16 else topk32
            for drv, d in topk.items():
                if P not in d['scales']: continue
                si = int(np.where(d['scales'] == P)[0][0])
                K_max_idx = int(np.argmax(d['K_list']))
                for ri in range(len(d['readouts'])):
                    v = float(np.nanmean(d['bal_acc'][si, K_max_idx, ri]))
                    if T is None or v > T:
                        T = v
            if T is None:
                continue
            flat = mean_final[ti]
            mask = flat >= T              # archs that reach the SVM target.
            if not np.any(mask):
                log(f'  g{g}P={P:2d}  SVM_target={T:.4f}: NO MLP matches '
                    f'(MLP max {float(np.nanmax(flat)):.4f})')
                sm_rows.append(dict(grid=g, P=P, target=T, matched=False))
                continue
            fwd = mlp['fwd_flops']
            cv = mlp['mean_curves'][ti]
            n_tr = int(mlp['n_train_per_task'][ti])
            f_train = np.full(flat.shape, np.inf)
            e_match = np.full(flat.shape, np.nan)
            for di in range(len(DEPTHS)):
                for wi in range(len(WIDTHS)):
                    if not mask[di, wi]: continue
                    c = cv[di, wi]
                    # First epoch the curve crosses the SVM target -- the
                    # arch's minimum training cost to match the substrate.
                    hits = np.where(c[:, 1] >= T)[0]
                    if len(hits) == 0: continue
                    e = float(c[hits[0], 0])
                    if e <= 0: e = 1.0
                    # Training FLOPs = 3 (fwd + bwd + update) * F_fwd * n_train * epochs.
                    f_train[di, wi] = 3.0 * int(fwd[di, wi]) * n_tr * e
                    e_match[di, wi] = e
            # Winner = arch with minimum training FLOPs that reaches the target.
            di_w, wi_w = np.unravel_index(int(np.argmin(f_train)),
                                          f_train.shape)
            log(f'  g{g}P={P:2d}  SVM_target={T:.4f}: '
                f'MLP d={DEPTHS[di_w]} w={WIDTHS[wi_w]}  '
                f'F_T={float(f_train[di_w,wi_w]):.3e}  '
                f'E={float(e_match[di_w,wi_w]):.0f} ep')
            sm_rows.append(dict(grid=g, P=P, target=T, matched=True,
                                best_d=DEPTHS[di_w], best_w=WIDTHS[wi_w],
                                F_train_match=float(f_train[di_w, wi_w]),
                                epochs_to_match=float(e_match[di_w, wi_w])))

        with open(os.path.join(DATA, 'smallest_matching_mlp.csv'), 'w',
                  encoding='utf-8') as f:
            cols = ['grid', 'P', 'target', 'matched', 'best_d', 'best_w',
                    'F_train_match', 'epochs_to_match']
            f.write(','.join(cols) + '\n')
            for r in sm_rows:
                f.write(','.join(
                    ('' if r.get(c) is None else str(r.get(c, '')))
                    for c in cols) + '\n')

    # ---- CNN analysis -------------------------------------------------------
    if cnn is not None:
        log('\n' + '-'*78)
        log('CNN-on-coords summary')
        log('-'*78)
        task_grid = cnn['task_grid'].astype(int)
        task_P    = cnn['task_P'].astype(int)
        mean_final = cnn['mean_final']
        for ti in range(len(task_P)):
            g = int(task_grid[ti]); P = int(task_P[ti])
            flat = mean_final[ti]
            di_w, wi_w = np.unravel_index(np.nanargmax(flat), flat.shape)
            log(f'  g{g}P={P:2d}  best={float(flat[di_w,wi_w]):.4f}  '
                f'd={DEPTHS_CNN[di_w]} w={WIDTHS_CNN[wi_w]}  '
                f'params={int(cnn["params"][di_w,wi_w])}  '
                f'F/pix={int(cnn["fwd_flops_per_pixel"][di_w,wi_w])}')

        # Smallest matching CNN per (g, P, threshold = best SVM ba @ K_max)
        log('\n  Smallest matching CNN per (grid, P) — target = best SVM at K_max:')
        sm_rows = []
        for ti in range(len(task_P)):
            g = int(task_grid[ti]); P = int(task_P[ti])
            T = None
            topk = topk16 if g == 16 else topk32
            for drv, d in topk.items():
                if P not in d['scales']: continue
                si = int(np.where(d['scales'] == P)[0][0])
                K_max_idx = int(np.argmax(d['K_list']))
                for ri in range(len(d['readouts'])):
                    v = float(np.nanmean(d['bal_acc'][si, K_max_idx, ri]))
                    if T is None or v > T:
                        T = v
            if T is None:
                continue
            flat = mean_final[ti]
            mask = flat >= T
            if not np.any(mask):
                log(f'  g{g}P={P:2d}  SVM_target={T:.4f}: NO CNN matches '
                    f'(CNN max {float(np.nanmax(flat)):.4f})')
                sm_rows.append(dict(grid=g, P=P, target=T, matched=False))
                continue
            fwd_img = cnn['fwd_flops_img_per_task'][ti]
            cv = cnn['mean_curves'][ti]
            f_train = np.full(flat.shape, np.inf)
            e_match = np.full(flat.shape, np.nan)
            for di in range(len(DEPTHS_CNN)):
                for wi in range(len(WIDTHS_CNN)):
                    if not mask[di, wi]: continue
                    c = cv[di, wi]
                    hits = np.where(c[:, 1] >= T)[0]
                    if len(hits) == 0: continue
                    e = float(c[hits[0], 0])
                    if e <= 0: e = 1.0
                    # CNN training cost = 3 * F_img * epochs (full-image fwd/step)
                    f_train[di, wi] = 3.0 * int(fwd_img[di, wi]) * e
                    e_match[di, wi] = e
            di_w, wi_w = np.unravel_index(int(np.argmin(f_train)),
                                          f_train.shape)
            log(f'  g{g}P={P:2d}  SVM_target={T:.4f}: '
                f'CNN d={DEPTHS_CNN[di_w]} w={WIDTHS_CNN[wi_w]}  '
                f'F_T={float(f_train[di_w,wi_w]):.3e}  '
                f'E={float(e_match[di_w,wi_w]):.0f} ep')
            sm_rows.append(dict(grid=g, P=P, target=T, matched=True,
                                best_d=DEPTHS_CNN[di_w],
                                best_w=WIDTHS_CNN[wi_w],
                                F_train_match=float(f_train[di_w, wi_w]),
                                epochs_to_match=float(e_match[di_w, wi_w])))

        with open(os.path.join(DATA, 'smallest_matching_cnn.csv'), 'w',
                  encoding='utf-8') as f:
            cols = ['grid', 'P', 'target', 'matched', 'best_d', 'best_w',
                    'F_train_match', 'epochs_to_match']
            f.write(','.join(cols) + '\n')
            for r in sm_rows:
                f.write(','.join(
                    ('' if r.get(c) is None else str(r.get(c, '')))
                    for c in cols) + '\n')

        # Side-by-side smallest-matching MLP vs CNN at the same SVM target
        log('\n  Side-by-side smallest-matching MLP vs CNN '
            '(train-FLOPs to SVM target):')
        log('  g  P    SVM_t   '
            'MLP_F_train   MLP_d/w        '
            'CNN_F_train   CNN_d/w     ratio (MLP/CNN)')
        side_rows = []
        if mlp is not None:
            mlp_tg = mlp['task_grid'].astype(int)
            mlp_tP = mlp['task_P'].astype(int)
            mlp_idx = {(int(mlp_tg[i]), int(mlp_tP[i])): i
                       for i in range(len(mlp_tP))}
            for ti in range(len(task_P)):
                g = int(task_grid[ti]); P = int(task_P[ti])
                # CNN side
                cnn_row = next((r for r in sm_rows
                                if r['grid'] == g and r['P'] == P), None)
                if cnn_row is None or not cnn_row.get('matched', False):
                    continue
                T = cnn_row['target']
                cnn_F = cnn_row['F_train_match']
                # MLP side
                mi = mlp_idx.get((g, P))
                if mi is None: continue
                m_flat = mlp['mean_final'][mi]
                m_mask = m_flat >= T
                if not np.any(m_mask):
                    log(f'  {g:2d} {P:2d}  {T:.3f}  '
                        f'(MLP did not reach target)  '
                        f'CNN={cnn_F:.2e} d={cnn_row["best_d"]} '
                        f'w={cnn_row["best_w"]}')
                    side_rows.append(dict(
                        grid=g, P=P, target=T, mlp_matched=False,
                        cnn_F=cnn_F, cnn_d=cnn_row['best_d'],
                        cnn_w=cnn_row['best_w']))
                    continue
                m_fwd = mlp['fwd_flops']
                m_cv = mlp['mean_curves'][mi]
                m_n_tr = int(mlp['n_train_per_task'][mi])
                m_f_train = np.full(m_flat.shape, np.inf)
                m_e_match = np.full(m_flat.shape, np.nan)
                for di in range(len(DEPTHS)):
                    for wi in range(len(WIDTHS)):
                        if not m_mask[di, wi]: continue
                        c = m_cv[di, wi]
                        hits = np.where(c[:, 1] >= T)[0]
                        if len(hits) == 0: continue
                        e = float(c[hits[0], 0])
                        if e <= 0: e = 1.0
                        m_f_train[di, wi] = (3.0 * int(m_fwd[di, wi])
                                             * m_n_tr * e)
                        m_e_match[di, wi] = e
                di_m, wi_m = np.unravel_index(int(np.argmin(m_f_train)),
                                              m_f_train.shape)
                mlp_F = float(m_f_train[di_m, wi_m])
                ratio = mlp_F / cnn_F if cnn_F > 0 else float('nan')
                log(f'  {g:2d} {P:2d}  {T:.3f}  '
                    f'{mlp_F:.2e}  d={DEPTHS[di_m]} w={WIDTHS[wi_m]:3d}    '
                    f'{cnn_F:.2e}  d={cnn_row["best_d"]} '
                    f'w={cnn_row["best_w"]:3d}   {ratio:.2f}x')
                side_rows.append(dict(
                    grid=g, P=P, target=T, mlp_matched=True,
                    mlp_F=mlp_F, mlp_d=DEPTHS[di_m], mlp_w=WIDTHS[wi_m],
                    cnn_F=cnn_F, cnn_d=cnn_row['best_d'],
                    cnn_w=cnn_row['best_w'], ratio_mlp_over_cnn=ratio))

        with open(os.path.join(DATA, 'mlp_vs_cnn_side_by_side.csv'), 'w',
                  encoding='utf-8') as f:
            cols = ['grid', 'P', 'target', 'mlp_matched',
                    'mlp_F', 'mlp_d', 'mlp_w',
                    'cnn_F', 'cnn_d', 'cnn_w', 'ratio_mlp_over_cnn']
            f.write(','.join(cols) + '\n')
            for r in side_rows:
                f.write(','.join(
                    ('' if r.get(c) is None else str(r.get(c, '')))
                    for c in cols) + '\n')

    # summary json
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'topk_16_drives': list(topk16.keys()),
        'topk_32_drives': list(topk32.keys()),
        'n_summary_rows': len(summary_rows),
        'mlp_loaded': mlp is not None,
        'cnn_loaded': cnn is not None,
    }
    with open(os.path.join(DATA, 'analyse_summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    log(f'\nWrote topk_summary.csv, topk_winner_matrix.csv, '
        f'wavelength_match_table.csv, smallest_K_per_threshold.csv, '
        f'analyse_summary.json')


if __name__ == '__main__':
    main()
