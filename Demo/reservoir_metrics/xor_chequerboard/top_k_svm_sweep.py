"""Top-K (pixel, frame) MI-feature LinearSVC sweep on XOR labels.

Pipeline:
  - labels = XOR-scale labels
  - classifier = LinearSVC (squared_hinge, dual='auto', L2) at fixed C
  - readouts = {analog, binary, ternary}

For each (drive, grid, scale, K, readout):
  rank pixel-MI cube voxels descending → top K (t, y, x)
  feats = m_z_cube[:, t, y, x]   # (N_pix, K)
  pipe = StandardScaler → [sign or ternary] → LinearSVC(C=1.0)
  bal_acc reported across RepeatedStratifiedKFold(5,3, random_state=0)
  flops_predict = 2 K (eq. 1)
  flops_fit ≈ 3 * n_iter * N_train * K (eq. 2, c=3)

CLI:
  python top_k_svm_sweep.py \
      --cube /path/to/nlt4_*_mz_cube.npy \
      --mi_cube /path/to/pixel_mi_xor_cube_{grid}_{drive}.npy \
      --xor_ds /path/to/xor_datasets.npz \
      --grid 16 --drive 15mT200 \
      --out data/topk_svm_16_15mT200.npz
"""
import os, sys, time, argparse, json
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)


# Ternary readout, applied AFTER StandardScaler: thresholds the standardised
# feature at +/- 1 standard deviation, mapping to {-1, 0, +1}. Models the
# 3-level quantised readout of a physical comparator with a deadband.
def _ternary(Z):
    out = np.zeros_like(Z)
    out[Z >  1.0] =  1.0
    out[Z < -1.0] = -1.0
    return out


def make_pipeline(readout, C, max_iter):
    # Build the readout chain: standardise -> optional quantiser -> LinearSVC.
    # `readout='analog'` skips the quantiser; 'binary' uses np.sign (sign of the
    # standardised feature); 'ternary' uses the deadband encoder above. The SVC
    # fits with squared-hinge loss (smooth gradient, well-behaved at high K),
    # L2 regularisation, fixed C, deterministic random_state.
    steps = [('sc', StandardScaler())]
    if readout == 'binary':
        steps.append(('tf', FunctionTransformer(np.sign)))
    elif readout == 'ternary':
        steps.append(('tf', FunctionTransformer(_ternary)))
    steps.append(('clf', LinearSVC(C=C, loss='squared_hinge', dual='auto',
                                   max_iter=max_iter, random_state=0)))
    return Pipeline(steps)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cube', required=True)
    p.add_argument('--mi_cube', required=True)
    p.add_argument('--xor_ds', required=True)
    p.add_argument('--grid', type=int, required=True,
                   help='input encoding grid side; pre-set datasets use 16 or 32')
    p.add_argument('--drive', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--K_list', nargs='+', type=int,
                   default=[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    p.add_argument('--readouts', nargs='+', default=['analog', 'binary', 'ternary'])
    p.add_argument('--C', type=float, default=1.0)
    p.add_argument('--max_iter', type=int, default=5000)
    p.add_argument('--n_splits', type=int, default=5)
    p.add_argument('--n_repeats', type=int, default=3)
    p.add_argument('--rs', type=int, default=0)
    return p.parse_args()


def progress(i, n, t0, label=''):
    pct = (i + 1) / n
    el = time.time() - t0
    eta = el / pct * (1 - pct) if pct > 0 else 0
    bar = ('#' * int(30 * pct)).ljust(30, '.')
    print(f'\r[{bar}] {i+1}/{n} {pct*100:5.1f}% | '
          f'{el/60:5.1f}m elapsed | ETA {eta/60:5.1f}m | {label}',
          end='', flush=True)
    if i + 1 == n:
        print()


def main():
    a = parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    # Load the substrate response cube. mmap_mode='r' keeps the multi-GB array
    # on disk; only the (pixel, frame) coords picked by top-K are pulled into RAM.
    print(f'[load] m_z cube: {a.cube}')
    mz = np.load(a.cube, mmap_mode='r')               # (N_pix, T, H, W)
    N_pix, T, H, W = mz.shape
    assert N_pix == a.grid * a.grid

    # Load the per-voxel MI cube (one MI slab per XOR scale). Used to RANK
    # substrate voxels by their informativeness for each label set.
    print(f'[load] MI cube : {a.mi_cube}')
    mi = np.load(a.mi_cube)                           # (n_scales, T, H, W)
    n_scales = mi.shape[0]
    assert mi.shape[1:] == (T, H, W), \
        f'mi {mi.shape} mismatch mz {mz.shape}'

    # XOR labels: one binary label vector per scale P, indexed by sample.
    ds = np.load(a.xor_ds)
    scales = ds[f'scales_{a.grid}'].astype(int)
    assert len(scales) == n_scales, \
        f'scales mismatch: ds={len(scales)} vs mi cube={n_scales}'

    K_max = max(a.K_list)
    n_part = a.n_splits * a.n_repeats               # total CV folds = splits * repeats
    n_K = len(a.K_list)
    n_ro = len(a.readouts)

    print(f'  N_pix={N_pix}, T={T}, H={H}, W={W}, n_scales={n_scales}')
    print(f'  scales={list(scales)} K_list={a.K_list} readouts={a.readouts}')
    print(f'  CV: RepeatedStratifiedKFold(n_splits={a.n_splits}, '
          f'n_repeats={a.n_repeats}, rs={a.rs}) = {n_part} folds')

    bal_acc = np.full((n_scales, n_K, n_ro, n_part), np.nan, dtype=np.float32)
    n_iter_arr = np.zeros((n_scales, n_K, n_ro, n_part), dtype=np.int32)
    n_train_arr = np.zeros(n_part, dtype=np.int32)
    n_test_arr = np.zeros(n_part, dtype=np.int32)
    top_coords = np.zeros((n_scales, K_max, 3), dtype=np.int64)
    mi_at_topK = np.zeros((n_scales, K_max), dtype=np.float32)

    total_cells = n_scales * n_K * n_ro * n_part
    t0 = time.time()
    done = 0

    for si, P in enumerate(scales):
        # ---- Rank MI voxels descending for this XOR scale -------------------
        # flat_mi[i] is the MI of voxel (t, y, x) with the binary XOR label at
        # period P. Sorting descending and taking the first K_max gives the
        # K_max most-informative substrate voxels for this scale.
        mi_vol = mi[si]                               # (T, H, W)
        flat_mi = mi_vol.ravel()
        order = np.argsort(-flat_mi, kind='stable')
        top_K_idx = order[:K_max]
        t_idx, y_idx, x_idx = np.unravel_index(top_K_idx, mi_vol.shape)
        top_coords[si] = np.stack([t_idx, y_idx, x_idx], axis=1)
        mi_at_topK[si] = flat_mi[top_K_idx]

        # ---- Extract feature matrix once at K_max ---------------------------
        # Fancy-index the mmapped cube along the (T, H, W) axes for every
        # sample at once -> (N_pix, K_max) float32 dense array. Doing this once
        # per scale (instead of once per K) lets the K-sweep just slice columns.
        t_extract = time.time()
        feats_all = mz[:, t_idx, y_idx, x_idx].astype(np.float32)
        feats_all = np.ascontiguousarray(feats_all)
        print(f'\n[scale {si+1}/{n_scales}] P={int(P)}  '
              f'MI max@topK={float(mi_at_topK[si][0]):.4f}b  '
              f'MI@K={K_max}={float(mi_at_topK[si][-1]):.4f}b  '
              f'feat extract={time.time()-t_extract:.1f}s')

        # ---- CV partition ---------------------------------------------------
        # The same RepeatedStratifiedKFold folds are used for every K and every
        # readout, so the (K, readout) sweep is apples-to-apples: differences
        # in bal_acc reflect K or readout, not the partition.
        y = ds[f'labels_{a.grid}_P{int(P)}'].astype(np.int64)
        rskf = RepeatedStratifiedKFold(n_splits=a.n_splits,
                                       n_repeats=a.n_repeats,
                                       random_state=a.rs)
        folds = list(rskf.split(feats_all, y))
        assert len(folds) == n_part
        for fi, (tr, te) in enumerate(folds):
            n_train_arr[fi] = len(tr)
            n_test_arr[fi] = len(te)

        # ---- Inner sweep: K x readout x fold --------------------------------
        # For each K (top-K by MI), each readout (analog / binary / ternary)
        # and each CV fold, fit a fresh pipeline on train, score balanced
        # accuracy on test, and record the SVC's convergence iteration count
        # (used downstream for the per-fit FLOPs estimate below).
        for ki, K in enumerate(a.K_list):
            feats_K = feats_all[:, :K]
            for ri, readout in enumerate(a.readouts):
                bas = []
                its = []
                for fi, (tr, te) in enumerate(folds):
                    pipe = make_pipeline(readout, a.C, a.max_iter)
                    pipe.fit(feats_K[tr], y[tr])
                    pr = pipe.predict(feats_K[te])
                    bas.append(balanced_accuracy_score(y[te], pr))
                    ni = pipe.named_steps['clf'].n_iter_
                    its.append(int(np.asarray(ni).max()))
                    done += 1
                    progress(done - 1, total_cells, t0,
                             label=f'P={int(P):2d} K={K:4d} '
                                   f'{readout[:3]} f={fi:2d} ba={bas[-1]:.3f}')
                bal_acc[si, ki, ri] = np.array(bas, dtype=np.float32)
                n_iter_arr[si, ki, ri] = np.array(its, dtype=np.int32)

    # ---- FLOPs accounting --------------------------------------------------
    # Inference cost per sample: 2K mul-adds for the linear SVC dot product.
    # Training cost per fit: ~3 * n_iter * n_train * K (LinearSVC's coordinate-
    # descent / dual gradient inner loop; coefficient ~3 from sklearn's source).
    K_arr = np.asarray(a.K_list, dtype=np.int64)
    flops_predict_sample = (2 * K_arr).astype(np.int64)      # (n_K,)
    flops_fit = np.zeros((n_scales, n_K, n_ro, n_part), dtype=np.float64)
    for ki, K in enumerate(K_arr):
        flops_fit[:, ki, :, :] = (3.0 * n_iter_arr[:, ki, :, :].astype(np.float64)
                                  * n_train_arr[None, None, :].astype(np.float64)
                                  * float(K))

    elapsed = time.time() - t0
    print(f'\n[done] {elapsed:.0f}s = {elapsed/60:.1f}m  ({elapsed/total_cells*1000:.0f} ms/cell)')

    np.savez_compressed(
        a.out,
        scales=np.asarray(scales, dtype=np.int64),
        K_list=K_arr,
        readouts=np.asarray(a.readouts),
        drive=np.asarray(a.drive),
        grid=np.int64(a.grid),
        H=np.int64(H), W=np.int64(W), T=np.int64(T),
        n_splits=np.int64(a.n_splits),
        n_repeats=np.int64(a.n_repeats),
        rs=np.int64(a.rs),
        C=np.float64(a.C),
        max_iter=np.int64(a.max_iter),
        bal_acc=bal_acc,
        n_iter=n_iter_arr,
        n_train=n_train_arr,
        n_test=n_test_arr,
        top_coords=top_coords,
        mi_at_topK=mi_at_topK,
        flops_predict_sample=flops_predict_sample,
        flops_fit=flops_fit,
    )
    print(f'[save] {a.out} ({os.path.getsize(a.out)/1024:.1f} kB)')

    # ---- summary -------------------------------------------------------------
    print(f'\n[summary] drive={a.drive}  grid={a.grid}x{a.grid}')
    print(f'  scale  K    readout    bal_acc        med_iter')
    for si, P in enumerate(scales):
        for ki, K in enumerate(a.K_list):
            for ri, ro in enumerate(a.readouts):
                ba = bal_acc[si, ki, ri]
                mu = float(np.nanmean(ba)); sd = float(np.nanstd(ba))
                it = int(np.median(n_iter_arr[si, ki, ri]))
                print(f'  P={int(P):2d}   K={int(K):4d} {ro:7s}    '
                      f'{mu:.4f} ± {sd:.4f}    {it}')


if __name__ == '__main__':
    main()
