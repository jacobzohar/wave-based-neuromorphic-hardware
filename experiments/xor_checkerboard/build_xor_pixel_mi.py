"""Per-pixel x per-frame mutual information on XOR labels.

For one (drive, grid) cube: compute I(m_z[i,j,t] ; Y_XOR(P)) for each scale P
in the relevant scale-set. Output one packed (n_scales, T, H, W) f32 cube
per drive, plus a summary npz with per-scale max-time map + scalars.

Uses a 5-bin equal-frequency MI estimator with Miller-Madow correction, in bits.

CLI:
    python build_xor_pixel_mi.py --cube nlt4_15mT200_mz_cube.npy --grid 16 \
        --drive 15mT200 --xor_ds /path/to/xor_datasets.npz --out_dir /path

Outputs (in --out_dir):
    pixel_mi_xor_cube_{grid}_{drive}.npy        (n_scales, T, H, W) f32
    pixel_mi_xor_summary_{grid}_{drive}.npz     per-scale max-t map + scalars
"""
import os, sys, time, argparse
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cube', required=True)
    p.add_argument('--grid', type=int, required=True,
                   help='input encoding grid side; pre-set datasets use 16 or 32')
    p.add_argument('--drive', required=True)
    p.add_argument('--xor_ds', required=True)
    p.add_argument('--out_dir', required=True)
    p.add_argument('--k_bins', type=int, default=5)
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # Load the substrate response cube. mmap_mode='r' keeps the multi-GB array
    # on disk; we pull the whole thing into RAM below for fast quantile + bin.
    print(f'[load] cube {a.cube}')
    cube = np.load(a.cube, mmap_mode='r')                  # (N_pix, T, H, W)
    N_PIX, T, H, W = cube.shape
    N_CELLS = T * H * W
    assert N_PIX == a.grid * a.grid, f'cube N_pix {N_PIX} != {a.grid}^2'
    print(f'  shape={cube.shape}, T*H*W={N_CELLS:,}')

    # ---- Bin m_z once (per-cell K-bin equal-frequency) ----------------------
    # For each voxel (t, y, x) independently, compute the K-1 quantile
    # thresholds that split its N_pix samples into K equal-frequency bins,
    # then encode every sample's m_z as an integer in 0..K-1. Equal-frequency
    # (not equal-width) bins maximise marginal entropy on x, which is the
    # standard recipe for histogram-based MI estimation.
    t0 = time.time()
    print(f'[bin] K_BINS={a.k_bins}, computing per-cell thresholds ...')
    flat = np.asarray(cube, dtype=np.float32).reshape(N_PIX, -1)   # (N_pix, n_cells)
    qs = np.arange(1, a.k_bins) / a.k_bins
    thr = np.quantile(flat, qs, axis=0)                            # (K-1, n_cells)
    codes = np.zeros((N_PIX, N_CELLS), dtype=np.int8)
    for k in range(a.k_bins - 1):
        codes += (flat >= thr[k][None, :]).astype(np.int8)         # threshold-stack -> bin idx
    del flat
    print(f'  codes {codes.shape} ; {time.time()-t0:.0f}s')

    # ---- Load XOR labels ----------------------------------------------------
    # One binary label vector per XOR period P (scale-set built by make_dataset.py).
    ds = np.load(a.xor_ds)
    scales = ds[f'scales_{a.grid}']
    n_scales = len(scales)
    print(f'[xor] scales for grid {a.grid}: {list(scales)}')

    # ---- MI per scale -------------------------------------------------------
    # For each scale P:
    #   1. Build a joint count tensor jc[bin, class, voxel] = #samples with
    #      (code[voxel] == bin) AND (label == class).
    #   2. Normalise to pxy and derive marginals px, py.
    #   3. MI = sum_{bin, class} pxy * log(pxy / (px * py)) / ln(2), in bits.
    #   4. Subtract the Miller-Madow finite-sample bias correction
    #         bias = (m_xy - m_x - m_y + 1) / (2 N ln 2)
    #      where m_* are counts of NONZERO cells in the joint / marginals -
    #      the standard small-sample correction for plug-in MI estimators.
    #   5. Clip negatives (post-bias) to zero.
    LN2 = np.log(2.0)
    MI_all = np.zeros((n_scales, N_CELLS), dtype=np.float32)
    for si, P in enumerate(scales):
        y = ds[f'labels_{a.grid}_P{int(P)}'].astype(np.int64)
        jc = np.zeros((a.k_bins, 2, N_CELLS), dtype=np.float64)
        for c in (0, 1):
            sub = codes[y == c]
            for k in range(a.k_bins):
                jc[k, c] = (sub == k).sum(axis=0)
        N = jc.sum(axis=(0, 1))                  # samples per voxel (== N_pix everywhere)
        pxy = jc / N
        px = pxy.sum(axis=1); py = pxy.sum(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            term = (pxy * (np.log(pxy) - np.log(px[:, None, :])
                           - np.log(py[None, :, :]))) / LN2
        mi = np.nansum(term, axis=(0, 1))
        m_xy = (jc > 0).sum(axis=(0, 1))          # # non-empty (bin, class) cells, per voxel
        m_x  = (px > 0).sum(axis=0)               # # non-empty bins, per voxel
        m_y  = (py > 0).sum(axis=0)               # # non-empty classes, per voxel (1 or 2)
        bias = (m_xy - m_x - m_y + 1) / (2.0 * N * LN2)
        MI_all[si] = np.clip(mi - bias, 0.0, None).astype(np.float32)
        print(f'  P={int(P):2d}: '
              f'max={MI_all[si].max():.3f}  mean={MI_all[si].mean():.4f}  '
              f'frac>0.05b={(MI_all[si]>0.05).mean():.3f}  '
              f'({time.time()-t0:.0f}s)')

    del codes
    MI = MI_all.reshape(n_scales, T, H, W)

    out_cube = os.path.join(
        a.out_dir, f'pixel_mi_xor_cube_{a.grid}_{a.drive}.npy')
    np.save(out_cube, MI)
    print(f'[save] {out_cube} ({os.path.getsize(out_cube)/1e6:.1f} MB)')

    # ---- summary -------------------------------------------------------------
    mi_maxt = MI.max(axis=1)                              # (n_scales, H, W)
    mi_max = MI.reshape(n_scales, -1).max(axis=1)
    mi_mean = MI.reshape(n_scales, -1).mean(axis=1)
    mi_fracpos = (MI.reshape(n_scales, -1) > 0.05).mean(axis=1)
    mi_t_total = MI.sum(axis=(2, 3))                      # (n_scales, T)
    mi_t_max = MI.max(axis=(2, 3))                        # (n_scales, T)
    argmax_vox = np.array([
        np.unravel_index(int(np.argmax(MI[si])), (T, H, W))
        for si in range(n_scales)
    ])                                                    # (n_scales, 3)
    peak_frame = argmax_vox[:, 0]

    out_summary = os.path.join(
        a.out_dir, f'pixel_mi_xor_summary_{a.grid}_{a.drive}.npz')
    np.savez_compressed(
        out_summary,
        scales=np.asarray(scales, dtype=np.int64),
        drive=np.asarray(a.drive),
        grid=np.int64(a.grid),
        mi_maxt=mi_maxt.astype(np.float32),
        mi_max=mi_max,
        mi_mean=mi_mean,
        mi_fracpos=mi_fracpos,
        mi_t_total=mi_t_total.astype(np.float32),
        mi_t_max=mi_t_max.astype(np.float32),
        argmax_vox=argmax_vox,
        peak_frame=peak_frame,
        gmax=np.array([float(MI.max())]),
        n_fr=np.array([T]), h=np.array([H]), w=np.array([W]),
        cube=np.array([a.cube]),
    )
    print(f'[save] {out_summary}')
    print(f'[done] {time.time()-t0:.0f}s ; peak frames per scale: '
          + ' '.join(f'P{int(scales[si])}:{int(peak_frame[si])}'
                     for si in range(n_scales)))


if __name__ == '__main__':
    main()
