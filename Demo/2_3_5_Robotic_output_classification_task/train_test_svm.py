"""train_test_svm.py — train + test linear SVMs on the obstacle-distance task.

Reproduces the manuscript's Figure 2e SVM-vs-WWRC-vs-SWRC comparison panels
from the bundled inputs:

  data/Train{2,3,5}X.csv         Safeer's distance-vector datasets
  data/Train{2,3,5}Y.csv         labels (2, 3, and 5-class respectively)
  data/training_frame3_results.csv  water-wave reservoir lab baseline
  data/swrc_features.npz         pre-extracted spin-wave-disk reservoir
                                 feature cache (m_z field at t = frame 100,
                                 z-scored and flattened to 2500-d per sample)

For each of the three output-class variants and each training-set size on
the sweep, the script:

  1. Trains a linear SVM on a random training subset of size f.
  2. Tests on the next 100 samples.
  3. Repeats N_SHUFFLES times to estimate mean + std.

Two feature spaces are evaluated separately:

  • Raw distance vector (6-d) — the SVM-baseline arm of Fig. 2e.
  • SWRC feature vector (2500-d, m_z field at frame 100) — the spin-wave
    reservoir readout arm.

The water-wave reservoir scores are loaded from the lab CSV and not
recomputed.

Outputs three figures into figures/:
  FinalResultsMain.png       SWRC alone, errorbar plot
  SWvsSVMW.png               SWRC vs Raw SVM
  WWvsSVMvsSW.png            full three-way comparison

Run:
    python train_test_svm.py                # full sweep (~5–10 min on a laptop)
    python train_test_svm.py --quick        # short sweep for a smoke test
    python train_test_svm.py --seed 1       # change RNG seed
"""
import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


HERE = os.path.dirname(os.path.abspath(__file__))


def load_inputs(data_dir):
    """Load distance-vector data, SWRC features, WWRC lab baseline."""
    X = np.zeros((3, 1000, 6), dtype=np.float64)
    Y = np.zeros((3, 1000), dtype=np.float64)
    sizes = (1000, 999, 1000)  # 3-class only has 999 samples (n_total % 3 == 0)
    for vi, k in enumerate((2, 3, 5)):
        n = sizes[vi]
        X[vi, :n] = np.loadtxt(os.path.join(data_dir, f"Train{k}X.csv"),
                               delimiter=",")
        Y[vi, :n] = np.loadtxt(os.path.join(data_dir, f"Train{k}Y.csv"))

    psi = np.load(os.path.join(data_dir, "swrc_features.npz"))["psi"]
    assert psi.shape == (3, 1000, 2500), f"unexpected SWRC shape {psi.shape}"

    ww = np.loadtxt(os.path.join(data_dir, "training_frame3_results.csv"),
                    delimiter=",", skiprows=1)
    return X, Y, psi, ww


def sweep(features, labels, fs, n_shuffles, rng, label):
    """One feature-space sweep.

    features : (N, D) sample matrix.
    labels   : (N,) integer labels.
    fs       : list of training-set sizes to evaluate.
    n_shuffles : repetitions to average over.

    Returns scores[n_shuffles, len(fs)] of test accuracies in percent.
    """
    n = features.shape[0]
    scores = np.zeros((n_shuffles, len(fs)))
    t0 = time.time()
    for k in range(n_shuffles):
        # Reshuffle until the first fs[0] training samples span >1 class
        # (otherwise SVC(kernel='linear') refuses to fit).
        while True:
            perm = rng.permutation(n)
            if len(np.unique(labels[perm[:fs[0]]])) > 1:
                break
        Xp, Yp = features[perm], labels[perm]
        for j, f in enumerate(fs):
            model = SVC(kernel="linear", probability=False)
            model.fit(Xp[:f], Yp[:f])
            pred = model.predict(Xp[f:f + 100])
            scores[k, j] = 100.0 * (pred == Yp[f:f + 100]).mean()
        if (k + 1) % max(1, n_shuffles // 10) == 0 or k == n_shuffles - 1:
            el = time.time() - t0
            eta = el / (k + 1) * (n_shuffles - 1 - k)
            print(f"  [{label}] shuffle {k+1:4d}/{n_shuffles}  "
                  f"elapsed {el:5.1f}s  ETA {eta:5.1f}s", flush=True)
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(HERE, "data"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "figures"))
    ap.add_argument("--seed", type=int, default=0,
                    help="numpy RNG seed (default 0)")
    ap.add_argument("--n-shuffles", type=int, default=500,
                    help="random splits per (variant, training-size) cell "
                         "(default 500)")
    ap.add_argument("--quick", action="store_true",
                    help="--n-shuffles=20 for a fast smoke run")
    args = ap.parse_args()
    if args.quick:
        args.n_shuffles = 20

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # --- 1. Load inputs ------------------------------------------------------
    print("[load] reading inputs")
    Xdata, Ydata, psi, ww = load_inputs(args.data_dir)
    print(f"  Xdata    : {Xdata.shape}    (3 variants, 1000 samples, 6-d distance)")
    print(f"  Ydata    : {Ydata.shape}")
    print(f"  psi      : {psi.shape}      (SWRC features, 2500-d per sample)")
    print(f"  ww (lab) : {ww.shape}       (training_size, score2, score3, score5)")

    # Sweep training-set sizes, matching the published plot (skip the tail to
    # match the manuscript axis range).
    fs = [int(f) for f in ww[1:-7, 0]]
    print(f"  training-set sweep: {fs}")

    # --- 2. Quick 800/200 sanity check  (mirrors notebook cell 39) -----------
    print("\n[sanity] 800/200 linear-SVC split per variant (raw distance vec):")
    for vi, k in enumerate((2, 3, 5)):
        m = SVC(kernel="linear", probability=False)
        m.fit(Xdata[vi, :800], Ydata[vi, :800])
        a = m.score(Xdata[vi, 800:], Ydata[vi, 800:])
        print(f"  {k}-class accuracy = {a:.4f}")

    # --- 3. Raw-SVM training-size sweep  (cell 45) ---------------------------
    print(f"\n[sweep raw SVM]   n_shuffles={args.n_shuffles}")
    raw_scores = np.zeros((3, args.n_shuffles, len(fs)))
    for vi, k in enumerate((2, 3, 5)):
        raw_scores[vi] = sweep(Xdata[vi], Ydata[vi], fs,
                               args.n_shuffles, rng, f"SVM {k}-class")
    raw_mean = raw_scores.mean(axis=1)        # mirrors `resultsesSVM` (mean %)

    # --- 4. SWRC training-size sweep  (cell 47) ------------------------------
    print(f"\n[sweep SWRC]     n_shuffles={args.n_shuffles}")
    sw_scores = np.zeros((3, args.n_shuffles, len(fs)))
    for vi, k in enumerate((2, 3, 5)):
        sw_scores[vi] = sweep(psi[vi], Ydata[vi], fs,
                              args.n_shuffles, rng, f"SWRC {k}-class")
    sw_mean = sw_scores.mean(axis=1)
    sw_std = sw_scores.std(axis=1)

    # --- 5. Save intermediate caches (alongside `figures/`) ------------------
    cache_dir = args.out_dir
    np.savetxt(os.path.join(cache_dir, "SWvsSVM_avs.csv"), sw_mean)
    np.savetxt(os.path.join(cache_dir, "SWvsSVM_std.csv"), sw_std)
    np.savetxt(os.path.join(cache_dir, "SWvsSVM_svm.csv"), raw_mean)

    # --- 6. Figure 1: FinalResultsMain.png  (cell 47) ------------------------
    print("\n[plot] FinalResultsMain.png")
    plt.figure(figsize=(8, 5))
    colours_main = ("red", "blue", "black")
    for vi, (k, c) in enumerate(zip((2, 3, 5), colours_main)):
        plt.errorbar(fs, sw_mean[vi], yerr=sw_std[vi], fmt="o-", capsize=2,
                     label=str(k), color=c)
    plt.ylabel("Recognition Rate (%)", fontsize=18)
    plt.xlabel("Size of training dataset", fontsize=18)
    plt.axhline(100, color="k", linestyle="--", linewidth=1)
    plt.ylim(15, 105)
    plt.legend(fontsize=14, title="No. of output decisions", title_fontsize=16)
    plt.savefig(os.path.join(args.out_dir, "FinalResultsMain.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # --- 7. Figure 2: SWvsSVMW.png  (cell 48) --------------------------------
    print("[plot] SWvsSVMW.png")
    plt.figure(figsize=(8, 5))
    raw_colours = ("lightcoral", "royalblue", "gray")
    sw_colours = ("red", "blue", "black")
    for vi, k in enumerate((2, 3, 5)):
        plt.plot(fs, raw_mean[vi], "o--", color=raw_colours[vi],
                 label=f"{k} (Raw SVM)")
        plt.plot(fs, sw_mean[vi], "s-", color=sw_colours[vi],
                 label=f"{k} (SWRC)")
    plt.ylabel("Recognition Rate (%)", fontsize=18)
    plt.xlabel("Size of training dataset", fontsize=18)
    plt.axhline(100, color="k", linestyle="--", linewidth=1)
    plt.ylim(15, 105)
    plt.legend(fontsize=14, title="No. of output decisions", title_fontsize=16)
    plt.savefig(os.path.join(args.out_dir, "SWvsSVMW.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # --- 8. Figure 3: WWvsSVMvsSW.png  (cell 50) -----------------------------
    print("[plot] WWvsSVMvsSW.png")
    ww_scores = ww[1:-7, 1:4] * 100.0           # columns: score2, score3, score5
    plt.figure(figsize=(8, 5))
    for vi, k in enumerate((2, 3, 5)):
        plt.plot(fs, raw_mean[vi], "o--", color=raw_colours[vi],
                 label=f"{k} (SVM)")
        plt.plot(fs, ww_scores[:, vi], "s-", color=sw_colours[vi],
                 label=f"{k} (WWRC)")
        plt.plot(fs, sw_mean[vi], "o--", color=sw_colours[vi],
                 label=f"{k} (SWRC)")
    plt.ylabel("Recognition Rate (%)", fontsize=18)
    plt.xlabel("Size of training dataset", fontsize=18)
    plt.axhline(100, color="k", linestyle="--", linewidth=1)
    plt.ylim(15, 105)
    plt.legend(fontsize=14, ncol=2,
               title="No. of output decisions", title_fontsize=16)
    plt.savefig(os.path.join(args.out_dir, "WWvsSVMvsSW.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nDone. Wrote figures to {args.out_dir}.")


if __name__ == "__main__":
    main()
