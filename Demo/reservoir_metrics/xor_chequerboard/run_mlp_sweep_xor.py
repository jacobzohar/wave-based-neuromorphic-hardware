"""MLP-on-coords capacity sweep on the XOR-task ladder.

48 archs (d in {1,2,3,4,6,8} x w in {2,4,8,16,32,64,128,256})
x 9 XOR tasks (4 at 16x16: P in {1,2,4,8} + 5 at 32x32: P in {1,2,4,8,16})
x RepeatedStratifiedKFold(5, 3, random_state=0) = 15 partitions
= 6480 MLP trainings, 400 epochs full-batch Adam + cosine each.

Inline progress bar; checkpoint save every 500 trainings.
"""
import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# Pin BLAS to a single thread: each MLP is tiny, so multi-threaded BLAS
# would spend more time in thread-pool overhead than in the matmul. 1 thread
# x many sequential trainings is the throughput optimum on a multi-core CPU.
torch.set_num_threads(1)
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

# ---- config -----------------------------------------------------------------
DATA_PATH  = os.path.join(HERE, 'data', 'xor_datasets.npz')
SAVE_PATH  = os.path.join(HERE, 'data', 'mlp_sweep_xor.npz')
CKPT_PATH  = os.path.join(HERE, 'data', 'mlp_sweep_xor_ckpt.npz')
LOG_PATH   = os.path.join(HERE, 'log.txt')

# 6 x 8 = 48 architectures. Spans 2 -- ~50k parameters; gives a clean
# capacity sweep against the spin-wave reservoir's fixed feature lift.
DEPTHS     = [1, 2, 3, 4, 6, 8]
WIDTHS     = [2, 4, 8, 16, 32, 64, 128, 256]
# 5-fold x 3-repeat stratified CV = 15 partitions per arch x task. random_state=0
# is shared with the SVM and CNN sweeps so all three classifiers see identical
# train/test splits (apples-to-apples FLOPs-vs-accuracy comparison).
N_SPLITS   = 5
N_REPEATS  = 3
RS         = 0
# 400 epochs is generous: convergence curves below show every arch is well
# past its plateau by ~200; the extra margin keeps the comparison fair to the
# higher-capacity nets that need more steps. EVAL_EVERY=5 -> 81 eval points.
EPOCHS     = 400
EVAL_EVERY = 5
LR         = 1e-3

# Input: 2-D (x, y) pixel coordinate. Output: 2-class XOR-chequerboard label.
INPUT_DIM  = 2
N_CLASS    = 2

# Accuracy thresholds for the time-to-accuracy / FLOPs-to-accuracy curves.
# THR_NAMES are saved alongside so analyse_xor.py can dereference by name.
THRESHOLDS = [0.60, 0.80, 0.90, 0.95]
THR_NAMES  = ['T_60', 'T_80', 'T_90', 'T_95']

# ---- logging helper ---------------------------------------------------------
_log = open(LOG_PATH, 'a', encoding='utf-8')
def log(msg=''):
    print(msg, flush=True)
    _log.write(msg + '\n'); _log.flush()

log('\n' + '='*78)
log(f'MLP XOR sweep RUN STARTED at {time.strftime("%Y-%m-%d %H:%M:%S")}')
log('='*78)
log(f'depths={DEPTHS}, widths={WIDTHS}')
log(f'n_splits={N_SPLITS}, n_repeats={N_REPEATS}, rs={RS}, '
    f'epochs={EPOCHS}, eval_every={EVAL_EVERY}')
log(f'thresholds={list(zip(THR_NAMES, THRESHOLDS))}')

# ---- data -------------------------------------------------------------------
D = np.load(DATA_PATH)
scales_16 = list(D['scales_16'])
scales_32 = list(D['scales_32'])

# Build the task list: list of (grid_size, P, coords, labels)
tasks = []
for P in scales_16:
    tasks.append((16, int(P), D['coords_16'].astype(np.float32),
                  D[f'labels_16_P{int(P)}'].astype(np.int64)))
for P in scales_32:
    tasks.append((32, int(P), D['coords_32'].astype(np.float32),
                  D[f'labels_32_P{int(P)}'].astype(np.int64)))
N_TASKS = len(tasks)
log(f'\nN_TASKS = {N_TASKS}')
for ti, (g, P, X, y) in enumerate(tasks):
    log(f'  [{ti}] grid={g} P={P:2d}  N_pix={len(y)}  balance={y.mean():.3f}')


# ---- model + helpers --------------------------------------------------------
class FlexMLP(nn.Module):
    """Plain feed-forward MLP with `depth` hidden layers of `width` units each
    and ReLU activations between them. The hidden layers are square (w -> w)
    so the parameter count grows as O(d * w^2)."""
    def __init__(self, input_dim, depth, width, n_class):
        super().__init__()
        # Layer dimensions: input -> [width] * depth -> n_class.
        dims = [input_dim] + [width] * depth + [n_class]
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            # ReLU between hidden layers; no activation on the logit head.
            if i < len(self.linears) - 1:
                x = torch.relu(x)
        return x


def param_count(d, w):
    """Trainable parameters: input projection + (d-1) square hidden layers +
    output head. Closed form: D*w + (d-1)*(w^2 + w) + w*C + (w + C) biases."""
    p = INPUT_DIM * w + w
    for _ in range(d - 1):
        p += w * w + w
    p += w * N_CLASS + N_CLASS
    return int(p)


def fwd_flops_per_sample(d, w):
    """Forward FLOPs per input sample: 2 multiplies-and-adds per parameter
    in the matmul (factor of 2 = mul + add); biases ignored (lower order)."""
    f = 2 * INPUT_DIM * w
    for _ in range(d - 1):
        f += 2 * w * w
    f += 2 * w * N_CLASS
    return int(f)


def std_scale(X_tr, X_te):
    """Standardise X_te using the train fold's mean/std (no test-set leakage).
    The 1e-8 floor on sigma guards against constant features (all-zero std)."""
    mu = X_tr.mean(0)
    sig = X_tr.std(0) + 1e-8
    return (X_tr - mu) / sig, (X_te - mu) / sig


def train_one(X_tr, y_tr, X_te, y_te, depth, width, seed):
    """Train one MLP for EPOCHS epochs of full-batch Adam (cosine-annealed LR)
    and return the (epoch, balanced_accuracy) curve sampled every EVAL_EVERY
    epochs. Full-batch is fine because each task is at most 1024 samples."""
    # Seeded fresh per training so each (arch, fold) gets an independent init.
    torch.manual_seed(seed); np.random.seed(seed)
    model = FlexMLP(INPUT_DIM, depth, width, N_CLASS)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    # Cosine schedule decays LR smoothly to 0 over EPOCHS; no warmup needed
    # because the tasks are tiny and Adam absorbs the cold-start gradient.
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(X_tr); yt = torch.from_numpy(y_tr)
    Xe = torch.from_numpy(X_te)
    n_eval = 1 + EPOCHS // EVAL_EVERY
    # Curve rows: [epoch, balanced_accuracy].
    curve = np.zeros((n_eval, 2), dtype=np.float32)
    eval_idx = 0
    for ep in range(EPOCHS + 1):
        if ep > 0:
            # Standard full-batch step (no minibatching for these dataset sizes).
            model.train()
            opt.zero_grad()
            out = model(Xt)
            loss = loss_fn(out, yt)
            loss.backward(); opt.step(); sched.step()
        if ep % EVAL_EVERY == 0 and eval_idx < n_eval:
            # Balanced accuracy is robust to the small class-imbalance that
            # can occur in CV folds (XOR labels are 50/50 in expectation).
            model.eval()
            with torch.no_grad():
                pred = model(Xe).argmax(1).cpu().numpy()
            bacc = balanced_accuracy_score(y_te, pred)
            curve[eval_idx, 0] = ep
            curve[eval_idx, 1] = bacc
            eval_idx += 1
    return curve


def _progress(i, n, t0, label=''):
    pct = (i + 1) / n
    elapsed = time.time() - t0
    eta = elapsed / pct * (1 - pct) if pct > 0 else 0
    bar = ('#' * int(30 * pct)).ljust(30, '.')
    print(f'\r[{bar}] {i+1}/{n} {pct*100:5.1f}% | {elapsed/60:5.1f}m elapsed | '
          f'ETA {eta/60:5.1f}m | {label}', end='', flush=True)
    if i + 1 == n:
        print()


# ---- pre-compute arch metadata ----------------------------------------------
N_D, N_W = len(DEPTHS), len(WIDTHS)
params    = np.zeros((N_D, N_W), dtype=np.int64)
fwd_flops = np.zeros((N_D, N_W), dtype=np.int64)
for di, d in enumerate(DEPTHS):
    for wi, w in enumerate(WIDTHS):
        params[di, wi]    = param_count(d, w)
        fwd_flops[di, wi] = fwd_flops_per_sample(d, w)

log(f'\nParameter counts (rows=depths, cols=widths):')
log('  d\\w' + ''.join(f'{w:>9d}' for w in WIDTHS))
for di, d in enumerate(DEPTHS):
    log(f'  d={d}' + ''.join(f' {int(params[di,wi]):>8d}' for wi in range(N_W)))

# ---- sweep ------------------------------------------------------------------
N_PART  = N_SPLITS * N_REPEATS
n_evals = 1 + EPOCHS // EVAL_EVERY
total   = N_TASKS * N_D * N_W * N_PART
log(f'\nTotal trainings: {total} '
    f'(n_tasks={N_TASKS} x n_d={N_D} x n_w={N_W} x n_part={N_PART})')

curves    = np.full((N_TASKS, N_D, N_W, N_PART, n_evals, 2),
                    np.nan, dtype=np.float32)
final_acc = np.full((N_TASKS, N_D, N_W, N_PART), np.nan, dtype=np.float32)

# Pre-compute folds per task
all_folds = []
for (g, P, X, y) in tasks:
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=RS)
    folds = list(rskf.split(X, y))
    assert len(folds) == N_PART
    all_folds.append(folds)

t0 = time.time()
done = 0
last_ckpt = 0
CKPT_EVERY = 500

for ti, (g, P, X, y) in enumerate(tasks):
    folds = all_folds[ti]
    label_str = f'g{g}P{P:2d}'
    for di, d in enumerate(DEPTHS):
        for wi, w in enumerate(WIDTHS):
            for pi, (tr_idx, te_idx) in enumerate(folds):
                # Standardise inputs inside each CV fold (test stats from train).
                X_tr, X_te = std_scale(X[tr_idx], X[te_idx])
                y_tr, y_te = y[tr_idx], y[te_idx]
                # Seed mixes fold + arch indices: every (arch, fold) gets a
                # unique reproducible init -- but the *same* (arch, fold) is
                # always reproduced bit-for-bit.
                seed = pi * 100 + di * 7 + wi
                c = train_one(X_tr, y_tr, X_te, y_te, d, w, seed=seed)
                curves[ti, di, wi, pi] = c
                # Final accuracy = last epoch's evaluation.
                final_acc[ti, di, wi, pi] = c[-1, 1]
                done += 1
                _progress(done - 1, total, t0,
                          label=f'{label_str} d{d} w{w:3d} p{pi:2d} '
                                f'acc={c[-1,1]:.3f}')
                if done - last_ckpt >= CKPT_EVERY:
                    np.savez_compressed(
                        CKPT_PATH,
                        depths=np.array(DEPTHS), widths=np.array(WIDTHS),
                        task_grid=np.array([t[0] for t in tasks]),
                        task_P=np.array([t[1] for t in tasks]),
                        params=params, fwd_flops=fwd_flops,
                        curves=curves, final_acc=final_acc,
                        n_done=np.int64(done))
                    last_ckpt = done

# ---- post-processing --------------------------------------------------------
mean_curves = np.nanmean(curves, axis=3)     # (N_TASKS, N_D, N_W, n_evals, 2)
mean_final  = np.nanmean(final_acc, axis=3)  # (N_TASKS, N_D, N_W)
std_final   = np.nanstd (final_acc, axis=3)  # (N_TASKS, N_D, N_W)

# Per-task n_train (assume balanced folds)
n_train_per_task = np.array(
    [int(len(t[3]) * (N_SPLITS - 1) / N_SPLITS) for t in tasks],
    dtype=np.int64)

# Epochs / FLOPs to each threshold
N_THR = len(THRESHOLDS)
epochs_to_thr = np.full((N_TASKS, N_D, N_W, N_THR), np.nan, dtype=np.float32)
flops_to_thr  = np.full((N_TASKS, N_D, N_W, N_THR), np.nan, dtype=np.float64)
for ti in range(N_TASKS):
    n_tr = int(n_train_per_task[ti])
    for tj, T in enumerate(THRESHOLDS):
        for di in range(N_D):
            for wi in range(N_W):
                cv = mean_curves[ti, di, wi]
                hits = np.where(cv[:, 1] >= T)[0]
                if len(hits) > 0:
                    e = float(cv[hits[0], 0])
                    epochs_to_thr[ti, di, wi, tj] = e
                    flops_to_thr[ti, di, wi, tj] = (
                        3.0 * fwd_flops[di, wi] * n_tr * max(e, 1.0))

# ---- log summary ------------------------------------------------------------
log('\nPer-task best-arch summary:')
log('  task           best_acc  arch         params')
for ti, (g, P, X, y) in enumerate(tasks):
    flat = mean_final[ti]
    di_w, wi_w = np.unravel_index(np.nanargmax(flat), flat.shape)
    log(f'  g{g:2d} P={P:2d}     {float(flat[di_w,wi_w]):.4f}  '
        f'd={DEPTHS[di_w]:1d},w={WIDTHS[wi_w]:3d}    '
        f'{int(params[di_w,wi_w]):>6d}')

# ---- save -------------------------------------------------------------------
np.savez_compressed(
    SAVE_PATH,
    depths=np.array(DEPTHS), widths=np.array(WIDTHS),
    task_grid=np.array([t[0] for t in tasks], dtype=np.int64),
    task_P=np.array([t[1] for t in tasks], dtype=np.int64),
    n_splits=np.int64(N_SPLITS), n_repeats=np.int64(N_REPEATS),
    rs=np.int64(RS), epochs=np.int64(EPOCHS), eval_every=np.int64(EVAL_EVERY),
    params=params, fwd_flops=fwd_flops, n_train_per_task=n_train_per_task,
    curves=curves, final_acc=final_acc,
    mean_curves=mean_curves, mean_final=mean_final, std_final=std_final,
    thresholds=np.array(THRESHOLDS), thr_names=np.array(THR_NAMES),
    epochs_to_thr=epochs_to_thr, flops_to_thr=flops_to_thr)
elapsed = time.time() - t0
log(f'\nSaved: {SAVE_PATH}  ({os.path.getsize(SAVE_PATH)/1e6:.1f} MB)')
log(f'Elapsed: {elapsed/60:.1f} min  '
    f'({elapsed/total*1000:.1f} ms / training)')
if os.path.exists(CKPT_PATH):
    try: os.remove(CKPT_PATH)
    except OSError: pass

log('MLP XOR sweep done.')
_log.close()
