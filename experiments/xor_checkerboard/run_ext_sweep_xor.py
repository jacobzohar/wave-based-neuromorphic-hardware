"""Extended MLP / CNN architecture sweep, gap-fill for 32x32 fine XOR.

The base sweeps (run_mlp_sweep_xor.py 48-arch, run_cnn_sweep_xor.py 25-arch)
left no feedforward arch matching the SW-disk target at 32x32 P in {1,2,4}.
This script extends both grids into bigger depth/width (kernels fixed 3x3) on
just those three tasks, under identical protocol — RepeatedStratifiedKFold(5,3,
rs=0), 400 epochs Adam + cosine — so the new points slot directly into the
comparison figure.

Outputs (flat per-arch arrays, consumed by make_xor_comparison_figure.py):
  data/mlp_ext_xor.npz
  data/cnn_ext_xor.npz
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
DATA = os.path.join(HERE, 'data')
LOG_PATH = os.path.join(HERE, 'log.txt')

NTHR = min(8, os.cpu_count() or 4)
torch.set_num_threads(NTHR)
os.environ.setdefault('OMP_NUM_THREADS', str(NTHR))
os.environ.setdefault('MKL_NUM_THREADS', str(NTHR))

# ---- config -----------------------------------------------------------------
EXT_P      = [1, 2, 4]                       # 32x32 fine rungs to gap-fill
N_SPLITS, N_REPEATS, RS = 5, 3, 0
EPOCHS, EVAL_EVERY, LR = 400, 5, 1e-3
KERNEL, INPUT_DIM, INPUT_CH, N_CLASS = 3, 2, 2, 2

# extended archs (beyond base grids): bigger depth + bigger width, k=3x3
MLP_EXT = [(10, 128), (12, 128), (16, 256), (10, 256),
           (12, 256), (4, 512), (8, 512), (12, 512)]
CNN_EXT = [(8, 64), (10, 64), (8, 96), (10, 96),
           (6, 128), (8, 128), (10, 128), (10, 32)]

_log = open(LOG_PATH, 'a', encoding='utf-8')
def log(msg=''):
    print(msg, flush=True)
    _log.write(msg + '\n'); _log.flush()

log('\n' + '=' * 78)
log(f'EXT (MLP+CNN gap-fill) XOR sweep STARTED at {time.strftime("%Y-%m-%d %H:%M:%S")}')
log('=' * 78)
log(f'EXT_P={EXT_P}  torch_threads={NTHR}  epochs={EPOCHS}')
log(f'MLP_EXT ({len(MLP_EXT)}): {MLP_EXT}')
log(f'CNN_EXT ({len(CNN_EXT)}): {CNN_EXT}')

# ---- data --------------------------------------------------------------------
D = np.load(os.path.join(DATA, 'xor_datasets.npz'))
coords_32 = D['coords_32'].astype(np.float32)            # (1024, 2)
labels_32 = {P: D[f'labels_32_P{P}'].astype(np.int64) for P in EXT_P}
coord_img_32 = coords_32.reshape(32, 32, 2).transpose(2, 0, 1).astype(np.float32)
N_TASK = len(EXT_P)
N_PART = N_SPLITS * N_REPEATS
N_EVAL = 1 + EPOCHS // EVAL_EVERY
n_train_per_task = np.array(
    [int(len(labels_32[P]) * (N_SPLITS - 1) / N_SPLITS) for P in EXT_P],
    dtype=np.int64)

# folds per task (identical RSKF spec as base sweeps)
folds_per_task = []
for P in EXT_P:
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=RS)
    folds_per_task.append(list(rskf.split(coords_32, labels_32[P])))


# ---- models ------------------------------------------------------------------
class FlexMLP(nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        dims = [INPUT_DIM] + [width] * depth + [N_CLASS]
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1:
                x = torch.relu(x)
        return x


class FlexCNN(nn.Module):
    def __init__(self, depth, width):
        super().__init__()
        pad = KERNEL // 2
        layers, c_prev = [], INPUT_CH
        for _ in range(depth):
            layers += [nn.Conv2d(c_prev, width, KERNEL, padding=pad),
                       nn.ReLU(inplace=False)]
            c_prev = width
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv2d(c_prev, N_CLASS, 1)
    def forward(self, x):
        return self.head(self.body(x))


def mlp_params(d, w):
    p = INPUT_DIM * w + w
    for _ in range(d - 1):
        p += w * w + w
    return int(p + w * N_CLASS + N_CLASS)


def mlp_fwd_flops(d, w):
    f = 2 * INPUT_DIM * w
    for _ in range(d - 1):
        f += 2 * w * w
    return int(f + 2 * w * N_CLASS)


def cnn_params(d, w):
    p = INPUT_CH * w * KERNEL * KERNEL + w
    for _ in range(d - 1):
        p += w * w * KERNEL * KERNEL + w
    return int(p + w * N_CLASS + N_CLASS)


def cnn_fwd_flops_pix(d, w):
    f = 2 * INPUT_CH * w * KERNEL * KERNEL
    for _ in range(d - 1):
        f += 2 * w * w * KERNEL * KERNEL
    return int(f + 2 * w * N_CLASS)


def std_scale(X_tr, X_te):
    mu, sig = X_tr.mean(0), X_tr.std(0) + 1e-8
    return (X_tr - mu) / sig, (X_te - mu) / sig


def train_mlp(X_tr, y_tr, X_te, y_te, d, w, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = FlexMLP(d, w)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X_tr), torch.from_numpy(y_tr)
    Xe = torch.from_numpy(X_te)
    curve = np.zeros((N_EVAL, 2), dtype=np.float32)
    ei = 0
    for ep in range(EPOCHS + 1):
        if ep > 0:
            model.train(); opt.zero_grad()
            loss = loss_fn(model(Xt), yt)
            loss.backward(); opt.step(); sched.step()
        if ep % EVAL_EVERY == 0 and ei < N_EVAL:
            model.eval()
            with torch.no_grad():
                pred = model(Xe).argmax(1).cpu().numpy()
            curve[ei] = (ep, balanced_accuracy_score(y_te, pred))
            ei += 1
    return curve


def train_cnn(coord_img, labels_img, tr_mask, te_mask, d, w, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    tr_x = coord_img[:, tr_mask]
    mu, sig = tr_x.mean(1), tr_x.std(1) + 1e-8
    X_norm = ((coord_img - mu[:, None, None]) / sig[:, None, None]).astype(np.float32)
    Xt = torch.from_numpy(X_norm).unsqueeze(0)
    yt = torch.from_numpy(labels_img).long()
    tr_t, te_t = torch.from_numpy(tr_mask), torch.from_numpy(te_mask)
    model = FlexCNN(d, w)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()
    curve = np.zeros((N_EVAL, 2), dtype=np.float32)
    ei = 0
    y_te_np = labels_img[te_mask]
    for ep in range(EPOCHS + 1):
        if ep > 0:
            model.train(); opt.zero_grad()
            out = model(Xt)[0].permute(1, 2, 0)[tr_t]
            loss = loss_fn(out, yt[tr_t])
            loss.backward(); opt.step(); sched.step()
        if ep % EVAL_EVERY == 0 and ei < N_EVAL:
            model.eval()
            with torch.no_grad():
                pred = model(Xt)[0].argmax(0).cpu().numpy()
            curve[ei] = (ep, balanced_accuracy_score(y_te_np, pred[te_mask]))
            ei += 1
    return curve


def progress(i, n, t0, label):
    pct = (i + 1) / n
    el = time.time() - t0
    eta = el / pct * (1 - pct) if pct > 0 else 0
    bar = ('#' * int(30 * pct)).ljust(30, '.')
    print(f'\r[{bar}] {i+1}/{n} {pct*100:5.1f}% | {el/60:5.1f}m | '
          f'ETA {eta/60:5.1f}m | {label}', end='', flush=True)
    if i + 1 == n:
        print()


# ============================================================================
# MLP extended sweep
# ============================================================================
log(f'\n--- MLP extended sweep: {len(MLP_EXT)} archs x {N_TASK} tasks '
    f'x {N_PART} folds = {len(MLP_EXT)*N_TASK*N_PART} trainings ---')
mlp_final = np.full((len(MLP_EXT), N_TASK, N_PART), np.nan, np.float32)
mlp_curves = np.full((len(MLP_EXT), N_TASK, N_EVAL, 2), np.nan, np.float32)
mlp_fwd = np.array([mlp_fwd_flops(d, w) for d, w in MLP_EXT], dtype=np.int64)
mlp_par = np.array([mlp_params(d, w) for d, w in MLP_EXT], dtype=np.int64)

total = len(MLP_EXT) * N_TASK * N_PART
t0 = time.time(); done = 0
for ai, (d, w) in enumerate(MLP_EXT):
    cur_acc = np.zeros((N_TASK, N_PART, N_EVAL, 2), np.float32)
    for ti, P in enumerate(EXT_P):
        y = labels_32[P]
        for pi, (tr, te) in enumerate(folds_per_task[ti]):
            X_tr, X_te = std_scale(coords_32[tr], coords_32[te])
            c = train_mlp(X_tr, y[tr], X_te, y[te], d, w, seed=pi * 100 + ai)
            cur_acc[ti, pi] = c
            mlp_final[ai, ti, pi] = c[-1, 1]
            done += 1
            progress(done - 1, total, t0, f'MLP d{d} w{w} P{P} p{pi}')
    mlp_curves[ai] = cur_acc.mean(1)
    np.savez_compressed(
        os.path.join(DATA, 'mlp_ext_xor.npz'),
        arch_d=np.array([a[0] for a in MLP_EXT]),
        arch_w=np.array([a[1] for a in MLP_EXT]),
        fwd_flops=mlp_fwd, params=mlp_par,
        task_P=np.array(EXT_P), task_grid=np.full(N_TASK, 32),
        n_splits=N_SPLITS, n_repeats=N_REPEATS, rs=RS,
        epochs=EPOCHS, eval_every=EVAL_EVERY,
        n_train_per_task=n_train_per_task,
        final_acc=mlp_final, mean_curves=mlp_curves,
        mean_final=np.nanmean(mlp_final, axis=2),
        std_final=np.nanstd(mlp_final, axis=2),
        n_arch_done=ai + 1)

log('\nMLP extended sweep — best new arch per task:')
mf = np.nanmean(mlp_final, axis=2)
for ti, P in enumerate(EXT_P):
    bi = int(np.nanargmax(mf[:, ti]))
    log(f'  32x32 P={P}: best ext MLP {mf[bi,ti]:.4f}  '
        f'd={MLP_EXT[bi][0]} w={MLP_EXT[bi][1]}')
log(f'MLP ext elapsed {(time.time()-t0)/60:.1f} min')

# ============================================================================
# CNN extended sweep
# ============================================================================
log(f'\n--- CNN extended sweep: {len(CNN_EXT)} archs x {N_TASK} tasks '
    f'x {N_PART} folds = {len(CNN_EXT)*N_TASK*N_PART} trainings ---')
cnn_final = np.full((len(CNN_EXT), N_TASK, N_PART), np.nan, np.float32)
cnn_curves = np.full((len(CNN_EXT), N_TASK, N_EVAL, 2), np.nan, np.float32)
cnn_fwd = np.array([cnn_fwd_flops_pix(d, w) for d, w in CNN_EXT], dtype=np.int64)
cnn_par = np.array([cnn_params(d, w) for d, w in CNN_EXT], dtype=np.int64)

total = len(CNN_EXT) * N_TASK * N_PART
t0 = time.time(); done = 0
for ai, (d, w) in enumerate(CNN_EXT):
    cur_acc = np.zeros((N_TASK, N_PART, N_EVAL, 2), np.float32)
    for ti, P in enumerate(EXT_P):
        labels_img = labels_32[P].reshape(32, 32)
        for pi, (tr, te) in enumerate(folds_per_task[ti]):
            tr_mask = np.zeros(1024, bool); tr_mask[tr] = True
            te_mask = np.zeros(1024, bool); te_mask[te] = True
            c = train_cnn(coord_img_32, labels_img,
                          tr_mask.reshape(32, 32), te_mask.reshape(32, 32),
                          d, w, seed=pi * 100 + ai)
            cur_acc[ti, pi] = c
            cnn_final[ai, ti, pi] = c[-1, 1]
            done += 1
            progress(done - 1, total, t0, f'CNN d{d} w{w} P{P} p{pi}')
    cnn_curves[ai] = cur_acc.mean(1)
    np.savez_compressed(
        os.path.join(DATA, 'cnn_ext_xor.npz'),
        arch_d=np.array([a[0] for a in CNN_EXT]),
        arch_w=np.array([a[1] for a in CNN_EXT]),
        kernel=KERNEL,
        fwd_flops_per_pixel=cnn_fwd, params=cnn_par,
        task_P=np.array(EXT_P), task_grid=np.full(N_TASK, 32),
        n_splits=N_SPLITS, n_repeats=N_REPEATS, rs=RS,
        epochs=EPOCHS, eval_every=EVAL_EVERY,
        n_train_per_task=n_train_per_task,
        final_acc=cnn_final, mean_curves=cnn_curves,
        mean_final=np.nanmean(cnn_final, axis=2),
        std_final=np.nanstd(cnn_final, axis=2),
        n_arch_done=ai + 1)

log('\nCNN extended sweep — best new arch per task:')
cf = np.nanmean(cnn_final, axis=2)
for ti, P in enumerate(EXT_P):
    bi = int(np.nanargmax(cf[:, ti]))
    log(f'  32x32 P={P}: best ext CNN {cf[bi,ti]:.4f}  '
        f'd={CNN_EXT[bi][0]} w={CNN_EXT[bi][1]}')
log(f'CNN ext elapsed {(time.time()-t0)/60:.1f} min')

log('\nEXT XOR sweep done. Wrote data/mlp_ext_xor.npz, data/cnn_ext_xor.npz')
_log.close()
