"""CNN-on-coords capacity sweep on the XOR-task ladder.

25 archs (d in {1,2,3,4,6} x w in {4,8,16,32,64})
x 9 XOR tasks (4 at 16x16: P in {1,2,4,8} + 5 at 32x32: P in {1,2,4,8,16})
x RepeatedStratifiedKFold(5, 3, random_state=0) = 15 partitions
= 3375 CNN trainings, 400 epochs full-image Adam + cosine each.

Input: 2-channel (x, y) coord image HxW (same per-pixel input as MLP, but
preserving the 2D layout the substrate also receives). Each conv has 3x3
kernel, padding=same, so output remains HxW. Final 1x1 conv produces a
per-pixel 2-class logit map.

Loss: cross-entropy on training-fold pixels only (mask out test pixels).
Eval: balanced_accuracy on test-fold pixels.

Same fold splits + 400 epochs + Adam lr=1e-3 + cosine LR as run_mlp_sweep_xor.py
so per-pixel forward-FLOPs vs bal_acc comparisons are apples-to-apples.

Inline progress bar; checkpoint save every 200 trainings.
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

# 4 BLAS threads is the throughput sweet spot for these conv-on-32x32 sizes:
# fewer leaves cores idle, more spends time in scheduler overhead.
_N_THREADS = int(os.environ.get('TORCH_THREADS', '4'))
torch.set_num_threads(_N_THREADS)
os.environ.setdefault('OMP_NUM_THREADS', str(_N_THREADS))
os.environ.setdefault('MKL_NUM_THREADS', str(_N_THREADS))

# ---- config -----------------------------------------------------------------
DATA_PATH = os.path.join(HERE, 'data', 'xor_datasets.npz')
SAVE_PATH = os.path.join(HERE, 'data', 'cnn_sweep_xor.npz')
CKPT_PATH = os.path.join(HERE, 'data', 'cnn_sweep_xor_ckpt.npz')
LOG_PATH  = os.path.join(HERE, 'log.txt')

# 5 x 5 = 25 architectures. 3x3 padded convs throughout so HxW is preserved
# layer-to-layer; receptive field grows linearly with depth (2d+1).
DEPTHS     = [1, 2, 3, 4, 6]
WIDTHS     = [4, 8, 16, 32, 64]
KERNEL     = 3
# Identical 5-fold x 3-repeat splits to the MLP sweep (same random_state=0)
# so per-pixel FLOPs vs accuracy is a fair side-by-side comparison.
N_SPLITS   = 5
N_REPEATS  = 3
RS         = 0
# Same 400 epochs / Adam lr=1e-3 / cosine schedule as the MLP -- different
# optimiser hyper-params would confound the architecture comparison.
EPOCHS     = 400
EVAL_EVERY = 5
LR         = 1e-3

# Input: 2-channel (x, y) coord image (same per-pixel info as the MLP's coord
# vector, but presented as an HxW tensor so a conv can use the 2-D layout).
INPUT_CH = 2
N_CLASS  = 2

THRESHOLDS = [0.60, 0.80, 0.90, 0.95]
THR_NAMES  = ['T_60', 'T_80', 'T_90', 'T_95']

# ---- logging helper ---------------------------------------------------------
_log = open(LOG_PATH, 'a', encoding='utf-8')
def log(msg=''):
    print(msg, flush=True)
    _log.write(msg + '\n'); _log.flush()

log('\n' + '='*78)
log(f'CNN XOR sweep RUN STARTED at {time.strftime("%Y-%m-%d %H:%M:%S")}')
log('='*78)
log(f'depths={DEPTHS}, widths={WIDTHS}, kernel={KERNEL}')
log(f'n_splits={N_SPLITS}, n_repeats={N_REPEATS}, rs={RS}, '
    f'epochs={EPOCHS}, eval_every={EVAL_EVERY}, lr={LR}')
log(f'thresholds={list(zip(THR_NAMES, THRESHOLDS))}')
log(f'torch threads={_N_THREADS}')

# ---- data -------------------------------------------------------------------
D = np.load(DATA_PATH)
scales_16 = list(D['scales_16'])
scales_32 = list(D['scales_32'])

def coord_image(N, coords_flat):
    """(N*N, 2) [coords from make_dataset, row-major over (i, j)] -> (2, N, N)."""
    return coords_flat.reshape(N, N, 2).transpose(2, 0, 1).astype(np.float32)

coord_img_16 = coord_image(16, D['coords_16'])
coord_img_32 = coord_image(32, D['coords_32'])

# (grid, P, coords_flat, labels_flat, coord_img, labels_img)
tasks = []
for P in scales_16:
    Pi = int(P)
    labs = D[f'labels_16_P{Pi}'].astype(np.int64)
    tasks.append({
        'grid': 16, 'P': Pi,
        'coords': D['coords_16'].astype(np.float32),
        'labels': labs,
        'coord_img': coord_img_16,
        'labels_img': labs.reshape(16, 16),
    })
for P in scales_32:
    Pi = int(P)
    labs = D[f'labels_32_P{Pi}'].astype(np.int64)
    tasks.append({
        'grid': 32, 'P': Pi,
        'coords': D['coords_32'].astype(np.float32),
        'labels': labs,
        'coord_img': coord_img_32,
        'labels_img': labs.reshape(32, 32),
    })
N_TASKS = len(tasks)
log(f'\nN_TASKS = {N_TASKS}')
for ti, t in enumerate(tasks):
    log(f"  [{ti}] grid={t['grid']} P={t['P']:2d}  "
        f"N_pix={len(t['labels'])}  balance={t['labels'].mean():.3f}")


# ---- model + helpers --------------------------------------------------------
class FlexCNN(nn.Module):
    """Stack of `depth` padded 3x3 conv-ReLU blocks of `width` channels, then a
    1x1 head producing per-pixel 2-class logits. padding=same keeps the spatial
    extent fixed at the input HxW so the head can be a pure pointwise conv."""
    def __init__(self, input_ch, depth, width, kernel, n_class):
        super().__init__()
        pad = kernel // 2                  # 'same' padding for odd kernels.
        layers = []
        c_prev = input_ch
        for _ in range(depth):
            layers.append(nn.Conv2d(c_prev, width, kernel_size=kernel, padding=pad))
            layers.append(nn.ReLU(inplace=False))
            c_prev = width
        self.body = nn.Sequential(*layers)
        # 1x1 head: per-pixel linear projection to class logits.
        self.head = nn.Conv2d(c_prev, n_class, kernel_size=1)

    def forward(self, x):
        return self.head(self.body(x))


def param_count_cnn(d, w, k=KERNEL, in_ch=INPUT_CH, n_cl=N_CLASS):
    """First conv (in_ch -> w) + (d-1) square (w -> w) + 1x1 head (w -> n_cl).
    Each conv contributes (in*out*k*k) weights + out biases."""
    p = in_ch * w * k * k + w
    for _ in range(d - 1):
        p += w * w * k * k + w
    p += w * n_cl * 1 * 1 + n_cl
    return int(p)


def fwd_flops_per_pixel_cnn(d, w, k=KERNEL, in_ch=INPUT_CH, n_cl=N_CLASS):
    """Per-pixel forward FLOPs (H*W independent: conv FLOPs/pixel = 2*in*out*k*k,
    factor of 2 = mul + add). Reported so the CNN sits on the same FLOPs axis
    as the MLP's per-sample count."""
    f = 2 * in_ch * w * k * k
    for _ in range(d - 1):
        f += 2 * w * w * k * k
    f += 2 * w * n_cl
    return int(f)


def fwd_flops_total_cnn(d, w, H, W, k=KERNEL, in_ch=INPUT_CH, n_cl=N_CLASS):
    """Total forward FLOPs over the full HxW image (one MAC = 2 FLOPs).
    Used to compare per-image cost against the MLP's per-pixel cost x H*W."""
    return int(fwd_flops_per_pixel_cnn(d, w, k, in_ch, n_cl)) * (H * W)


def std_scale_coords(coord_img, tr_mask):
    """Standardize the 2-channel (x, y) coord image using ONLY training pixels
    (test pixels are masked out -- no statistic leakage into the test fold)."""
    tr_x = coord_img[:, tr_mask]
    mu = tr_x.mean(axis=1)
    sig = tr_x.std(axis=1) + 1e-8
    return ((coord_img - mu[:, None, None]) / sig[:, None, None]).astype(np.float32)


def train_one_cnn(coord_img, labels_img, tr_mask, te_mask, depth, width, seed):
    """Train one CNN on a single XOR task / fold; return (n_eval, [epoch, bacc])
    curve sampled every EVAL_EVERY epochs. Each forward pass runs over the
    full HxW image; training loss is computed only on the training-pixel
    mask, test bacc only on the test-pixel mask, so the CNN sees the same
    train/test partition as the MLP and SVM."""
    torch.manual_seed(seed); np.random.seed(seed)

    X_norm = std_scale_coords(coord_img, tr_mask)
    Xt = torch.from_numpy(X_norm).unsqueeze(0)              # (1, 2, H, W)
    yt = torch.from_numpy(labels_img).long()                # (H, W)
    tr_t = torch.from_numpy(tr_mask)                        # (H, W) bool
    te_t = torch.from_numpy(te_mask)                        # (H, W) bool

    model = FlexCNN(INPUT_CH, depth, width, KERNEL, N_CLASS)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.CrossEntropyLoss()

    n_eval = 1 + EPOCHS // EVAL_EVERY
    curve = np.zeros((n_eval, 2), dtype=np.float32)
    eval_idx = 0
    y_te_np = labels_img[te_mask]
    for ep in range(EPOCHS + 1):
        if ep > 0:
            model.train()
            opt.zero_grad()
            out = model(Xt)                                 # (1, 2, H, W)
            # Mask the (H, W) -> (n_tr, 2) prediction to training pixels only.
            out_tr = out[0].permute(1, 2, 0)[tr_t]
            y_tr = yt[tr_t]                                 # (n_tr,)
            loss = loss_fn(out_tr, y_tr)
            loss.backward(); opt.step(); sched.step()
        if ep % EVAL_EVERY == 0 and eval_idx < n_eval:
            model.eval()
            with torch.no_grad():
                out = model(Xt)
                pred = out[0].argmax(0).cpu().numpy()       # (H, W)
            # Balanced accuracy on the test-pixel slice only.
            bacc = balanced_accuracy_score(y_te_np, pred[te_mask])
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
params = np.zeros((N_D, N_W), dtype=np.int64)
fwd_flops_pix = np.zeros((N_D, N_W), dtype=np.int64)
fwd_flops_img_16 = np.zeros((N_D, N_W), dtype=np.int64)
fwd_flops_img_32 = np.zeros((N_D, N_W), dtype=np.int64)
for di, d in enumerate(DEPTHS):
    for wi, w in enumerate(WIDTHS):
        params[di, wi] = param_count_cnn(d, w)
        fwd_flops_pix[di, wi] = fwd_flops_per_pixel_cnn(d, w)
        fwd_flops_img_16[di, wi] = fwd_flops_total_cnn(d, w, 16, 16)
        fwd_flops_img_32[di, wi] = fwd_flops_total_cnn(d, w, 32, 32)

log('\nParameter counts (rows=depths, cols=widths):')
log('  d\\w' + ''.join(f'{w:>9d}' for w in WIDTHS))
for di, d in enumerate(DEPTHS):
    log(f'  d={d}' + ''.join(f' {int(params[di,wi]):>8d}' for wi in range(N_W)))

log('\nForward FLOPs / pixel (rows=depths, cols=widths):')
log('  d\\w' + ''.join(f'{w:>10d}' for w in WIDTHS))
for di, d in enumerate(DEPTHS):
    log(f'  d={d}' + ''.join(f' {int(fwd_flops_pix[di,wi]):>9d}'
                            for wi in range(N_W)))

# ---- sweep ------------------------------------------------------------------
N_PART  = N_SPLITS * N_REPEATS
n_evals = 1 + EPOCHS // EVAL_EVERY
total   = N_TASKS * N_D * N_W * N_PART
log(f'\nTotal trainings: {total} '
    f'(n_tasks={N_TASKS} x n_d={N_D} x n_w={N_W} x n_part={N_PART})')

curves    = np.full((N_TASKS, N_D, N_W, N_PART, n_evals, 2),
                    np.nan, dtype=np.float32)
final_acc = np.full((N_TASKS, N_D, N_W, N_PART), np.nan, dtype=np.float32)

# Pre-compute folds per task (same RSKF spec as MLP — keeps splits identical)
all_folds = []
for t in tasks:
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                   random_state=RS)
    folds = list(rskf.split(t['coords'], t['labels']))
    assert len(folds) == N_PART
    all_folds.append(folds)

t0 = time.time()
done = 0
last_ckpt = 0
CKPT_EVERY = 200

for ti, t in enumerate(tasks):
    g = t['grid']; P = t['P']
    coord_img = t['coord_img']; labels_img = t['labels_img']
    folds = all_folds[ti]
    label_str = f'g{g}P{P:2d}'
    for di, d in enumerate(DEPTHS):
        for wi, w in enumerate(WIDTHS):
            for pi, (tr_idx, te_idx) in enumerate(folds):
                # CV split is over flattened pixel indices; turn each into a
                # (g, g) boolean image so the CNN can mask train/test pixels
                # in place without reshaping its output.
                tr_mask = np.zeros(g * g, dtype=bool)
                te_mask = np.zeros(g * g, dtype=bool)
                tr_mask[tr_idx] = True
                te_mask[te_idx] = True
                tr_mask = tr_mask.reshape(g, g)
                te_mask = te_mask.reshape(g, g)
                # Same seed formula as the MLP sweep so identical (arch, fold)
                # indices select the same RNG state across the two pipelines.
                seed = pi * 100 + di * 7 + wi
                c = train_one_cnn(coord_img, labels_img, tr_mask, te_mask,
                                  d, w, seed=seed)
                curves[ti, di, wi, pi] = c
                final_acc[ti, di, wi, pi] = c[-1, 1]
                done += 1
                _progress(done - 1, total, t0,
                          label=f'{label_str} d{d} w{w:3d} p{pi:2d} '
                                f'acc={c[-1,1]:.3f}')
                if done - last_ckpt >= CKPT_EVERY:
                    np.savez_compressed(
                        CKPT_PATH,
                        depths=np.array(DEPTHS), widths=np.array(WIDTHS),
                        kernel=np.int64(KERNEL),
                        task_grid=np.array([tt['grid'] for tt in tasks]),
                        task_P=np.array([tt['P'] for tt in tasks]),
                        params=params,
                        fwd_flops_per_pixel=fwd_flops_pix,
                        fwd_flops_img_16=fwd_flops_img_16,
                        fwd_flops_img_32=fwd_flops_img_32,
                        curves=curves, final_acc=final_acc,
                        n_done=np.int64(done))
                    last_ckpt = done

# ---- post-processing --------------------------------------------------------
mean_curves = np.nanmean(curves, axis=3)
mean_final  = np.nanmean(final_acc, axis=3)
std_final   = np.nanstd (final_acc, axis=3)

n_train_per_task = np.array(
    [int(len(t['labels']) * (N_SPLITS - 1) / N_SPLITS) for t in tasks],
    dtype=np.int64)

fwd_flops_img_per_task = np.zeros((N_TASKS, N_D, N_W), dtype=np.int64)
for ti, t in enumerate(tasks):
    src = fwd_flops_img_16 if t['grid'] == 16 else fwd_flops_img_32
    fwd_flops_img_per_task[ti] = src

N_THR = len(THRESHOLDS)
epochs_to_thr = np.full((N_TASKS, N_D, N_W, N_THR), np.nan, dtype=np.float32)
flops_to_thr  = np.full((N_TASKS, N_D, N_W, N_THR), np.nan, dtype=np.float64)
for ti in range(N_TASKS):
    fwd_img = fwd_flops_img_per_task[ti]
    for tj, T in enumerate(THRESHOLDS):
        for di in range(N_D):
            for wi in range(N_W):
                cv = mean_curves[ti, di, wi]
                hits = np.where(cv[:, 1] >= T)[0]
                if len(hits) > 0:
                    e = float(cv[hits[0], 0])
                    epochs_to_thr[ti, di, wi, tj] = e
                    # one training step = one forward over the whole image,
                    # so per-step FLOPs = fwd_img regardless of mask.
                    flops_to_thr[ti, di, wi, tj] = (
                        3.0 * int(fwd_img[di, wi]) * max(e, 1.0))

# ---- log summary ------------------------------------------------------------
log('\nPer-task best-arch summary:')
log('  task           best_acc  arch          params   F/pix')
for ti, t in enumerate(tasks):
    flat = mean_final[ti]
    di_w, wi_w = np.unravel_index(np.nanargmax(flat), flat.shape)
    log(f"  g{t['grid']:2d} P={t['P']:2d}     {float(flat[di_w,wi_w]):.4f}  "
        f"d={DEPTHS[di_w]:1d},w={WIDTHS[wi_w]:3d}    "
        f"{int(params[di_w,wi_w]):>6d}  {int(fwd_flops_pix[di_w,wi_w]):>8d}")

# ---- save -------------------------------------------------------------------
np.savez_compressed(
    SAVE_PATH,
    depths=np.array(DEPTHS), widths=np.array(WIDTHS),
    kernel=np.int64(KERNEL),
    task_grid=np.array([t['grid'] for t in tasks], dtype=np.int64),
    task_P=np.array([t['P'] for t in tasks], dtype=np.int64),
    n_splits=np.int64(N_SPLITS), n_repeats=np.int64(N_REPEATS),
    rs=np.int64(RS), epochs=np.int64(EPOCHS), eval_every=np.int64(EVAL_EVERY),
    params=params,
    fwd_flops_per_pixel=fwd_flops_pix,
    fwd_flops_img_16=fwd_flops_img_16,
    fwd_flops_img_32=fwd_flops_img_32,
    fwd_flops_img_per_task=fwd_flops_img_per_task,
    n_train_per_task=n_train_per_task,
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

log('CNN XOR sweep done.')
_log.close()
