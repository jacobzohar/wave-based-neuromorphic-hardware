"""build_cube.py — assemble one mumax3 sweep .out dir tree into an m_z cube
.npy. Takes the source dir and the output name as args, picks up every paired
(X1, X2) run, and stacks them into a `(N, F, H, W)` float32 array.

The cube shape is autodetected: `N` = number of paired runs found, `(H, W)` =
the spatial size of the first CSV in the first run dir. `F` (frame count) is
configurable via `--frames` (default 201). Optional `--n` and `--side` flags
let you assert specific values up front if you want to fail early on bad data.

Usage:
    # production: pick up all paired runs in a sweep dir
    python build_cube.py --src /path/to/sweep_out --out sw_disk_mz_cube.npy --frames 201

    # minimal example: 4 runs, 64x64 spatial, 101 frames at Tmax=2 ns
    python build_cube.py --src ./sweep_out --out mini_cube.npy --frames 101

Writes (this dir):
    <out>                       (N, F, H, W) float32
    run_index<_tag>.json        idx -> {X1, X2, dir}

If <out> matches the pattern <prefix>_<tag>_mz_cube.npy, <tag> is the middle
segment (e.g. "sw_disk_g32_mz_cube.npy" -> tag "g32"); otherwise no tag is
appended. Prints m_z stats + per-pixel std + dead-cell count.
"""
import os, re, json, time, glob, argparse
import numpy as np
try:
    import pandas as pd
    def _read_csv(fp):
        return pd.read_csv(fp, header=None, dtype=np.float32).to_numpy()
except ImportError:
    def _read_csv(fp):
        return np.loadtxt(fp, delimiter=",", dtype=np.float32)

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="dir containing the *X1iX2j*.out run dirs")
ap.add_argument("--out", required=True, help="output .npy name (this dir)")
ap.add_argument("--frames", type=int, default=201, help="frames per sample (default: 201)")
ap.add_argument("--n", type=int, default=None,
                help="expected number of paired runs; auto-detected if omitted")
ap.add_argument("--side", type=int, default=None,
                help="expected spatial side H = W of each frame; auto-detected if omitted")
args = ap.parse_args()
# Resolve --out naturally (absolute path, or relative to CWD) and put the
# run_index.json next to the cube. The optional tag in run_index<_tag>.json
# is the middle segment of <prefix>_<tag>_mz_cube.npy if present.
OUT_PATH = os.path.abspath(args.out)
OUT_DIR  = os.path.dirname(OUT_PATH)
mtag = re.match(r"^[A-Za-z0-9]+_(\w+?)_mz_cube\.npy$", os.path.basename(args.out))
TAG = mtag.group(1) if mtag else ""
N_FRAMES = args.frames

t0 = time.time()
rx = re.compile(r"X1(\d+)X2(\d+)")
all_out = sorted(d for d in os.listdir(args.src) if d.endswith(".out") and rx.search(d))
print(f"build_cube.py  src={args.src}  out={args.out}  tag='{TAG}'")
n_total_out = len([d for d in os.listdir(args.src) if d.endswith(".out")])
print(f"found {len(all_out)} paired (X1,X2) .out dirs (of {n_total_out} total .out)")
N = len(all_out)
if N == 0:
    raise SystemExit(f"no paired (X1iX2j).out dirs found under {args.src}")
if args.n is not None:
    assert N == args.n, f"--n {args.n} given, but found {N} paired runs"

# Autodetect spatial side from the first frame of the first run, unless --side asserts it
first_frame = sorted(glob.glob(os.path.join(args.src, all_out[0], "m_z*.csv")))[0]
H0, W0 = _read_csv(first_frame).shape
if args.side is not None:
    assert H0 == args.side and W0 == args.side, \
        f"--side {args.side} given, but first frame has shape ({H0}, {W0})"
SIDE = H0
assert H0 == W0, f"non-square frames not supported: ({H0}, {W0})"
print(f"  N={N} runs, {N_FRAMES} frames each, {SIDE}x{SIDE} spatial")

cube = np.zeros((N, N_FRAMES, SIDE, SIDE), dtype=np.float32)
run_index = {}
fc = []
for ri, d in enumerate(all_out):
    m = rx.search(d); x1, x2 = int(m.group(1)), int(m.group(2))
    run_index[ri] = {"X1": x1, "X2": x2, "dir": d}
    frames = sorted(glob.glob(os.path.join(args.src, d, "m_z*.csv")))
    fc.append(len(frames))
    assert len(frames) == N_FRAMES, f"{d}: {len(frames)} frames (expected {N_FRAMES})"
    for fi, fp in enumerate(frames):
        arr = _read_csv(fp)
        assert arr.shape == (SIDE, SIDE), f"{fp}: shape {arr.shape} != ({SIDE}, {SIDE})"
        cube[ri, fi] = arr
    # progress every ~10% (at least every 1 for tiny N)
    step = max(1, N // 10)
    if ri % step == 0 or ri == N - 1:
        el = time.time() - t0
        eta = el / (ri + 1) * (N - 1 - ri)
        print(f"  {ri+1}/{N}  ({el:.0f}s, ETA {eta:.0f}s)  {d[:40]}", flush=True)

print(f"frame counts: min={min(fc)} max={max(fc)} (all=={N_FRAMES})")
print(f"cube {cube.shape} {cube.dtype}  {cube.nbytes/1e6:.1f} MB")
print(f"m_z: min={cube.min():.5f} max={cube.max():.5f} mean={cube.mean():.6f} std={cube.std():.5f}")
# Per-pixel std-across-runs (mean over frames): low values = pixels that never respond
spp = cube.std(axis=0).mean(axis=0)
print(f"per-pixel std-across-runs (mean over frames): min={spp.min():.2e} max={spp.max():.2e} median={np.median(spp):.2e}")
# Dead cells: pixels in a centred 96%-side crop that never moved (std < 1e-5 across runs)
crop_side = max(1, int(SIDE * 0.96))
c0 = (SIDE - crop_side) // 2
n_dead = int((spp[c0:c0+crop_side, c0:c0+crop_side] < 1e-5).sum())
print(f"dead cells (std<1e-5) in centred {crop_side}x{crop_side} crop: {n_dead}/{crop_side*crop_side}")
os.makedirs(OUT_DIR, exist_ok=True) if OUT_DIR else None
np.save(OUT_PATH, cube)
run_index_path = os.path.join(OUT_DIR, f"run_index{('_'+TAG) if TAG else ''}.json")
with open(run_index_path, "w") as f:
    json.dump(run_index, f, indent=0)
# sanity: if a canonical run_index.json exists next to the cube, confirm (X1,X2) ordering matches
canon = os.path.join(OUT_DIR, "run_index.json")
if os.path.exists(canon) and N == 256 and canon != run_index_path:
    with open(canon) as f:
        c = {int(k): v for k, v in json.load(f).items()}
    same = all(c[i]["X1"] == run_index[i]["X1"] and c[i]["X2"] == run_index[i]["X2"] for i in range(256))
    print(f"(X1,X2) ordering matches canonical run_index.json: {same}")
print(f"saved {args.out} + run_index{('_'+TAG) if TAG else ''}.json   done ({time.time()-t0:.0f}s)")
