"""run_sweep.py -- production multi-GPU spin-wave-disk simulation sweep.

For each input pair (X1, X2) in a GRID x GRID input space, drives the disk
with two Gaussian-windowed 800 MHz pulses at coordinate-encoded times, records
the time series of the out-of-plane magnetisation m_z, crops + block-means
each frame to the chosen output resolution, and writes per-sample CSV
directories. Runs N_GPUS workers in parallel within a single job (each worker
pins to one GPU via CUDA_VISIBLE_DEVICES). The build is resumable: a sample
whose output dir already contains N_FRAMES CSVs is skipped.

Default configuration (matches Supplementary Fig. 7 / the manuscript's Fig. 5
production sweep): 32 x 32 = 1024 samples, T_step = 0.05 ns per coordinate
step, Tmax = 4 ns, output block-meaned to 64 x 64 per frame, 201 frames per
sample.

Per sample (X1, X2 in 0..GRID-1):
  - 512x512x1 disk (2 nm xy cells, 1.5 nm layer, Msat = 1e6 A/m,
    Ku1 = 6.3e5 J/m^3 [just above SRT], B_ext = 45 mT in-plane), warm-started
    from a precomputed relax state.
  - 2 input transducers, each fired with ONE 800 MHz Gaussian-windowed pulse
    (sigma = 1.25 ns) centred at  coord * T_step + 0.2 ns offset.
  - run for Tmax = 4 ns; m_z (full field) and the 6-region table saved every
    100*FixDt = 20 ps  ->  201 frames (t = 0, 20, ..., 4000 ps inclusive).
  - m_z OVFs compressed in-process to 64x64 CSVs (crop [64:448] of 512^2 = 384,
    block-mean 6 -> 12 nm per output cell).

Output: SWRC_OUTPUT_DIR/sample_X1{ii}X2{jj}.out/{m_z000000.csv .. , table.txt}
        + grid_int.npy + MANIFEST.txt

Env:
  SWRC_OUTPUT_DIR  output base dir          (default: this folder / sweep_out)
  SWRC_N_GPUS      parallel batch width     (default: 4)
  SWRC_GRID        grid size G              (default: 32  ->  G*G samples)
  SWRC_TMAX        Tmax (s)                 (default: 4e-9)
  SWRC_TSTEP       coord->time scale (s)    (default: 5e-11 = 0.05 ns)
  SWRC_CROP_LO/HI  m_z crop window          (default: 64 / 448 -> 384x384)
  SWRC_M           CSV/cube spatial side    (default: 64  -> block-mean 384/64 = 6)
  MUMAX3_PATH      mumax3 binary            (else `mumax3` on PATH)
"""
import os, sys, time, glob, shutil, subprocess
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.environ.get("SWRC_OUTPUT_DIR", os.path.join(HERE, "sweep_out"))
N_GPUS   = int(os.environ.get("SWRC_N_GPUS", "4"))
GRID     = int(os.environ.get("SWRC_GRID", "32"))
TMAX     = float(os.environ.get("SWRC_TMAX", "4e-9"))
TSTEP    = float(os.environ.get("SWRC_TSTEP", "5e-11"))
MUMAX    = os.environ.get("MUMAX3_PATH") or shutil.which("mumax3") or shutil.which("mumax3.exe")

FIXDT, SAVE_EVERY = 2e-13, 100
OFFSET, PULSEWIDTH, FREQ, AMPL = 0.2e-9, 1.25e-9, 8e8, 8e5
N_FRAMES = int(round(TMAX / (SAVE_EVERY * FIXDT)))      # 200 for Tmax=4e-9 (mumax emits N_FRAMES+1: t=0..Tmax)
# m_z field crop + block-mean coarse-grain: [64:448] of the 512^2 grid = 384x384
# (most of the 1 um disk, transducers ~centred), block 384/64 = 6 -> 64x64 CSVs
# (12 nm per output cell).
CROP = (int(os.environ.get("SWRC_CROP_LO", "64")), int(os.environ.get("SWRC_CROP_HI", "448")))
M_CG = int(os.environ.get("SWRC_M", "64"))

if MUMAX is None:
    sys.exit("mumax3 not found (set MUMAX3_PATH or put it on PATH)")
os.makedirs(OUT_BASE, exist_ok=True)
RELAX_OVF = os.path.join(OUT_BASE, "relax_sw_disk_512.ovf")

# --- mumax3 script templates --------------------------------------------------
_GEOM = """setgridsize(512, 512, 1)
setcellsize(2e-9, 2e-9, 1.5e-9)
disk := circle(1000e-9)
setgeom(disk)
defregion(0, disk)
Msat = 1e6
Aex = 1.5e-11
alpha = 0.012
DisableZhangLiTorque = true
anisU = vector(0, 0, 1)
Ku1 = 6.3e5
num_inputs := 2
num_transducers := (num_inputs + 1) * 2
base_actuator := circle(5e-08).transl(300e-9, 0, 0)
for i := 0; i < num_transducers; i++ { defregion(i+1, base_actuator.rotz(2*Pi*i/num_transducers)) }
for i := 0; i < num_transducers+1; i++ { Ku1.setRegion(i, 6.3e5) }
B_ext = vector(0.045, 0, 0)
"""

def relax_script():
    return (_GEOM.replace("for i := 0; i < num_transducers+1; i++ { Ku1.setRegion(i, 6.3e5) }",
                          "m = uniform(0, 0.01, 1)\nfor i := 0; i < num_transducers+1; i++ { Ku1.setRegion(i, 6.3e5) }")
            + "relax()\nsaveas(m, \"M_Relax_Initial\")\n")

def _pulse(tc):
    return (f"{AMPL:.6e}*exp(-((t-{tc:.6e}-{OFFSET:.6e})/pulsewidth)*((t-{tc:.6e}-{OFFSET:.6e})/pulsewidth))"
            f"*sin(2*pi*frequency*(t-{tc:.6e}-{OFFSET:.6e}))")

def sample_script(x1, x2, relax_abs):
    return (_GEOM
            + f'm.LoadFile("{relax_abs}")\n'
            + f"FixDt = {FIXDT}\nTmax := {TMAX:.6e}\nAmplitude := {AMPL}\nfrequency := {FREQ}\npulsewidth := {PULSEWIDTH:.6e}\n"
            + f"tableautosave({SAVE_EVERY} * FixDt)\nautosave(m.Comp(2), {SAVE_EVERY}*FixDt)\n"
            + f"Ku1.setRegion(1, {_pulse(x1 * TSTEP)})\n"
            + f"Ku1.setRegion(2, {_pulse(x2 * TSTEP)})\n"
            + "run(Tmax)\n")

# --- OVF read + condense ------------------------------------------------------
def read_ovf2_mz(path):
    nx = ny = nz = 1; vd = 1; nb = 4
    with open(path, "rb") as f:
        while True:
            raw = f.readline()
            if not raw: raise ValueError(f"EOF in OVF header {path}")
            ln = raw.decode("latin-1").strip().lower()
            if   ln.startswith("# xnodes:"):   nx = int(ln.split(":")[1])
            elif ln.startswith("# ynodes:"):   ny = int(ln.split(":")[1])
            elif ln.startswith("# znodes:"):   nz = int(ln.split(":")[1])
            elif ln.startswith("# valuedim:"): vd = int(ln.split(":")[1])
            elif "begin: data binary 4" in ln: nb = 4; break
            elif "begin: data binary 8" in ln: nb = 8; break
        dt = np.float32 if nb == 4 else np.float64
        f.read(nb)
        data = np.frombuffer(f.read(nz*ny*nx*vd*nb), dtype=dt)
    return data.reshape(nz, ny, nx)

def condense(a, M):
    N = a.shape[0]
    if N % M != 0: N = M * (N // M); a = a[:N, :N]
    b = N // M
    return a.reshape(M, b, M, b).mean(axis=(1, 3))

def out_name(x1, x2): return f"sample_X1{x1:02d}X2{x2:02d}.out"

# --- worker ------------------------------------------------------------------
def _worker_init(q):
    try: gid = q.get(timeout=120)
    except Exception: gid = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gid)

def run_one(args):
    x1, x2 = args
    gid = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    odir = os.path.join(OUT_BASE, out_name(x1, x2))
    if len(glob.glob(os.path.join(odir, "m_z*.csv"))) >= N_FRAMES:
        return (x1, x2, "SKIP", gid, 0.0)
    shutil.rmtree(odir, ignore_errors=True)            # clean any partial dir
    script_path = os.path.join(OUT_BASE, out_name(x1, x2).replace(".out", ".mx3"))
    with open(script_path, "w") as f: f.write(sample_script(x1, x2, RELAX_OVF))
    t0 = time.perf_counter()
    p = subprocess.run([MUMAX, "-f", script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        return (x1, x2, f"MUMAXFAIL:{p.stdout[-300:].strip()}", gid, time.perf_counter()-t0)
    ovfs = sorted(glob.glob(os.path.join(odir, "m_z*.ovf")))
    if not ovfs:
        return (x1, x2, "NOOVF", gid, time.perf_counter()-t0)
    for op in ovfs:
        mz = read_ovf2_mz(op)[0]
        cs = condense(mz[CROP[0]:CROP[1], CROP[0]:CROP[1]], M_CG)
        np.savetxt(os.path.splitext(op)[0] + ".csv", cs, delimiter=",", fmt="%.8g")
        os.remove(op)
    try: os.remove(script_path)
    except OSError: pass
    return (x1, x2, "OK", gid, time.perf_counter()-t0)

# --- main --------------------------------------------------------------------
def main():
    print(f"mumax3: {MUMAX}", flush=True)
    print(f"GRID={GRID} ({GRID*GRID} samples)  N_GPUS={N_GPUS}  Tmax={TMAX*1e9:.1f}ns  "
          f"T_step={TSTEP*1e9:.3f}ns  encoding_window={(GRID-1)*TSTEP*1e9:.2f}ns  N_FRAMES~{N_FRAMES}(+1)", flush=True)
    print(f"OUT_BASE={OUT_BASE}", flush=True)
    II, JJ = np.meshgrid(np.arange(GRID), np.arange(GRID), indexing="xy")
    grid_int = np.stack([II.ravel(), JJ.ravel()], axis=-1)   # (G*G, 2) = (X1, X2)
    np.save(os.path.join(OUT_BASE, "grid_int.npy"), grid_int)

    if not os.path.exists(RELAX_OVF):
        rs = os.path.join(OUT_BASE, "relax_build.mx3"); open(rs, "w").write(relax_script())
        t0 = time.perf_counter()
        p = subprocess.run([MUMAX, "-f", rs], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if p.returncode != 0: sys.exit(f"relax build failed: {p.stdout[-600:]}")
        src = os.path.join(OUT_BASE, "relax_build.out", "M_Relax_Initial.ovf")
        if not os.path.exists(src): sys.exit(f"relax OVF not found at {src}")
        shutil.copy(src, RELAX_OVF)
        print(f"relax cache built in {time.perf_counter()-t0:.1f}s -> {RELAX_OVF}", flush=True)
    else:
        print(f"relax cache present: {RELAX_OVF}", flush=True)

    tasks = [(int(x1), int(x2)) for x1, x2 in grid_int]
    mgr = mp.Manager(); q = mgr.Queue()
    for _ in range(64):
        for g in range(N_GPUS): q.put(g)

    t0 = time.perf_counter(); n_ok = n_skip = n_fail = 0; fails = []
    with ProcessPoolExecutor(max_workers=N_GPUS, initializer=_worker_init, initargs=(q,)) as ex:
        futs = {ex.submit(run_one, t): t for t in tasks}
        for fut in as_completed(futs):
            x1, x2 = futs[fut]
            try: _, _, status, gid, took = fut.result()
            except Exception as e: status = f"EXC:{e}"; gid = "?"; took = 0.0
            if   status == "OK":   n_ok += 1
            elif status == "SKIP": n_skip += 1
            else: n_fail += 1; fails.append((x1, x2, status))
            done = n_ok + n_skip + n_fail
            if done % 32 == 0 or status not in ("OK", "SKIP"):
                el = time.perf_counter() - t0
                eta = el / max(done, 1) * (len(tasks) - done)
                print(f"[{done}/{len(tasks)}] X1={x1:02d}X2={x2:02d} {status} gpu={gid} {took:.0f}s "
                      f"| {el/60:.1f}min, ETA {eta/60:.0f}min | ok={n_ok} skip={n_skip} fail={n_fail}", flush=True)

    wall = (time.perf_counter() - t0) / 60
    print(f"\nDONE: {n_ok} ok, {n_skip} skip, {n_fail} fail.  wall {wall:.1f} min", flush=True)
    if fails:
        print("FAILURES:", flush=True)
        for a, b, s in fails: print(f"  X1={a:02d}X2={b:02d}: {s[:200]}", flush=True)
    with open(os.path.join(OUT_BASE, "MANIFEST.txt"), "w") as f:
        f.write(f"grid={GRID} samples={GRID*GRID} Tmax={TMAX} Tstep={TSTEP} "
                f"encoding_window_ns={(GRID-1)*TSTEP*1e9:.3f} n_frames~{N_FRAMES}(+1) "
                f"frame_dt={SAVE_EVERY*FIXDT} crop={CROP} M_cg={M_CG}\n")
        f.write(f"ok={n_ok} skip={n_skip} fail={n_fail} wall_min={wall:.1f}\n")
        for a, b, s in fails: f.write(f"FAIL X1={a:02d}X2={b:02d} {s[:300]}\n")

if __name__ == "__main__":
    main()
