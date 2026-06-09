"""XOR-task ladder dataset.

For 16x16 and 32x32 grids, build:
  - (X1, X2) integer coord arrays
  - XOR labels at scales P = 1, 2, 4, 8 (and P=16 on 32x32)

XOR label at pixel (i, j) for scale P:
    label(i, j; P) = ((i // P) + (j // P)) mod 2
Period-P chequerboard. Class balance = 50% exactly.
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE = os.path.join(HERE, 'data', 'xor_datasets.npz')

SCALES_16 = [1, 2, 4, 8]
SCALES_32 = [1, 2, 4, 8, 16]


def xor_labels(N, P):
    """(N, N) int8 labels for XOR scale P pixels."""
    ii = np.arange(N)[:, None]
    jj = np.arange(N)[None, :]
    return (((ii // P) + (jj // P)) & 1).astype(np.int8)


def coord_grid(N):
    """(N*N, 2) f32 grid mapped to [-1, 1]^2 (matches typical MLP-on-coords convention)."""
    pts = np.arange(N, dtype=np.float32)
    pts = -1.0 + 2.0 * pts / (N - 1)
    xv, yv = np.meshgrid(pts, pts, indexing='ij')
    return np.stack([xv.ravel(), yv.ravel()], axis=1)


payload = {}

# 16x16
N = 16
coords_16 = coord_grid(N)
payload['coords_16'] = coords_16
for P in SCALES_16:
    lab = xor_labels(N, P).reshape(-1)
    payload[f'labels_16_P{P}'] = lab.astype(np.int64)
    print(f'  16x16 P={P:2d}: balance={lab.mean():.3f}, '
          f'#0={int((lab==0).sum())}, #1={int((lab==1).sum())}')

# 32x32
N = 32
coords_32 = coord_grid(N)
payload['coords_32'] = coords_32
for P in SCALES_32:
    lab = xor_labels(N, P).reshape(-1)
    payload[f'labels_32_P{P}'] = lab.astype(np.int64)
    print(f'  32x32 P={P:2d}: balance={lab.mean():.3f}, '
          f'#0={int((lab==0).sum())}, #1={int((lab==1).sum())}')

payload['scales_16'] = np.asarray(SCALES_16, dtype=np.int64)
payload['scales_32'] = np.asarray(SCALES_32, dtype=np.int64)

os.makedirs(os.path.dirname(SAVE), exist_ok=True)
np.savez_compressed(SAVE, **payload)
print(f'Saved: {SAVE}  ({os.path.getsize(SAVE)/1024:.1f} kB)')
