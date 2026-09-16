"""
preprocess_femto.py — build per-bearing HI trajectories for PRONOSTIA/FEMTO.

Same HI as XJTU: RMS of the horizontal vibration channel per snapshot.
FEMTO acc_*.csv: [hour, minute, second, microsecond, horiz_accel, vert_accel],
2560 rows/snapshot @ 25.6kHz, one snapshot every 10s. Delimiter is ',' or ';'
depending on the bearing; detected per bearing from the first file.

For each of the 17 bearings we take the directory copy with the MOST acc files
(the full run-to-failure trajectory; the challenge test copies are truncated).

Output: femto_hi_trajectories.npz  key 'F{cond}_Bearing{X}_{Y}' -> float64 RMS-HI,
plus 'meta' json (condition, EOL).
"""
import os, re, json, time, glob
import numpy as np
import pandas as pd

# Local copy of the PRONOSTIA / FEMTO dataset (download link in README.md).
# Default: <repo>/data/PRONOSTIA/ ; override with the FEMTO_ROOT environment variable.
ROOT = os.environ.get(
    'FEMTO_ROOT',
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'PRONOSTIA'))
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'data', 'femto_hi_trajectories.npz')


def snap_idx(p):
    m = re.search(r'acc_(\d+)\.csv$', os.path.basename(p))
    return int(m.group(1)) if m else -1


def detect_sep(path):
    with open(path) as f:
        line = f.readline()
    return ';' if line.count(';') > line.count(',') else ','


def rms_horizontal(path, sep):
    col = pd.read_csv(path, sep=sep, header=None, usecols=[4]).values[:, 0]
    col = np.asarray(col, dtype=np.float64)
    return float(np.sqrt(np.mean(col * col)))


def main():
    t0 = time.time()
    # collect, per bearing, the directory with the most acc files
    best = {}   # bearing -> (count, dirpath)
    for dirpath, dirs, files in os.walk(ROOT):
        b = os.path.basename(dirpath)
        if re.match(r'Bearing\d_\d$', b):
            n = len([f for f in files if re.match(r'acc_\d+\.csv$', f)])
            if n > 0 and n > best.get(b, (0, None))[0]:
                best[b] = (n, dirpath)

    trajectories, meta = {}, {}
    for b in sorted(best):
        n, dirpath = best[b]
        cond = 'F' + b.replace('Bearing', '')[0]      # F1/F2/F3
        accs = sorted(glob.glob(os.path.join(dirpath, 'acc_*.csv')), key=snap_idx)
        sep = detect_sep(accs[0])
        hi = np.empty(len(accs), dtype=np.float64)
        for i, p in enumerate(accs):
            hi[i] = rms_horizontal(p, sep)
        key = f'{cond}_{b}'
        trajectories[key] = hi
        meta[key] = {'condition': cond, 'EOL': int(len(hi))}
        print(f'  {key:16s} T={len(hi):5d} sep="{sep}" '
              f'RMS[0]={hi[0]:.3f} RMS[-1]={hi[-1]:.3f} '
              f'ratio={hi[-1]/hi[:5].mean():.2f} [{int(time.time()-t0)}s]', flush=True)

    np.savez_compressed(OUT, meta=json.dumps(meta), **trajectories)
    print(f'\nSaved {len(trajectories)} FEMTO bearings -> {OUT} [{int(time.time()-t0)}s]')


if __name__ == '__main__':
    main()
