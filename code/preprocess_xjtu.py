"""
preprocess_xjtu.py — build per-bearing health-indicator (HI) trajectories for the
DiffCal calibration diagnostic.

HI = RMS of the horizontal vibration signal per snapshot (standard XJTU practice).
One snapshot per minute; each snapshot CSV has 32768 rows x 2 channels
(Horizontal_vibration_signals, Vertical_vibration_signals) @ 25.6 kHz.

Output: hi_trajectories.npz with
    <bearing_key> -> float64 array (T,) of raw RMS-HI, ordered snapshot 1..T
    plus 'meta' json (condition, EOL=T) — bearing_key = e.g. 'C1_Bearing1_1'.

Run:  python preprocess_xjtu.py
"""
import os, re, json, time
import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'data', 'xjtu_raw', 'XJTU-SY_Bearing_Datasets')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   'data', 'hi_trajectories.npz')

COND_MAP = {'35Hz12kN': 'C1', '37.5Hz11kN': 'C2', '40Hz10kN': 'C3'}


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def snapshot_index(fname):
    m = re.match(r'(\d+)\.csv$', fname)
    return int(m.group(1)) if m else -1


def main():
    t0 = time.time()
    trajectories = {}
    meta = {}
    for cond_dir in sorted(os.listdir(RAW)):
        cpath = os.path.join(RAW, cond_dir)
        if not os.path.isdir(cpath):
            continue
        cond = COND_MAP.get(cond_dir, cond_dir)
        for bearing in sorted(os.listdir(cpath)):
            bpath = os.path.join(cpath, bearing)
            if not os.path.isdir(bpath):
                continue
            files = [f for f in os.listdir(bpath) if f.endswith('.csv')]
            files.sort(key=snapshot_index)
            hi = np.empty(len(files), dtype=np.float64)
            for i, f in enumerate(files):
                # horizontal channel only (usecols=0); C engine, fast
                col = pd.read_csv(os.path.join(bpath, f), usecols=[0]).values[:, 0]
                hi[i] = rms(col)
            key = f'{cond}_{bearing}'
            trajectories[key] = hi
            meta[key] = {'condition': cond, 'EOL': int(len(hi))}
            print(f'  {key:20s}  T={len(hi):5d}  RMS[0]={hi[0]:.4f} '
                  f'RMS[-1]={hi[-1]:.4f}  ratio={hi[-1]/hi[:5].mean():.2f}  '
                  f'[{int(time.time()-t0)}s]', flush=True)

    np.savez_compressed(OUT, meta=json.dumps(meta), **trajectories)
    print(f'\nSaved {len(trajectories)} bearings -> {OUT}  '
          f'[{int(time.time()-t0)}s total]')


if __name__ == '__main__':
    main()
