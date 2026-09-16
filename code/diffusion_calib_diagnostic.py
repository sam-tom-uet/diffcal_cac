"""
diffusion_calib_diagnostic.py
=============================================================================
Pre-specified calibration diagnostic for the DiffCal study (see ../PROTOCOL.md).

QUESTION: is the RUL predictive distribution obtained by (a) training a
conditional denoising-diffusion generator on bearing health-indicator (HI)
windows, (b) autoregressively sampling K future degradation trajectories and
(c) taking the first-passage-time (FPT) of a shared failure threshold ---
CALIBRATED?

Falsifier (committed BEFORE building):
  P1  miscalibration exists   : Coverage@0.9 robustly outside [0.80, 0.98]
  P2  structured              : late-life coverage << early-life coverage (paired)
  P3  correctable             : post-hoc dispersion recalibration (gamma tuned on
                                calib bearings) restores Coverage@0.9 in [0.85,0.95]
                                and lowers CRPS on held-out test bearings.

CLEAN-KILL outcome (P1 FAIL = "diffusion FPT is already calibrated") is a valid,
reportable result. We do NOT re-tune HI/threshold to manufacture a positive P1.

Implementation note (declared): each bearing HI is resampled to a common grid of
N_RS points so that first-passage rollouts are bounded and comparable across the
60x lifetime range (42..2538 snapshots). RUL is therefore in resampled-index
units. This is a neutral normalisation; it does not bias the falsifier. A final
paper would repeat in absolute-time units.

Usage:
  python diffusion_calib_diagnostic.py            # full: 5 seeds
  python diffusion_calib_diagnostic.py --fast     # quick pilot (1 seed, small)
"""
import os, sys, json, time, argparse, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
XJTU_NPZ  = os.path.join(HERE, '..', 'data', 'hi_trajectories.npz')
FEMTO_NPZ = os.path.join(HERE, '..', 'data', 'femto_hi_trajectories.npz')
RESULTS = os.path.join(HERE, '..', 'results')
os.makedirs(RESULTS, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- config ----------------
ap = argparse.ArgumentParser()
ap.add_argument('--fast', action='store_true')
ap.add_argument('--seeds', type=int, default=5)
ap.add_argument('--dataset', choices=['xjtu', 'femto', 'combined'], default='xjtu',
                help='which bearing fleet to run on')
ap.add_argument('--ablation', action='store_true',
                help='threshold-sweep ablation: naive-FPT vs direct-RUL calibration as the '
                     'failure threshold rises (isolates rollout-cause from extrapolation-cause)')
ap.add_argument('--fixed_d', action='store_true',
                help='hold the threshold constant across all seeds (control for the '
                     'per-seed threshold confound): D = 25th pctl of ALL terminal HI')
args = ap.parse_args()

N_RS      = 128           # resample every bearing HI to this many points
SMOOTH    = 5             # moving-average window on raw RMS
N0        = 5             # #healthy snapshots for baseline ratio
LOG_HI    = True          # model log(relative-ratio): tames flat-then-spike dynamics
MONO      = True          # monotone (cumulative-max) health indicator
ONSET     = 1.5           # degradation onset: first time relative ratio >= ONSET
L         = 16            # lookback window
H         = 8             # forecast block (rollout step)
TDIFF     = 100 if not args.fast else 50   # diffusion steps
EPOCHS    = 300 if not args.fast else 60
BATCH     = 128
LR        = 1e-3
K_PATHS   = 100 if not args.fast else 40   # sampled trajectories per point
FRACS     = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]   # inference phase-fractions (later phase)
MIN_DEGRAD = 1.5          # exclude bearings whose terminal log-ratio < this (non-degraders)
D_MARGIN  = 0.97          # shared threshold = D_MARGIN * min(train terminal HI)
ROLLOUT_CAP = int(2.5 * N_RS)   # censor horizon (monotone rollout -> few true censors)
N_SEEDS   = 1 if args.fast else args.seeds
SEEDS     = list(range(24))[:N_SEEDS]
REL_LEVELS = [round(x, 3) for x in np.linspace(0.05, 0.95, 19)]   # reliability-curve nominal levels

# ============================================================= data / HI
def moving_avg(x, w):
    if w <= 1: return x
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(x, size=w, mode='nearest')   # no edge zero-padding

def _npz_items():
    files = {'xjtu': [XJTU_NPZ], 'femto': [FEMTO_NPZ],
             'combined': [XJTU_NPZ, FEMTO_NPZ]}[args.dataset]
    raws, meta = {}, {}
    for fp in files:
        if not os.path.exists(fp):
            raise FileNotFoundError(f'{fp} missing — run its preprocessor first.')
        d = np.load(fp, allow_pickle=True)
        m = json.loads(str(d['meta']))
        for key in m:
            raws[key] = d[key].astype(np.float64); meta[key] = m[key]
    return raws, meta

def load_hi():
    """Returns G (key -> HI trajectory on N_RS grid, in the modelling space) and
       ONSET_IDX (key -> degradation-onset index on that grid)."""
    raws, meta = _npz_items()
    out, onset = {}, {}
    for key in list(meta):
        raw = raws[key]
        sm = moving_avg(raw, SMOOTH)
        base = sm[:N0].mean()
        g = sm / (base + 1e-12)              # relative degradation ratio, ~1 at start
        # resample to common grid
        xp = np.linspace(0, 1, len(g))
        xq = np.linspace(0, 1, N_RS)
        g_rs = np.interp(xq, xp, g)
        # degradation onset on the ratio (before log transform)
        oi = int(np.argmax(g_rs >= ONSET))
        if g_rs[oi] < ONSET:
            oi = N_RS // 2                    # fallback (shouldn't trigger)
        onset[key] = oi
        hi = np.log(np.maximum(g_rs, 1e-3)) if LOG_HI else g_rs
        if MONO:
            hi = np.maximum.accumulate(hi)   # monotone non-decreasing health index
        out[key] = hi.astype(np.float32)
    # exclude genuine non-degraders (insufficient run-to-failure signature)
    excl = [k for k in out if float(out[k].max()) < MIN_DEGRAD]
    for k in excl:
        del out[k], onset[k], meta[k]
    if excl:
        print('  excluded non-degraders:', excl)
    return out, meta, onset


ONSET_IDX = {}   # populated in main()

# ============================================================= threshold D
def calibrate_threshold(train_keys, G):
    """Shared 'significant-degradation' threshold = D_PCTL percentile of TRAIN
       bearings' terminal HI. A moderate shared level (not the per-unit peak) that
       most bearings reach, so the first-passage event is well-populated. Defined
       on train only; applied unchanged to held-out bearings."""
    return D_MARGIN * float(min(G[k].max() for k in train_keys))

def fpt(traj, D):
    """first index where traj >= D, else len(traj) (censored)."""
    idx = np.argmax(traj >= D)
    if traj[idx] < D:
        return len(traj)
    return int(idx)

# ============================================================= diffusion model
def timestep_embed(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=-1)

class Denoiser(nn.Module):
    """eps-prediction MLP: (noisy_target[H], cond[L], t_emb) -> eps[H]."""
    def __init__(self, L=L, H=H, temb=32, hid=128):
        super().__init__()
        self.temb = temb
        self.net = nn.Sequential(
            nn.Linear(H + L + temb, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, H),
        )
    def forward(self, y, cond, t):
        te = timestep_embed(t, self.temb)
        return self.net(torch.cat([y, cond, te], dim=-1))

class Diffusion:
    def __init__(self, tdiff=TDIFF):
        self.T = tdiff
        betas = torch.linspace(1e-4, 0.02, tdiff, device=DEVICE)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.acp = torch.cumprod(self.alphas, 0)          # alpha_bar
        self.acp_prev = torch.cat([torch.ones(1, device=DEVICE), self.acp[:-1]])

    def q_sample(self, y0, t, noise):
        a = self.acp[t][:, None]
        return torch.sqrt(a) * y0 + torch.sqrt(1 - a) * noise

    @torch.no_grad()
    def sample(self, model, cond, noise_scale=1.0):
        """ancestral sampling; cond (B,L) -> y0 (B,H). noise_scale inflates the
           injected reverse noise (used only if a sampler-level dispersion knob is
           wanted; P3 uses a post-hoc knob instead, see main)."""
        B = cond.shape[0]
        y = torch.randn(B, H, device=DEVICE)
        for i in reversed(range(self.T)):
            t = torch.full((B,), i, device=DEVICE, dtype=torch.long)
            eps = model(y, cond, t)
            a = self.alphas[i]; ab = self.acp[i]; ab_prev = self.acp_prev[i]
            beta = self.betas[i]
            mean = (y - beta / torch.sqrt(1 - ab) * eps) / torch.sqrt(a)
            if i > 0:
                var = beta * (1 - ab_prev) / (1 - ab)
                mean = mean + noise_scale * torch.sqrt(var) * torch.randn_like(y)
            y = mean
        return y

# ============================================================= train
def make_windows(keys, G):
    """Windows whose FORECAST target lies at/after degradation onset, so the model
       learns the rising regime rather than the long flat pre-onset stretch."""
    conds, tgts = [], []
    for k in keys:
        g = G[k]; onset = ONSET_IDX[k]
        for s in range(0, len(g) - L - H + 1):
            if s + L < onset:                 # forecast target before onset -> skip
                continue
            conds.append(g[s:s + L])
            tgts.append(g[s + L:s + L + H])
    return np.stack(conds), np.stack(tgts)

def train_diffusion(keys, G, norm, seed):
    mu, sd = norm
    C, Tg = make_windows(keys, G)
    C = (C - mu) / sd; Tg = (Tg - mu) / sd
    C = torch.tensor(C, device=DEVICE); Tg = torch.tensor(Tg, device=DEVICE)
    model = Denoiser().to(DEVICE)
    diff = Diffusion()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(C)
    g = torch.Generator(device='cpu').manual_seed(seed)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g).to(DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            y0 = Tg[idx]; cond = C[idx]
            t = torch.randint(0, diff.T, (len(idx),), device=DEVICE)
            noise = torch.randn_like(y0)
            yt = diff.q_sample(y0, t, noise)
            eps = model(yt, cond, t)
            loss = F.mse_loss(eps, noise)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, diff

# ============================================================= direct-RUL diffusion (CARD-style control)
# Not a novelty claim (direct conditional-diffusion regression = CARD, NeurIPS'22). This is a
# CONTROL arm: does the over-confidence come from the autoregressive FPT rollout, or is it
# intrinsic to diffusion-for-prognostics? Direct-RUL diffusion has no rollout / no first-passage.
class DenoiserRUL(nn.Module):
    def __init__(self, L=L, temb=32, hid=128):
        super().__init__()
        self.temb = temb
        self.net = nn.Sequential(
            nn.Linear(1 + L + temb, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, hid), nn.SiLU(),
            nn.Linear(hid, 1),
        )
    def forward(self, y, cond, t):
        te = timestep_embed(t, self.temb)
        return self.net(torch.cat([y, cond, te], dim=-1))

def build_rul_pairs(keys, G, D):
    """(window, rul) pairs across degradation-phase windows; rul threshold-consistent."""
    conds, ruls = [], []
    for k in keys:
        g = G[k]; onset = ONSET_IDX[k]; fptt = fpt(g, D)
        if fptt >= N_RS:
            continue
        for t in range(max(L, onset), fptt):
            conds.append(g[t - L:t]); ruls.append(fptt - t)
    if not conds:
        return np.zeros((0, L)), np.zeros((0,))
    return np.stack(conds), np.array(ruls, dtype=np.float64)

def train_direct_rul(keys, G, D, norm, seed):
    mu, sd = norm
    C, R = build_rul_pairs(keys, G, D)
    if len(C) == 0:
        return None
    rmu, rsd = float(R.mean()), float(R.std() + 1e-8)
    Cn = (C - mu) / sd; Rn = ((R - rmu) / rsd)[:, None]
    Cn = torch.tensor(Cn, device=DEVICE).float()
    Rn = torch.tensor(Rn, device=DEVICE).float()
    model = DenoiserRUL().to(DEVICE); diff = Diffusion()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(Cn); g = torch.Generator(device='cpu').manual_seed(seed + 777)
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g).to(DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            y0 = Rn[idx]; cond = Cn[idx]
            t = torch.randint(0, diff.T, (len(idx),), device=DEVICE)
            noise = torch.randn_like(y0)
            yt = diff.q_sample(y0, t, noise)
            loss = F.mse_loss(model(yt, cond, t), noise)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, diff, (rmu, rsd)

@torch.no_grad()
def sample_direct_rul(bundle, cond_raw, norm, K=K_PATHS):
    model, diff, (rmu, rsd) = bundle
    mu, sd = norm
    cond = torch.tensor(((cond_raw - mu) / sd), device=DEVICE).float()[None].repeat(K, 1)
    y = torch.randn(K, 1, device=DEVICE)
    for i in reversed(range(diff.T)):
        t = torch.full((K,), i, device=DEVICE, dtype=torch.long)
        eps = model(y, cond, t)
        a = diff.alphas[i]; ab = diff.acp[i]; ab_prev = diff.acp_prev[i]; beta = diff.betas[i]
        mean = (y - beta / torch.sqrt(1 - ab) * eps) / torch.sqrt(a)
        if i > 0:
            var = beta * (1 - ab_prev) / (1 - ab)
            mean = mean + torch.sqrt(var) * torch.randn_like(y)
        y = mean
    rul = y.cpu().numpy()[:, 0] * rsd + rmu
    return np.maximum(rul, 0.0)

# ============================================================= FPT rollout
OU_RHO = 0.9   # mean-reversion of the log-rate innovation (bounds horizon width-growth)

@torch.no_grad()
def rollout_rul(model, diff, cond_raw, t_now, D, norm, K=K_PATHS, noise_scale=1.0,
                innov=0.0):
    """From lookback cond_raw (L,) at index t_now, sample K trajectories forward,
       return array (K,) of predicted RUL (crossing_index - t_now), censored at CAP.

       innov>0 activates the UNCERTAINTY-PROPAGATING ROLLOUT (UPR). The diffusion still
       produces the per-step conditional mean; UPR modulates the increment MULTIPLICATIVELY
       by a mean-reverting (OU) log-rate innovation:
           delta_j = max(0, y_hat_j - level) * exp(w_j),   w_j = OU_RHO*w_{j-1} + innov*N(0,1)
       - multiplicative & symmetric in log-rate  -> some paths climb faster, some slower
         => SYMMETRIC crossing-time spread (removes the upward-only RUL-low bias);
       - OU mean-reversion => rate variation is STATIONARY (bounded), so interval width no
         longer blows up ~sqrt(horizon);
       - delta >= 0 keeps the health index monotone.
       The perturbed trajectory is fed forward, so uncertainty compounds through the model."""
    mu, sd = norm
    cond = torch.tensor(((cond_raw - mu) / sd), device=DEVICE).float()
    cond = cond[None].repeat(K, 1)                       # (K, L)
    pos = t_now
    crossed = np.full(K, -1, dtype=np.int64)
    level = np.full(K, float(cond_raw[-1]), dtype=np.float64)   # perturbed running level
    w = np.zeros(K, dtype=np.float64)                          # OU log-rate state
    hist = np.tile(cond_raw.astype(np.float64), (K, 1))       # (K, L) raw
    while pos < ROLLOUT_CAP and (crossed < 0).any():
        y = diff.sample(model, cond, noise_scale=noise_scale)   # (K,H) normalised
        y_raw = y.cpu().numpy() * sd + mu                        # (K,H) absolute mean
        block = np.empty((K, H), dtype=np.float64)
        for j in range(H):
            delta = np.maximum(0.0, y_raw[:, j] - level)        # base increment (monotone)
            if innov > 0.0:
                w = OU_RHO * w + innov * np.random.randn(K)      # OU innovation on log-rate
                delta = delta * np.exp(w)                        # multiplicative, symmetric
            level = level + delta
            block[:, j] = level
            step_pos = pos + 1 + j
            newly = (crossed < 0) & (level >= D)
            crossed[newly] = step_pos
        pos += H
        hist = np.concatenate([hist, block], axis=1)[:, -L:]
        cond = torch.tensor(((hist - mu) / sd), device=DEVICE).float()
    crossed[crossed < 0] = ROLLOUT_CAP
    rul = crossed - t_now
    return rul.astype(np.float64)

# ============================================================= MC-dropout forecaster (non-diffusion baseline)
# Non-diffusion probabilistic control: an MC-dropout MLP forecaster (dropout kept ON
# at inference) plugged into the IDENTICAL first-passage rollout. The ONLY change vs
# naive diffusion FPT is the generator family, so if this is ALSO over-confident the
# miscalibration is driven by the autoregressive rollout mechanism, not by diffusion
# specifically (answers the "diffusion-vs-itself" critique).
class MCForecaster(nn.Module):
    """cond[L] -> forecast[H]; dropout kept active for MC predictive sampling."""
    def __init__(self, L=L, H=H, hid=128, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(L, hid), nn.SiLU(), nn.Dropout(p),
            nn.Linear(hid, hid), nn.SiLU(), nn.Dropout(p),
            nn.Linear(hid, hid), nn.SiLU(), nn.Dropout(p),
            nn.Linear(hid, H),
        )
    def forward(self, cond):
        return self.net(cond)

def train_mc_forecaster(keys, G, norm, seed, p=0.2):
    mu, sd = norm
    C, Tg = make_windows(keys, G)
    C = (C - mu) / sd; Tg = (Tg - mu) / sd
    C = torch.tensor(C, device=DEVICE).float(); Tg = torch.tensor(Tg, device=DEVICE).float()
    model = MCForecaster(p=p).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    n = len(C); g = torch.Generator(device='cpu').manual_seed(seed + 555)
    model.train()
    for ep in range(EPOCHS):
        perm = torch.randperm(n, generator=g).to(DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            loss = F.mse_loss(model(C[idx]), Tg[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return model

@torch.no_grad()
def rollout_rul_mc(model, cond_raw, t_now, D, norm, K=K_PATHS):
    """First-passage rollout identical to rollout_rul, but each per-step forecast
       block is drawn from the MC-dropout forecaster (dropout ON -> K independent
       masks per batch) instead of the diffusion sampler. Monotone increment
       construction is byte-identical, isolating the generator family as the only
       difference from naive diffusion FPT."""
    model.train()   # keep dropout active (no_grad still disables gradients)
    mu, sd = norm
    cond = torch.tensor(((cond_raw - mu) / sd), device=DEVICE).float()[None].repeat(K, 1)
    pos = t_now
    crossed = np.full(K, -1, dtype=np.int64)
    level = np.full(K, float(cond_raw[-1]), dtype=np.float64)
    hist = np.tile(cond_raw.astype(np.float64), (K, 1))
    while pos < ROLLOUT_CAP and (crossed < 0).any():
        y_raw = model(cond).cpu().numpy() * sd + mu       # (K,H), K indep dropout draws
        block = np.empty((K, H), dtype=np.float64)
        for j in range(H):
            delta = np.maximum(0.0, y_raw[:, j] - level)  # monotone increment
            level = level + delta
            block[:, j] = level
            step_pos = pos + 1 + j
            newly = (crossed < 0) & (level >= D)
            crossed[newly] = step_pos
        pos += H
        hist = np.concatenate([hist, block], axis=1)[:, -L:]
        cond = torch.tensor(((hist - mu) / sd), device=DEVICE).float()
    crossed[crossed < 0] = ROLLOUT_CAP
    return (crossed - t_now).astype(np.float64)

# ============================================================= metrics
def crps_ens(samples, y):
    s = np.sort(samples)
    term1 = np.mean(np.abs(s - y))
    # E|X-X'| via sorted formula
    n = len(s)
    term2 = 2.0 * np.sum((2 * np.arange(1, n + 1) - n - 1) * s) / (n * n)
    return term1 - 0.5 * term2

def central_cov(samples, y, q):
    lo = np.quantile(samples, (1 - q) / 2)
    hi = np.quantile(samples, (1 + q) / 2)
    return float(lo <= y <= hi)

def point_metrics(samples, y):
    return {
        'pit': float(np.mean(samples <= y)),
        'cov50': central_cov(samples, y, 0.5),
        'cov90': central_cov(samples, y, 0.9),
        'crps': crps_ens(samples, y),
        'w90': float(np.quantile(samples, 0.95) - np.quantile(samples, 0.05)),
        'y': float(y),
    }

def disperse(samples, gamma):
    m = np.median(samples)
    return m + gamma * (samples - m)

# ============================================================= one seed
def split_bearings(keys, seed):
    """Stratified by operating condition (C1/C2/C3) so test/cal always get a mix of
       strong and weak risers rather than all-weak by chance."""
    rng = np.random.default_rng(seed)
    by_cond = {}
    for k in keys:
        by_cond.setdefault(k.split('_')[0], []).append(k)
    train, cal, test = [], [], []
    for c, ks in sorted(by_cond.items()):
        ks = list(ks); rng.shuffle(ks)
        # ~1 test, 1 cal, rest train per stratum; tiny strata fall back gracefully
        if len(ks) >= 3:
            test.append(ks[0]); cal.append(ks[1]); train.extend(ks[2:])
        elif len(ks) == 2:
            cal.append(ks[0]); train.append(ks[1])
        else:
            train.extend(ks)
    return train, cal, test

def eval_points(model, diff, keys, G, D, norm, noise_scale=1.0):
    """returns list of dicts with raw predictive samples per (bearing, fraction)."""
    pts = []
    for k in keys:
        g = G[k]
        eol = N_RS - 1
        onset = ONSET_IDX[k]
        fpt_true = fpt(g, D)                    # threshold-consistent event time
        if fpt_true >= N_RS:                    # curve never crosses D -> skip bearing
            continue
        for fr in FRACS:
            t = int(round(onset + fr * (fpt_true - onset)))  # phase up to the event
            if t < L or t >= fpt_true: continue
            rul_true = fpt_true - t
            cond_raw = g[t - L + 1:t + 1]
            samples = rollout_rul(model, diff, cond_raw, t, D, norm, noise_scale=noise_scale)
            pts.append({'key': k, 'frac': fr, 'rul_true': rul_true,
                        'samples': samples})
    return pts

def make_eval_inputs(keys, G, D):
    """(key, frac, rul_true, cond, t) per inference point — no rollout yet."""
    inps = []
    for k in keys:
        g = G[k]; onset = ONSET_IDX[k]; fptt = fpt(g, D)
        if fptt >= N_RS:
            continue
        for fr in FRACS:
            t = int(round(onset + fr * (fptt - onset)))
            if t < L or t >= fptt:
                continue
            inps.append({'key': k, 'frac': fr, 'rul_true': fptt - t,
                         'cond': g[t - L + 1:t + 1], 't': t})
    return inps

def rollout_inputs(model, diff, inps, D, norm, innov=0.0, K=K_PATHS):
    pts = []
    for q in inps:
        s = rollout_rul(model, diff, q['cond'], q['t'], D, norm, K=K, innov=innov)
        pts.append({'key': q['key'], 'frac': q['frac'],
                    'rul_true': q['rul_true'], 'samples': s})
    return pts

def fit_innovation(model, diff, cal_inps, D, norm):
    """Fit the single propagated-innovation scale a on the cal fleet, minimising
       calibration error at the 90% and 50% levels. One physical parameter,
       mechanism-constrained -> far more data-efficient than nonparametric conformal."""
    grid = [0.0, 0.05, 0.1, 0.2, 0.4]   # OU log-rate innovation scale (drift-mode UPR)
    K_FIT = min(K_PATHS, 50)
    best_a, best_err = 0.0, 1e18
    for a in grid:
        pts = rollout_inputs(model, diff, cal_inps, D, norm, innov=a, K=K_FIT)
        ag, _ = summarise(pts)
        err = abs(ag['cov90'] - 0.9) + abs(ag['cov50'] - 0.5)
        if err < best_err:
            best_err, best_a = err, a
    return best_a

def summarise(pts, gamma=1.0):
    rows = []
    for p in pts:
        s = disperse(p['samples'], gamma) if gamma != 1.0 else p['samples']
        m = point_metrics(s, p['rul_true'])
        m['frac'] = p['frac']
        rows.append(m)
    agg = {
        'cov50': np.mean([r['cov50'] for r in rows]),
        'cov90': np.mean([r['cov90'] for r in rows]),
        'crps':  np.mean([r['crps'] for r in rows]),
        'w90':   np.mean([r['w90'] for r in rows]),
        'n': len(rows),
    }
    # early vs late thirds by life fraction
    early = [r['cov90'] for r in rows if r['frac'] <= 0.5]
    late  = [r['cov90'] for r in rows if r['frac'] >= 0.8]
    agg['cov90_early'] = float(np.mean(early)) if early else float('nan')
    agg['cov90_late']  = float(np.mean(late)) if late else float('nan')
    # PIT-based, cancellation-free calibration error (committed metric in PROTOCOL)
    pits = np.array([r['pit'] for r in rows])
    agg['pits'] = pits.tolist()
    u = (np.arange(1, len(pits) + 1) - 0.5) / len(pits)   # ideal uniform order stats
    agg['pit_ks'] = float(np.max(np.abs(np.sort(pits) - u))) if len(pits) else float('nan')
    agg['pit_mean'] = float(pits.mean()) if len(pits) else float('nan')
    # QICE (mean quantile-interval coverage error) + reliability curve (obs vs nominal)
    obs = [float(np.mean(pits <= lv)) for lv in REL_LEVELS] if len(pits) else \
          [float('nan')] * len(REL_LEVELS)
    agg['reliability'] = obs
    agg['qice'] = float(np.mean([abs(o - lv) for o, lv in zip(obs, REL_LEVELS)])) \
        if len(pits) else float('nan')
    return agg, rows

# ============================================================= conformal-FPT arm
def stage_of(frac):
    if frac <= 0.5: return 'early'
    if frac <= 0.7: return 'mid'
    return 'late'

def conformal_eval(cal_pts, test_pts, alpha, adaptive=True, min_grp=4):
    """Censoring-aware, degradation-stage-adaptive CQR recalibration of the diffusion
       ensemble. CQR score E = max(q_lo - y, y - q_hi) on the ensemble's alpha/2 &
       1-alpha/2 quantiles; per-stage (Mondrian) finite-sample conformal quantile,
       falling back to the pooled score when a stage group is too small. The score is
       signed so conformal can BOTH widen (fix over-confidence) and tighten (fix the
       over-wide early points) — unlike the symmetric global-gamma scaling."""
    def qpair(s):
        return float(np.quantile(s, alpha / 2)), float(np.quantile(s, 1 - alpha / 2))
    cal_scores = {}
    for p in cal_pts:
        lo, hi = qpair(p['samples']); y = p['rul_true']
        cal_scores.setdefault(stage_of(p['frac']), []).append(max(lo - y, y - hi))
    pooled = [e for v in cal_scores.values() for e in v]
    def qhat(stage):
        grp = cal_scores.get(stage, [])
        pool = grp if (adaptive and len(grp) >= min_grp) else pooled
        n = len(pool)
        if n == 0: return 0.0
        k = min(n, int(np.ceil((n + 1) * (1 - alpha))))
        return float(np.sort(pool)[k - 1])
    covs, widths = [], []
    for p in test_pts:
        lo, hi = qpair(p['samples']); y = p['rul_true']
        Q = qhat(stage_of(p['frac']))
        L, U = lo - Q, hi + Q
        covs.append(float(L <= y <= U)); widths.append(U - L)
    return float(np.mean(covs)), float(np.mean(widths))

def run_seed(seed, G, meta, fixed_D=None):
    torch.manual_seed(seed); np.random.seed(seed)
    keys = list(meta.keys())
    train, cal, test = split_bearings(keys, seed)
    C, _ = make_windows(train, G)
    mu, sd = float(C.mean()), float(C.std() + 1e-8)
    norm = (mu, sd)
    D = fixed_D if fixed_D is not None else calibrate_threshold(train, G)
    model, diff = train_diffusion(train, G, norm, seed)

    test_inps = make_eval_inputs(test, G, D)
    cal_inps  = make_eval_inputs(cal, G, D)

    # naive diffusion rollout (innov=0)
    test_pts = rollout_inputs(model, diff, test_inps, D, norm, innov=0.0)
    cal_pts  = rollout_inputs(model, diff, cal_inps,  D, norm, innov=0.0)
    naive_agg, _ = summarise(test_pts, gamma=1.0)

    # baseline 1: global-gamma dispersion, tuned on cal
    gammas = np.linspace(0.5, 6.0, 23)
    best_g, best_err = 1.0, 1e9
    for gm in gammas:
        a, _ = summarise(cal_pts, gamma=gm)
        e = abs(a['cov90'] - 0.9)
        if e < best_err:
            best_err, best_g = e, gm
    corr_agg, _ = summarise(test_pts, gamma=best_g)

    # baseline 2: conformal-FPT (cal-calibrated)
    cf90_cov, cf90_w = conformal_eval(cal_pts, test_pts, 0.10)
    cf50_cov, _      = conformal_eval(cal_pts, test_pts, 0.50)
    conformal = {'cov90': cf90_cov, 'cov50': cf50_cov, 'w90': cf90_w}

    # OURS: uncertainty-propagating rollout, innovation a* fit on cal
    a_star = fit_innovation(model, diff, cal_inps, D, norm)
    upr_pts = rollout_inputs(model, diff, test_inps, D, norm, innov=a_star)
    upr_agg, _ = summarise(upr_pts, gamma=1.0)

    # CONTROL: direct-RUL diffusion (CARD-style, no rollout / no FPT)
    dr_bundle = train_direct_rul(train, G, D, norm, seed)
    dr_pts = [{'key': q['key'], 'frac': q['frac'], 'rul_true': q['rul_true'],
               'samples': sample_direct_rul(dr_bundle, q['cond'], norm)}
              for q in test_inps]
    directrul_agg, _ = summarise(dr_pts, gamma=1.0)

    # NON-DIFFUSION BASELINE: MC-dropout forecaster through the SAME FPT rollout
    mc_model = train_mc_forecaster(train, G, norm, seed)
    mc_pts = [{'key': q['key'], 'frac': q['frac'], 'rul_true': q['rul_true'],
               'samples': rollout_rul_mc(mc_model, q['cond'], q['t'], D, norm)}
              for q in test_inps]
    mcdrop_agg, _ = summarise(mc_pts, gamma=1.0)

    return {
        'seed': seed, 'D': D, 'gamma': float(best_g), 'a_star': float(a_star),
        'train': train, 'cal': cal, 'test': test,
        'naive': naive_agg, 'corrected': corr_agg, 'conformal': conformal,
        'upr': upr_agg, 'directrul': directrul_agg, 'mcdrop': mcdrop_agg,
    }

def ablation_dsweep(G, meta):
    """Threshold-sweep ablation. For each seed: train the trajectory diffusion once; then at a
       range of failure thresholds D (percentiles of train terminal HI, low->high = easy->
       extrapolation-stressed), evaluate BOTH naive-FPT and a direct-RUL diffusion retrained
       for that D. Isolates causes:
         - if naive-FPT is miscalibrated at ALL D but direct-RUL is calibrated at LOW D and
           degrades at HIGH D  ->  rollout hurts everywhere; extrapolation hurts both at high D.
    """
    P_GRID = [0.1, 0.25, 0.4, 0.55, 0.7]
    seeds = SEEDS if len(SEEDS) > 1 else [0]
    rows = {p: {'naive': [], 'direct': []} for p in P_GRID}
    t0 = time.time()
    for sd in seeds:
        torch.manual_seed(sd); np.random.seed(sd)
        keys = list(meta.keys())
        train, cal, test = split_bearings(keys, sd)
        C, _ = make_windows(train, G)
        mu, s = float(C.mean()), float(C.std() + 1e-8); norm = (mu, s)
        traj_model, diff = train_diffusion(train, G, norm, sd)
        term = [float(G[k].max()) for k in train]
        for p in P_GRID:
            D = float(np.quantile(term, p))
            test_inps = make_eval_inputs(test, G, D)
            if not test_inps:
                continue
            nv = summarise(rollout_inputs(traj_model, diff, test_inps, D, norm, innov=0.0))[0]
            drb = train_direct_rul(train, G, D, norm, sd)
            if drb is None:
                continue
            dpts = [{'key': q['key'], 'frac': q['frac'], 'rul_true': q['rul_true'],
                     'samples': sample_direct_rul(drb, q['cond'], norm)} for q in test_inps]
            dr = summarise(dpts)[0]
            rows[p]['naive'].append((nv['cov90'], nv['cov50'], nv['pit_ks'], nv['qice']))
            rows[p]['direct'].append((dr['cov90'], dr['cov50'], dr['pit_ks'], dr['qice']))
        print(f'  seed {sd} done [{int(time.time()-t0)}s]', flush=True)

    print('\n===== THRESHOLD-SWEEP ABLATION (mean over seeds) =====')
    print(f'{"p":>5} {"arm":>7} | {"cov90":>6} {"cov50":>6} {"pit_ks":>6} {"qice":>6}')
    out = {}
    for p in P_GRID:
        out[str(p)] = {}
        for arm in ('naive', 'direct'):
            a = np.array(rows[p][arm])
            m = a.mean(0) if len(a) else [float('nan')] * 4
            out[str(p)][arm] = {'cov90': float(m[0]), 'cov50': float(m[1]),
                                'pit_ks': float(m[2]), 'qice': float(m[3]), 'n': len(a)}
            print(f'{p:>5} {arm:>7} | {m[0]:6.2f} {m[1]:6.2f} {m[2]:6.2f} {m[3]:6.2f}')
    fn = os.path.join(RESULTS, f'ablation_dsweep_{args.dataset}.json')
    with open(fn, 'w') as f:
        json.dump({'p_grid': P_GRID, 'results': out, 'n_seeds': len(seeds)}, f, indent=2)
    print(f'saved -> {fn}  [{int(time.time()-t0)}s]')


def main():
    G, meta, onset = load_hi()
    ONSET_IDX.update(onset)
    if args.ablation:
        print(f'ABLATION | dataset={args.dataset} | {len(meta)} bearings | seeds={len(SEEDS)}')
        ablation_dsweep(G, meta)
        return
    _main_benchmark(G, meta, onset)

def _main_benchmark(G, meta, onset):
    ONSET_IDX.update(onset)
    print(f'Loaded {len(meta)} bearings. DEVICE={DEVICE}  fast={args.fast}  '
          f'LOG_HI={LOG_HI}')
    print('EOL range (snaps):', min(m["EOL"] for m in meta.values()),
          '..', max(m["EOL"] for m in meta.values()))
    print('onset idx (N_RS grid):', {k: onset[k] for k in list(onset)[:15]})
    fixed_D = None
    if args.fixed_d:
        fixed_D = float(np.quantile([float(G[k].max()) for k in meta], 0.25))
        print(f'FIXED-D CONTROL: D held constant = {fixed_D:.3f} for all seeds')
    t0 = time.time()
    runs = []
    for sd in SEEDS:
        r = run_seed(sd, G, meta, fixed_D=fixed_D)
        runs.append(r)
        n = r['naive']; c = r['corrected']; cf = r['conformal']; up = r['upr']; dr = r['directrul']; mc = r['mcdrop']
        print(f"seed {sd}: D={r['D']:.2f} a*={r['a_star']:.2f} | "
              f"NAIVE c90={n['cov90']:.2f} c50={n['cov50']:.2f} ks={n['pit_ks']:.2f} | "
              f"CONF c90={cf['cov90']:.2f} | "
              f"DIRECT c90={dr['cov90']:.2f} c50={dr['cov50']:.2f} ks={dr['pit_ks']:.2f} "
              f"[{int(time.time()-t0)}s]", flush=True)

    # aggregate across seeds
    def col(path, key):
        return np.array([r[path][key] for r in runs])
    verdict = {
        'n_seeds': len(runs),
        'naive_cov90_mean': float(col('naive','cov90').mean()),
        'naive_cov90_std':  float(col('naive','cov90').std()),
        'naive_cov50_mean': float(col('naive','cov50').mean()),
        'naive_crps_mean':  float(col('naive','crps').mean()),
        'naive_w90_mean':   float(col('naive','w90').mean()),
        'cov90_early_mean': float(np.nanmean(col('naive','cov90_early'))),
        'cov90_late_mean':  float(np.nanmean(col('naive','cov90_late'))),
        'corr_cov90_mean':  float(col('corrected','cov90').mean()),
        'corr_crps_mean':   float(col('corrected','crps').mean()),
        'corr_w90_mean':    float(col('corrected','w90').mean()),
        'gamma_mean':       float(np.mean([r['gamma'] for r in runs])),
        'conf_cov90_mean':  float(col('conformal','cov90').mean()),
        'conf_cov90_std':   float(col('conformal','cov90').std()),
        'conf_cov50_mean':  float(col('conformal','cov50').mean()),
        'conf_w90_mean':    float(col('conformal','w90').mean()),
        'upr_cov90_mean':   float(col('upr','cov90').mean()),
        'upr_cov90_std':    float(col('upr','cov90').std()),
        'upr_cov50_mean':   float(col('upr','cov50').mean()),
        'upr_w90_mean':     float(col('upr','w90').mean()),
        'upr_pit_ks_mean':  float(col('upr','pit_ks').mean()),
        'a_star_mean':      float(np.mean([r['a_star'] for r in runs])),
        'direct_cov90_mean': float(col('directrul','cov90').mean()),
        'direct_cov90_std':  float(col('directrul','cov90').std()),
        'direct_cov50_mean': float(col('directrul','cov50').mean()),
        'direct_crps_mean':  float(col('directrul','crps').mean()),
        'direct_w90_mean':   float(col('directrul','w90').mean()),
        'direct_pit_ks_mean': float(col('directrul','pit_ks').mean()),
        'mcdrop_cov90_mean': float(col('mcdrop','cov90').mean()),
        'mcdrop_cov90_std':  float(col('mcdrop','cov90').std()),
        'mcdrop_cov50_mean': float(col('mcdrop','cov50').mean()),
        'mcdrop_crps_mean':  float(col('mcdrop','crps').mean()),
        'mcdrop_w90_mean':   float(col('mcdrop','w90').mean()),
        'mcdrop_pit_ks_mean': float(col('mcdrop','pit_ks').mean()),
    }
    # QICE (mean quantile-interval coverage error) + averaged reliability curves
    for arm in ('naive', 'upr', 'directrul', 'mcdrop'):
        verdict[f'{arm}_qice_mean'] = float(col(arm, 'qice').mean())
    verdict['rel_levels'] = REL_LEVELS
    for arm in ('naive', 'directrul'):
        curves = np.array([r[arm]['reliability'] for r in runs])   # (seeds, levels)
        verdict[f'{arm}_reliability'] = np.nanmean(curves, axis=0).tolist()
    dcov = col('directrul','cov90')
    verdict['direct_cov90_ci95'] = float(1.96 * dcov.std() / max(1, math.sqrt(len(dcov))))
    verdict['DIRECT_fixes_it'] = bool(
        0.85 <= verdict['direct_cov90_mean'] <= 0.95
        and abs(verdict['direct_cov50_mean'] - 0.5) < abs(verdict['naive_cov50_mean'] - 0.5)
    )
    # OURS decision: robustly in [0.85,0.95] at 90% AND Cov@0.5 nearer 0.5 than naive
    uc = col('upr','cov90')
    verdict['upr_cov90_ci95'] = float(1.96 * uc.std() / max(1, math.sqrt(len(uc))))
    verdict['UPR_fixes_it'] = bool(
        0.85 <= verdict['upr_cov90_mean'] <= 0.95
        and abs(verdict['upr_cov50_mean'] - 0.5) < abs(verdict['naive_cov50_mean'] - 0.5)
    )
    # conformal decision: robustly in [0.85,0.95] at 90% AND closer to 0.5 at 50%
    cc = col('conformal','cov90')
    verdict['conf_cov90_ci95'] = float(1.96 * cc.std() / max(1, math.sqrt(len(cc))))
    verdict['CONFORMAL_fixes_it'] = bool(
        0.85 <= verdict['conf_cov90_mean'] <= 0.95
        and abs(verdict['conf_cov50_mean'] - 0.5) < abs(verdict['naive_cov50_mean'] - 0.5)
    )
    # pooled PIT across all seeds/points -> cancellation-free calibration error
    all_pits = np.array([p for r in runs for p in r['naive']['pits']])
    u = (np.arange(1, len(all_pits) + 1) - 0.5) / len(all_pits)
    verdict['pit_ks_pooled'] = float(np.max(np.abs(np.sort(all_pits) - u)))
    verdict['pit_mean_pooled'] = float(all_pits.mean())
    verdict['pit_ks_seedmean'] = float(col('naive','pit_ks').mean())
    # pre-specified decisions
    nc = col('naive','cov90')
    ci = 1.96 * nc.std() / max(1, math.sqrt(len(nc)))
    verdict['P1_miscalibrated'] = bool(
        (verdict['naive_cov90_mean'] < 0.80 or verdict['naive_cov90_mean'] > 0.98)
    )
    # P1 supplement: PIT KS significantly non-uniform (structured miscalibration
    # that average coverage can mask). KS 5% crit ~ 1.36/sqrt(n).
    ks_crit = 1.36 / math.sqrt(max(1, len(all_pits)))
    verdict['P1_pit_miscalibrated'] = bool(verdict['pit_ks_pooled'] > ks_crit)
    verdict['pit_ks_crit5pct'] = float(ks_crit)
    verdict['P2_structured'] = bool(
        verdict['cov90_late_mean'] < verdict['cov90_early_mean'] - 0.05
    )
    dcrps = col('naive','crps') - col('corrected','crps')
    verdict['P3_correctable'] = bool(
        (0.85 <= verdict['corr_cov90_mean'] <= 0.95) and (dcrps.mean() > 0)
        and (verdict['corr_w90_mean'] <= 2.0 * verdict['naive_w90_mean'])
    )
    verdict['naive_cov90_ci95'] = float(ci)
    verdict['dcrps_mean'] = float(dcrps.mean())

    out = {'verdict': verdict, 'runs': [
        {k: (v if k not in ('train','cal','test') else v) for k, v in r.items()}
        for r in runs]}
    suffix = '_fast' if args.fast else ('_fixedD' if args.fixed_d else '')
    fn = os.path.join(RESULTS, f'diagnostic_{args.dataset}{suffix}.json')
    with open(fn, 'w') as f:
        json.dump(out, f, indent=2)
    print('\n===== VERDICT =====')
    for k, v in verdict.items():
        print(f'  {k}: {v}')
    print(f'saved -> {fn}  [{int(time.time()-t0)}s]')

if __name__ == '__main__':
    main()
