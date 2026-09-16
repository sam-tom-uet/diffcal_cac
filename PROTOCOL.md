# Pre-Specified Protocol — Calibration of Diffusion First-Passage-Time RUL

**Study:** calibration audit of diffusion-generated first-passage-time (FPT) RUL
distributions on rolling bearings.
**Datasets:** XJTU-SY (primary), PRONOSTIA/FEMTO (confirmation).
**Committed:** 2026-07-20, before any model was trained.

The health-indicator construction, failure-threshold rule, data splits, metrics and
acceptance criteria below were fixed prior to modelling and are reported here verbatim.

## 1. Setup

- **Health indicator (HI):** per-snapshot RMS of the horizontal vibration channel,
  smoothed with a length-5 moving average. Relative degradation ratio
  `g_t = HI_t / baseline_b`, where `baseline_b` is the mean HI over the first `N0 = 5`
  healthy snapshots of bearing `b` (early healthy data only; no end-of-life leakage).
- **Failure threshold `D`:** a single constant on the relative ratio `g`, selected on
  **training bearings only** (the value whose FPT-implied end-of-life best matches the true
  end-of-life, MAE) and applied unchanged to held-out bearings. No per-test-bearing tuning.
- **Label:** `RUL_true(t) = EOL_b - t`, in resampled grid-step units.
- **Generator:** conditional 1-D denoising diffusion (DDPM). Condition = lookback HI window
  of length `L`; target = forecast window of length `H`; blockwise autoregressive rollout to
  failure. `K` sampled trajectories give `K` first-passage times and hence the empirical RUL
  predictive distribution.
- **Splits:** bearing-level, disjoint train/calibration/test. `S` seeds over split
  assignment, model initialisation and sampling; results reported as mean ± std over seeds.
- **Normalisation:** each bearing HI is resampled to a common grid of `N = 128` points so
  first-passage rollouts are bounded and comparable across the wide lifetime range; RUL is
  therefore expressed in resampled grid-step units.

## 2. Metrics

- **PIT** `= F_pred(RUL_true)`; calibrated iff PIT ~ Uniform(0,1). Report the PIT histogram
  and the KS distance to Uniform.
- **Coverage@q:** fraction of points where `RUL_true` lies in the central-`q` predictive
  interval, for `q ∈ {0.5, 0.9}`.
- **CRPS** of the predictive distribution against `RUL_true`.
- **Interval width** at `q = 0.9` (sharpness).

## 3. Pre-specified acceptance criteria

- **A1 — Miscalibration.** The naive diffusion-FPT distribution is mis-covered:
  `Coverage@0.9 < 0.80` or `> 0.98` (mean over seeds, seed-wise CI excluding 0.90). If not
  met, the result is reported as calibrated (a null finding); the HI and threshold are not
  re-tuned to induce miscalibration.
- **A2 — Structure.** Miscalibration is structured and reproducible: late-life coverage is
  significantly worse than early-life coverage (paired over bearings, CI excluding 0), or
  coverage degrades monotonically as sampler diversity `K` is reduced.
- **A3 — Correctability.** A post-hoc dispersion recalibration (spread scaled by a factor
  tuned on the calibration bearings) brings held-out `Coverage@0.9` into `[0.85, 0.95]` and
  lowers CRPS versus the naive distribution, with only a modest sharpness cost (multi-seed,
  CI on ΔCRPS excluding 0).

## 4. Reporting commitments

- HI (RMS relative ratio) and the train-only threshold are fixed and not swapped post hoc.
- Sharpness cost is reported alongside any coverage improvement.
- All results are multi-seed with confidence intervals; single-seed outcomes are not claimed.
