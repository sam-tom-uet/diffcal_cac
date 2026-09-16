# Pre-Specified Protocol — Calibration of Diffusion-Generated Trajectory RUL

**Study:** DiffCal — a standalone calibration audit (no new algorithm claimed).
**Datasets:** XJTU-SY bearings (primary), PRONOSTIA/FEMTO (confirmation).
**Method family:** score-based / denoising diffusion (generative).
**Date committed:** 2026-07-20 (before any method build).

This protocol — the health-indicator construction, failure-threshold rule, data splits,
metrics and the pass/fail falsifiers — was fixed before any model was trained or any result
was seen. It is included verbatim so the pre-specification can be inspected.

---

## 1. The gap (from the 2026-07-20 literature check)

Two literatures that do not intersect:
- **Classical first-passage-time (FPT) RUL** (Wiener / gamma / inverse-Gaussian): gives a
  *calibrated* RUL distribution but from a **rigid parametric** degradation path family.
  (Here "diffusion" means the Wiener diffusion coefficient, not a generative diffusion model.)
- **Generative denoising-diffusion degradation models** (temporal latent diffusion for trend
  forecasting; defect-guided conditional-diffusion digital twins): **flexible learned**
  trajectories, but generation is used for point/trend forecasting or data augmentation —
  the calibration of the resulting RUL distribution is not checked.
- Generative-UQ RUL that does exist is **VAE / BNN / LSTM-UQ**, not score-based diffusion
  trajectory ensembles + FPT.

The exact intersection — *calibration/coverage of an FPT RUL distribution derived from
denoising-diffusion-generated trajectory ensembles* — has no direct prior hit. The gap is open.

**A gap is not a paper.** It is only worth building on if the following pre-specified
pathology holds. If it does not, the study is reported as a clean null result
("diffusion FPT is already calibrated").

---

## 2. Setup (fixed before seeing results)

- **HI (health indicator):** per-snapshot RMS of the horizontal vibration channel,
  smoothed (moving average, win=5). Relative degradation ratio `g_t = HI_t / baseline_b`,
  where `baseline_b` = mean HI over the first `N0=5` healthy snapshots of bearing `b`
  (uses only early healthy data, so no end-of-life leakage).
- **Failure threshold `D`:** a single constant on the relative ratio `g`, selected on
  **training bearings only** (value that makes the FPT-implied end-of-life best match the true
  end-of-life, MAE). Applied unchanged to held-out bearings. Cross-bearing threshold
  uncertainty is real and is part of what makes the FPT distribution non-trivial; we do NOT
  tune per-test-bearing.
- **True label:** `RUL_true(t) = EOL_b - t` (in resampled grid-step units; see setup note).
- **Generator:** conditional 1-D denoising diffusion (DDPM). Condition = lookback HI window
  of length `L`; target = forecast HI window of length `H`. Blockwise autoregressive rollout
  to failure. `K` sampled trajectories per inference point, each crossing `D` at some step,
  give `K` samples of predicted EOL and hence the empirical RUL predictive distribution.
- **Splits:** bearing-level. Train / calibration / test disjoint bearings. `S` seeds over
  {split assignment + model init + sampling}. Report mean +/- std across seeds.
- **Inference points:** several `t` per test bearing spanning early-to-late life.
- **Setup note:** each bearing's HI is resampled to a common grid of `N=128` points so that
  first-passage rollouts are bounded and comparable across the wide lifetime range; RUL is
  therefore expressed in resampled grid-step units. This is a neutral normalisation.

## 3. Metrics (fixed)

- **PIT** = `F_pred(RUL_true)`; calibrated iff PIT ~ Uniform(0,1). Report PIT histogram and
  KS-distance to Uniform.
- **Coverage@q** = fraction of points where `RUL_true` falls in the central-`q` predictive
  interval; nominal `q in {0.5, 0.9}`.
- **CRPS** of the predictive distribution vs `RUL_true` (proper score; sharpness x calibration).
- **Interval width** (sharpness) at `q=0.9`.

## 4. Pre-committed pass/fail (the falsifier)

**P1 — Miscalibration exists (primary).**
PASS if the diffusion FPT RUL distribution is robustly **mis-covered**:
`Coverage@0.9 < 0.80` (over-confident) OR `> 0.98` (under-confident),
mean over seeds AND the seed-wise CI excludes the nominal 0.90.
FAIL => clean null ("diffusion FPT is already calibrated"). Report and stop; do NOT
re-tune the HI/threshold to manufacture miscalibration.

**P2 — Structure (makes it a *mechanism*, not just noise).**
PASS if miscalibration is **structured & reproducible**: coverage in the **late-life third**
is significantly worse than the **early-life third** (paired over bearings, CI excludes 0),
OR coverage degrades monotonically as the sampler diversity / `K` is reduced.

**P3 — Correctable (there is a method to build).**
Candidate *simple* correction (diagnostic-stage only): post-hoc **dispersion recalibration** —
inflate the predictive spread by a factor `gamma` tuned on the **calibration bearings** to hit
`Coverage@0.9 ~ 0.90`.
PASS if, on **held-out test bearings**, the correction brings `Coverage@0.9` into `[0.85, 0.95]`
AND lowers CRPS vs the naive diffusion, with only a modest sharpness cost. Multi-seed, CI on
delta-CRPS excludes 0 in the correction's favour.

## 5. Decision table

| P1 | P2 | P3 | Verdict |
|----|----|----|---------|
| FAIL | — | — | Clean null — diffusion FPT already calibrated. Report and stop. |
| PASS | FAIL | — | Weak: unstructured miscalibration. Report as null-ish; reconsider. |
| PASS | PASS | FAIL | Opening, no method yet — pathology real & structured but the simple fix fails; a principled estimator would be the follow-on contribution. |
| PASS | PASS | PASS | Pathology real, structured, fixable — the method is the principled version of the P3 correction. |

## 6. Anti-overfitting commitments

- HI = RMS-relative-ratio and threshold-on-train-only are **fixed here**; they will not be
  swapped to chase a positive P1.
- If P1 fails, the study is reported as a null — no re-tuning of data-generating knobs to
  induce miscalibration.
- Sharpness cost is reported even when coverage improves (no coverage-by-blowup wins).
- All results carried at multi-seed with CIs; single-seed positives are not claimed.
