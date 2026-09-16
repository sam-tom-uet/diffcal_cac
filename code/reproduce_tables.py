"""
reproduce_tables.py -- regenerate the paper's numeric tables from the result JSONs.

Reproduces, from results/diagnostic_combined.json (matched threshold) and
results/diagnostic_combined_fixedD.json (fixed-D control):

  * Table I  (tab:overconf) -- calibration of the naive diffusion-FPT distribution.
       Cov@0.5, Cov@0.9 are mean +/- std over seeds; PIT-KS is POOLED over all test
       points (verdict.pit_ks_pooled); QICE is the seed mean (verdict.naive_qice_mean).
  * Table II (tab:arms) -- recalibration benchmark, every arm.
       Cov@0.9 and matched PIT-KS are mean +/- std over seeds; Cov@0.5, CRPS and the
       fixed-D PIT-KS are seed means. (Table II's PIT-KS is the per-seed mean, which
       differs by construction from Table I's pooled PIT-KS.)
  * Paired significance -- naive vs. direct-RUL (paired t-test + Wilcoxon), the claim
       behind "direct-RUL is better calibrated at every threshold".

Numbers here should match the CAC 2026 camera-ready Tables I-II. Aggregation choices
(pooled vs. per-seed PIT-KS, QICE definition) follow the table captions verbatim.

Run:  python reproduce_tables.py
"""
import json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')

ARMS = [('Naive FPT', 'naive'), ('MC-dropout FPT', 'mcdrop'),
        ('Global-gamma', 'corrected'), ('Stage-conformal', 'conformal'),
        ('UPR', 'upr'), ('Direct-RUL', 'directrul')]


def load(fname):
    return json.load(open(os.path.join(RES, fname)))


def ms(runs, key, field):
    vals = [r[key][field] for r in runs if field in r[key]]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals))


def pm(t, dec=2):
    return ' n/a ' if t is None else f'{t[0]:.{dec}f}+/-{t[1]:.{dec}f}'


def pt(t, dec=2):
    return 'n/a' if t is None else f'{t[0]:.{dec}f}'


def table1(name, j):
    v = j['verdict']
    c5 = (v['naive_cov50_mean'], j_std(j, 'naive', 'cov50'))
    c9 = (v['naive_cov90_mean'], v['naive_cov90_std'])
    print(f'{name:<10}{pm(c5):>14}{pm(c9):>14}'
          f'{v["pit_ks_pooled"]:>9.2f}{v["naive_qice_mean"]:>8.2f}')


def j_std(j, key, field):
    return float(np.std([r[key][field] for r in j['runs'] if field in r[key]]))


def main():
    Jm = load('diagnostic_combined.json')
    Jf = load('diagnostic_combined_fixedD.json')
    Rm, Rf = Jm['runs'], Jf['runs']

    print(f'\n== Table I -- naive diffusion-FPT calibration '
          f'(n_matched={len(Rm)}, n_fixed={len(Rf)}) ==')
    print(f'{"Protocol":<10}{"Cov@0.5":>14}{"Cov@0.9":>14}{"PIT-KS":>9}{"QICE":>8}'
          '   (PIT-KS pooled, QICE seed-mean)')
    table1('Matched', Jm)
    table1('Fixed', Jf)

    print(f'\n== Table II -- recalibration benchmark ==')
    print(f'{"Method":<16}{"Cov@0.9(M)":>14}{"Cov@0.5":>9}{"PIT-KS(M)":>14}'
          f'{"CRPS":>7}   {"Cov@0.9(F)":>14}{"Cov@0.5":>9}{"PIT-KS(F)":>10}')
    for lab, k in ARMS:
        c9m, c5m = ms(Rm, k, 'cov90'), ms(Rm, k, 'cov50')
        ksm, crm = ms(Rm, k, 'pit_ks'), ms(Rm, k, 'crps')
        c9f, c5f = ms(Rf, k, 'cov90'), ms(Rf, k, 'cov50')
        ksf = ms(Rf, k, 'pit_ks')
        print(f'{lab:<16}{pm(c9m):>14}{pt(c5m):>9}{pm(ksm):>14}'
              f'{pt(crm,1):>7}   {pm(c9f):>14}{pt(c5f):>9}{pt(ksf):>10}')

    print('\n== Paired significance: naive vs. direct-RUL ==')
    for tag, R in [('MATCHED', Rm), ('FIXED', Rf)]:
        print(f'-- {tag} (n={len(R)}) --')
        for field in ['cov90', 'cov50', 'pit_ks']:
            nv = np.array([r['naive'][field] for r in R])
            dr = np.array([r['directrul'][field] for r in R])
            t = stats.ttest_rel(nv, dr)
            line = (f'  {field:<7} naive {nv.mean():.3f}+/-{nv.std():.3f}  '
                    f'direct {dr.mean():.3f}+/-{dr.std():.3f}  paired-t p={t.pvalue:.2e}')
            if not np.allclose(nv, dr):
                line += f'  wilcoxon p={stats.wilcoxon(nv, dr).pvalue:.2e}'
            print(line)


if __name__ == '__main__':
    main()
