"""
make_figures.py — generate the three paper figures for DiffCal from the result JSONs.

Outputs (PDF, IEEE-column sized) into ../figures/ (self-contained in this repo):
  fig_framework.pdf   — schematic of the diffusion-FPT pipeline + direct-RUL branch
  fig_reliability.pdf — PIT reliability curves (naive-FPT vs direct-RUL vs ideal)
  fig_ablation.pdf    — threshold-sweep: Cov@0.9 and PIT-KS vs failure threshold

Run:  python make_figures.py
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, '..', 'results')
FIGS = os.path.join(HERE, '..', 'figures')   # self-contained: write into the repo
os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.titlesize': 8,
    'font.family': 'serif', 'pdf.fonttype': 42, 'ps.fonttype': 42,
})
C_NAIVE, C_DIRECT, C_IDEAL = '#c0392b', '#2471a3', '#7f8c8d'


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


# ---------------------------------------------------------------- Fig 1: framework
def framework():
    fig, ax = plt.subplots(figsize=(7.0, 2.35))
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis('off')

    def box(x, y, w, h, text, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.3,rounding_size=1.2',
                                    fc=fc, ec='#333333', lw=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7.2)

    def arrow(x0, y0, x1, y1, style='-'):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>',
                                     mutation_scale=9, lw=0.8, color='#333333',
                                     linestyle=style))

    y0 = 19
    box(1,  y0, 15, 9, 'Raw vibration\n(XJTU / FEMTO)', '#eaf2f8')
    box(19, y0, 16, 9, 'Health indicator\nRMS $\\to$ log $\\to$ monotone', '#eaf2f8')
    box(38, y0, 17, 9, 'Conditional diffusion\nlookback $\\to$ forecast', '#e8f8f5')
    box(58, y0, 18, 9, 'Autoregressive rollout\n$K$ trajectories', '#e8f8f5')
    box(79, y0, 19, 9, 'First-passage of $D$\n$\\Rightarrow$ RUL distribution', '#fdf2e9')
    for x in (16, 35, 55, 76):
        arrow(x, y0 + 4.5, x + 3, y0 + 4.5)

    # calibration output
    box(79, 4, 19, 9, 'Calibration audit\nPIT / coverage / CRPS', '#f9ebea')
    arrow(88.5, y0, 88.5, 13)

    # direct-RUL branch (bypasses rollout + FPT)
    box(46, 4, 24, 9, 'Direct-RUL diffusion (control)\nwindow $\\to$ RUL, no rollout/FPT', '#f4ecf7')
    arrow(46, y0, 58, 8.5, style='--')          # from HI/diffusion region down to direct
    arrow(70, 8.5, 79, 8.5, style='--')         # direct -> calibration
    ax.text(50, 1.2, 'dashed: rollout-free path', fontsize=6.3, color='#555555')

    fig.tight_layout(pad=0.2)
    fig.savefig(os.path.join(FIGS, 'fig_framework.pdf'), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- Fig 2: reliability
def reliability():
    v = load('diagnostic_combined.json')['verdict']
    lv = np.array(v['rel_levels'])
    naive = np.array(v['naive_reliability'])
    direct = np.array(v['directrul_reliability'])
    fig, ax = plt.subplots(figsize=(3.3, 2.7))
    ax.plot([0, 1], [0, 1], '--', color=C_IDEAL, lw=1.0, label='ideal (calibrated)')
    ax.plot(lv, naive, '-o', color=C_NAIVE, ms=3, lw=1.2, label='naive FPT')
    ax.plot(lv, direct, '-s', color=C_DIRECT, ms=3, lw=1.2, label='direct-RUL')
    ax.set_xlabel('nominal level $q$')
    ax.set_ylabel(r'empirical $\Pr(\mathrm{PIT}\leq q)$')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title('PIT reliability (matched threshold)')
    ax.legend(loc='upper left', frameon=False)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIGS, 'fig_reliability.pdf'), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- Fig 3: threshold sweep
def ablation():
    d = load('ablation_dsweep_combined.json')
    ps = d['p_grid']; res = d['results']
    n90 = [res[str(p)]['naive']['cov90'] for p in ps]
    d90 = [res[str(p)]['direct']['cov90'] for p in ps]
    nks = [res[str(p)]['naive']['pit_ks'] for p in ps]
    dks = [res[str(p)]['direct']['pit_ks'] for p in ps]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    a1.axhline(0.9, ls='--', color=C_IDEAL, lw=1.0, label='nominal 0.90')
    a1.plot(ps, n90, '-o', color=C_NAIVE, ms=4, lw=1.3, label='naive FPT')
    a1.plot(ps, d90, '-s', color=C_DIRECT, ms=4, lw=1.3, label='direct-RUL')
    a1.set_xlabel('failure threshold percentile $p$')
    a1.set_ylabel('Cov@0.9')
    a1.set_ylim(0, 1.02); a1.set_title('(a) coverage vs.\\ threshold')
    a1.legend(loc='lower left', frameon=False); a1.grid(alpha=0.25, lw=0.4)

    a2.plot(ps, nks, '-o', color=C_NAIVE, ms=4, lw=1.3, label='naive FPT')
    a2.plot(ps, dks, '-s', color=C_DIRECT, ms=4, lw=1.3, label='direct-RUL')
    a2.set_xlabel('failure threshold percentile $p$')
    a2.set_ylabel('PIT-KS (lower = better)')
    a2.set_ylim(0, 0.85); a2.set_title('(b) miscalibration vs.\\ threshold')
    a2.legend(loc='upper left', frameon=False); a2.grid(alpha=0.25, lw=0.4)

    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(FIGS, 'fig_ablation.pdf'), bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- Fig 4: coverage benchmark
def benchmark():
    """Grouped Cov@0.9 bars for every arm under matched vs fixed-D thresholds, with the
       nominal [0.85,0.95] band shaded. Visualizes 'no method robustly reaches nominal'
       (table-only otherwise) and the generator-agnostic MC-dropout result."""
    arms = {'Naive\nFPT': 'naive', 'MC-drop\nFPT': 'mcdrop', r'Global-$\gamma$': 'corrected',
            'Stage-\nconf.': 'conformal', 'UPR': 'upr', 'Direct-\nRUL': 'directrul'}

    def stats(fname):
        runs = load(fname)['runs']
        return {lab: (np.mean([r[k]['cov90'] for r in runs]),
                      np.std([r[k]['cov90'] for r in runs])) for lab, k in arms.items()}

    s = stats('diagnostic_combined.json'); f = stats('diagnostic_combined_fixedD.json')
    labels = list(arms.keys()); x = np.arange(len(labels)); w = 0.38
    C_STD, C_FIX = '#2471a3', '#e67e22'

    fig, ax = plt.subplots(figsize=(3.5, 2.75))
    ax.axhspan(0.85, 0.95, color='#27ae60', alpha=0.15, zorder=0, label='nominal band [.85,.95]')
    ax.axhline(0.90, ls='--', color=C_IDEAL, lw=1.0, zorder=1)
    ax.bar(x - w/2, [s[l][0] for l in labels], w, yerr=[s[l][1] for l in labels], capsize=2,
           color=C_STD, alpha=0.9, label='matched $D$', error_kw={'lw': 0.7})
    ax.bar(x + w/2, [f[l][0] for l in labels], w, yerr=[f[l][1] for l in labels], capsize=2,
           color=C_FIX, alpha=0.9, label='fixed $D$', error_kw={'lw': 0.7})
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.4)
    ax.set_ylabel('Cov@0.9'); ax.set_ylim(0, 1.06)
    ax.set_title('$90\\%$ interval coverage by method')
    ax.legend(loc='upper center', frameon=False, fontsize=6.2, ncol=1,
              handlelength=1.3, labelspacing=0.25)
    ax.grid(axis='y', alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(FIGS, 'fig_benchmark.pdf'), bbox_inches='tight')
    fig.savefig(os.path.join(FIGS, 'fig_benchmark_preview.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    # benchmark() available but unused — Table II is kept as a table (bar chart was redundant)
    framework(); reliability(); ablation()
    print('figures written to', os.path.abspath(FIGS))
    for f in sorted(os.listdir(FIGS)):
        print('  ', f)
