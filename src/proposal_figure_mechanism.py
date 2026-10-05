"""Proposal-sized overview A+B, using saved evaluation data.
Run: experiments-1/.venv/bin/python experiments-1/src/proposal_figure_mechanism.py
No simulations, policy changes, or seed selection occur here.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, default=ROOT / 'figures')
args = parser.parse_args()
OUT = args.output_dir
OUT.mkdir(parents=True, exist_ok=True)
findings = json.loads((ROOT / 'results/findings.json').read_text())
model = json.loads((ROOT / 'configs/locked.json').read_text())['model']
seed = findings['representative_seed']
data = np.load(ROOT / f'results/raw/evaluation/m00_s{seed:04d}.npz')
horizon = model['horizon']
deadline = model['deadline']
n_agents = data['common_0__action'].shape[1]
colors = {'common_0': '#355C7D', 'staggered': '#BE6A3C', 'randomized': '#238579'}
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix',
    'font.size': 9, 'axes.titlesize': 8, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .6, 'pdf.fonttype': 42, 'savefig.dpi': 450,
})
fig = plt.figure(figsize=(6.5, 2.55))
cmap = plt.colormaps['RdBu_r'].copy()
# Unavailable gaps mean service is complete, distinct from a zero gap (white).
cmap.set_bad('#F0F1F2')
for j, (policy, label) in enumerate([('common_0', 'Common signal'), ('staggered', 'Staggered signal')]):
    ax = fig.add_axes([.055 + j * .164, .23, .143, .65])
    margin = data[f'{policy}__margin'].T
    # In the simulator margin = cost(wait) - cost(activate).
    assert np.array_equal(np.isnan(margin.T), data[f'{policy}__backlog'][:-1] == 0)
    im = ax.imshow(np.ma.masked_invalid(margin), aspect='auto', origin='lower',
                   extent=(-.5, horizon-.5, .5, n_agents+.5), cmap=cmap,
                   norm=TwoSlopeNorm(vmin=-.6, vcenter=0, vmax=.6),
                   interpolation='none', rasterized=True)
    for boundary in np.flatnonzero(np.diff(data[f'{policy}__group'])) + 1:
        ax.axhline(boundary + .5, color='white', linewidth=.4, alpha=.85)
    ax.axvline(deadline+.5, color='#444444', linestyle=':', linewidth=.7)
    ax.set(xlabel='Round', xticks=[0, 20, 40], yticks=[1, 12, n_agents])
    ax.set_title((r'$\mathbf{A}$  ' if j == 0 else '') + label, loc='left', fontsize=9, pad=7)
    ax.xaxis.labelpad = 2
    if j == 0:
        ax.set_ylabel('Agent', labelpad=3)
    else:
        ax.tick_params(labelleft=False)
    ax.tick_params(length=2, pad=2)
cax = fig.add_axes([.389, .23, .012, .65])
cb = fig.colorbar(im, cax=cax, orientation='vertical')
cb.set_ticks([-.6, 0, .6])
cb.set_label('Preference for activation', fontsize=8, labelpad=4)
cb.ax.tick_params(length=2, pad=2, labelsize=7)

ax = fig.add_axes([.565, .23, .425, .65])
ax.set_title(r'$\mathbf{B}$  Collective response', loc='left', fontsize=9, pad=7)
for policy, label in [('common_0', 'Common periodic'), ('staggered', 'Group-staggered'), ('randomized', 'Common randomized')]:
    ax.plot(np.arange(horizon), data[f'{policy}__load'], color=colors[policy],
            label=label, linewidth=1.25, linestyle=':' if policy == 'randomized' else '-')
ax.axvline(deadline+.5, color='#666666', linestyle=':', linewidth=.7)
ax.text((deadline+horizon)/2, .97, 'Follow-up', ha='center', va='top', fontsize=7, color='#555555')
ax.set(xlim=(0, horizon-1), ylim=(-.025, 1.035), xticks=[0, 10, 20, 30, 40],
       yticks=np.arange(0, 1.01, .2), xlabel='Round', ylabel='Fraction active')
ax.xaxis.labelpad = 2
ax.yaxis.labelpad = 3
ax.grid(axis='y', color='#E5E7E9', linewidth=.5)
ax.tick_params(length=2, pad=2)
fig.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(.778, .006),
           ncol=3, frameon=False, fontsize=6.8, handlelength=1.5, columnspacing=.8)
for ext in ['pdf', 'svg', 'png']:
    fig.savefig(OUT / f't1-mechanism-evidence.{ext}', bbox_inches='tight', pad_inches=.035)
plt.close(fig)
print(f'Saved mechanism proposal figure from unchanged matched seed {seed}.')
