"""Compact proposal figure: saved binary actions and matched aggregate response.
Run with experiments-1/.venv/bin/python; no simulation or seed selection occurs here.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
R=Path(__file__).resolve().parents[1];out=R/'figures'
out.mkdir(parents=True,exist_ok=True)
f=json.loads((R/'results/findings.json').read_text());seed=f['representative_seed']
z=np.load(R/f'results/raw/evaluation/m00_s{seed:04d}.npz')
plt.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral'],'mathtext.fontset':'stix','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'savefig.dpi':400})
colors={'common_0':'#355C7D','staggered':'#BE6A3C','randomized':'#238579'}
fig=plt.figure(figsize=(6.5,2.55));g=fig.add_gridspec(2,2,width_ratios=[.95,1.15],wspace=.34,hspace=.34)
for j,(p,label) in enumerate([('common_0','Common'),('staggered','Staggered')]):
 ax=fig.add_subplot(g[j,0]);ax.imshow(z[p+'__action'].T,aspect='auto',origin='lower',extent=(-.5,41.5,.5,24.5),cmap=ListedColormap(['#F1F3F4',colors[p]]),vmin=0,vmax=1,interpolation='none',rasterized=True)
 ax.set(yticks=[1,24],xticks=[0,10,20,30,40],ylabel=label+'\nagent');ax.axvline(30.5,color='#777',ls=':',lw=.7)
 if j==0:ax.set_title(r'$\mathbf{A}$  Individual activation',loc='left',pad=7);ax.tick_params(labelbottom=False)
 else:ax.set_xlabel('Round',labelpad=2)
 ax.tick_params(length=2,pad=2)
ax=fig.add_subplot(g[:,1]);ax.set_title(r'$\mathbf{B}$  Collective response',loc='left',pad=7)
for p,label in [('common_0','Common periodic'),('staggered','Group-staggered'),('randomized','Common randomized')]:
 ax.plot(z[p+'__load'],color=colors[p],label=label,lw=1.35,ls=':' if p=='randomized' else '-')
ax.axvspan(30.5,41,color='#F1F3F4',zorder=0);ax.text(35.7,.93,'Follow-up',ha='center',fontsize=7,color='#555')
ax.set(xlim=(0,41),ylim=(-.025,1.035),xticks=[0,10,20,30,40],yticks=[0,.5,1],xlabel='Round',ylabel='Fraction active')
ax.grid(axis='y',color='#E5E7E9',lw=.5);ax.tick_params(length=2,pad=2)
fig.subplots_adjust(left=.085,right=.985,top=.87,bottom=.25)
fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.55,.015),ncol=3,frameon=False,fontsize=8,handlelength=2,columnspacing=1.4)
for ext in ['pdf','svg','png']:fig.savefig(out/f't1-discrete-evidence-5N.{ext}',bbox_inches='tight',pad_inches=.03)
print('Saved proposal figure for representative seed',seed)
