"""Publication figures, generated exclusively from saved evaluation data."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm,ListedColormap
from matplotlib.lines import Line2D
from policy_selection import read
from analyze import ci
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['STIX Two Text','STIXGeneral','DejaVu Serif'],'mathtext.fontset':'stix',
 'font.size':10,'axes.titlesize':11,'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,'axes.edgecolor':'#555555',
 'xtick.color':'#444444','ytick.color':'#444444','text.color':'#20272B','axes.labelcolor':'#20272B',
 'legend.frameon':False,'legend.fontsize':9,'lines.linewidth':1.65,'pdf.fonttype':42,'ps.fonttype':42,
 'savefig.dpi':350,'figure.facecolor':'white','axes.facecolor':'white'})
C={'common_0':'#355C7D','staggered':'#BE6A3C','randomized':'#238579','full':'#777777','public_only':'#AA718F','frozen':'#A0A0A0',
 'nominal':'#238579','feedback':'#8B589C','conservative':'#BE6A3C','known_model':'#5D6368'}
LABEL={'common_0':'Common periodic','staggered':'Group-staggered','randomized':'Common randomized','full':'Full information',
 'public_only':'Public state only','frozen':'Frozen beliefs','nominal':'Nominal policy','feedback':'Feedback adjustment','conservative':'Conservative selection','known_model':'Known-memory selection'}
f=json.loads((ROOT/'results/findings.json').read_text());lock=json.loads((ROOT/'configs/locked.json').read_text());selection=json.loads((ROOT/'configs/selection.json').read_text())
rows=read('evaluation');H=lock['model']['horizon'];deadline=lock['model']['deadline'];seed=f['representative_seed'];target=lock['empirical_target']['herding_mean'];memories=[0,24,16,8,4]
raw=np.load(ROOT/f'results/raw/evaluation/m00_s{seed:04d}.npz')
def tr(p,k):return raw[p+'__'+k]
def title(ax,letter,text):ax.set_title(r'$\mathbf{'+letter+r'}$  '+text,loc='left',pad=10)
def clean(ax,grid=True):
 if grid:ax.grid(axis='y',color='#E6E8E9',lw=.6,zorder=0)
 ax.tick_params(length=3,width=.65)
def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',bbox_inches='tight',facecolor='white',metadata={'Creator':'experiments-1/src/plot.py'} if ext=='pdf' else None)
 plt.close(fig)
def yerr(d):return [[d['mean']-d['lo']],[d['hi']-d['mean']]]
def scatter_frontier(ax,labels=True):
 s=f['summary']['0']; c=f['summary_ci']['0']
 for p,v in s.items():
  if p in ('full','public_only','frozen'):continue
  eligible=v['quality_worst']<=.04 and v['deadline_completion']>=.98
  col=C.get(p,'#A7B0B5');mark='o' if eligible else 'x'
  ax.scatter(v['delta'],v['herding'],s=24 if p in C else 16,color=col,marker=mark,zorder=3,alpha=1 if p in C else .65)
 for p,offset in [('common_0',(8,7)),('staggered',(-65,12)),('randomized',(9,-13))]:
  v=s[p];d=c[p]['herding'];ax.errorbar(v['delta'],v['herding'],yerr=yerr(d),color=C[p],lw=1,capsize=2,zorder=4)
  if labels:ax.annotate(LABEL[p],(v['delta'],v['herding']),xytext=offset,textcoords='offset points',color=C[p],fontsize=8.5)
 ax.axhline(target,color='#646A6E',lw=.8,ls=(0,(4,3)))
 ax.set(xlabel='Departure from a common delivery phase',ylabel='Excess-buildup score $H$',xlim=(-.025,.405));clean(ax)
def mismatch(ax,all_methods=False):
 for method in (['nominal','feedback','conservative','known_model'] if all_methods else ['nominal','feedback','known_model']):
  d=[f['feedback_ci'][str(m)][method]['herding'] for m in memories];y=np.array([x['mean'] for x in d]);lo=np.array([x['lo'] for x in d]);hi=np.array([x['hi'] for x in d]);x=np.arange(5)
  ax.plot(x,y,label=LABEL[method],color=C[method],marker='o' if method=='feedback' else 's' if method=='nominal' else '.',ms=4,ls='--' if method=='known_model' else '-')
  if method in ('nominal','feedback'):ax.fill_between(x,lo,hi,color=C[method],alpha=.10,lw=0)
 ax.axhline(target,color='#646A6E',lw=.8,ls=(0,(4,3)));ax.set(xticks=range(5),xticklabels=['Full\nhistory','24','16','8','4'],xlabel='Memory window (rounds)',ylabel='Excess-buildup score $H$');clean(ax)

# Main: one argument per panel, no decorative three-dimensional axes.
fig=plt.figure(figsize=(8.4,7.2));gs=fig.add_gridspec(2,2,hspace=.53,wspace=.36,height_ratios=[1,1])
sub=gs[0,0].subgridspec(1,3,width_ratios=[1,1,.055],wspace=.13)
lim=.6
for j,p in enumerate(['common_0','staggered']):
 ax=fig.add_subplot(sub[0,j]);a=tr(p,'margin').T
 im=ax.imshow(a,aspect='auto',origin='lower',extent=(-.5,H-.5,.5,24.5),cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-lim,vcenter=0,vmax=lim),interpolation='none',rasterized=True)
 ax.axvline(deadline+.5,color='#353535',ls=':',lw=.8);ax.set(xlabel='Round',xticks=[0,20,40],yticks=[1,12,24])
 if j==0:ax.set_ylabel('Agent');title(ax,'A','Common signal')
 else:ax.set_yticklabels([]);ax.set_title('Staggered signal',loc='left',pad=10)
 for b in [6.5,12.5,18.5]:ax.axhline(b,color='white',lw=.4,alpha=.8)
cb=fig.colorbar(im,cax=fig.add_subplot(sub[0,2]));cb.set_label('$Q(\\mathrm{wait})-Q(\\mathrm{activate})$',labelpad=4);cb.set_ticks([-.6,0,.6])
ax=fig.add_subplot(gs[0,1]);title(ax,'B','Collective response')
for p in ['common_0','staggered','randomized']:ax.plot(np.arange(H),tr(p,'load'),label=LABEL[p],color=C[p],lw=1.5,ls=':' if p=='randomized' else '-')
ax.axvspan(deadline+.5,H-1,color='#F1F3F3',zorder=0);ax.text(35,.99,'Follow-up',ha='center',va='top',fontsize=8,color='#697077');ax.set(xlabel='Round',ylabel='Fraction active',ylim=(-.03,1.05),xlim=(0,H-1));clean(ax);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=3,fontsize=7,columnspacing=.7,handlelength=1.5)
ax=fig.add_subplot(gs[1,0]);title(ax,'C','A common randomized rule can suffice');scatter_frontier(ax)
ax=fig.add_subplot(gs[1,1]);title(ax,'D','Feedback does not ensure recovery');mismatch(ax);ax.legend(loc='upper left',fontsize=8)
fig.subplots_adjust(left=.085,right=.94,top=.94,bottom=.19)
fig.text(.085,.025,f'A–B: representative matched seed {seed}.  C: 96 held-out seeds.  D: 12 independent probe/evaluation batches per memory.\nShading/error bars: 95% bootstrap intervals for means. Dashed horizontal line: empirical target, not a certificate.',fontsize=8,color='#555D62')
save(fig,'01_overview')

# Detailed mechanism: true population action discreteness and full causal chain.
fig,axs=plt.subplots(4,2,figsize=(8.4,9.6),sharex=True,gridspec_kw={'hspace':.25,'wspace':.18})
for j,p in enumerate(['common_0','staggered']):
 ax=axs[0,j];title(ax,'A' if j==0 else 'B',LABEL[p]);bm=tr(p,'belief_mean');ax.plot(bm,color=C[p],alpha=.13,lw=.6);ax.plot(bm.mean(1),color=C[p],lw=1.6);ax.set(ylim=(0,1));clean(ax)
 ax=axs[1,j];ax.imshow(tr(p,'margin').T,aspect='auto',origin='lower',extent=(-.5,H-.5,.5,24.5),cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-.6,vcenter=0,vmax=.6),interpolation='none',rasterized=True);ax.set(yticks=[1,12,24]);
 ax=axs[2,j];ax.imshow(tr(p,'action').T,aspect='auto',origin='lower',extent=(-.5,H-.5,.5,24.5),cmap=ListedColormap(['#F2F3F4',C[p]]),vmin=0,vmax=1,interpolation='none',rasterized=True);ax.set(yticks=[1,12,24]);
 ax=axs[3,j];ax.plot(tr(p,'load'),color=C[p],label='Active fraction');ax.plot(tr(p,'state')[:-1]/4,color='#626970',ls='--',label='Physical pressure');ax.set(xlabel='Round',ylim=(-.03,1.05));clean(ax);None
 for row in range(4):
  axs[row,j].axvline(deadline+.5,color='#444',ls=':',lw=.75)
  if j:axs[row,j].tick_params(labelleft=False)
for ax,label in zip(axs[:,0],['Belief: competing use','Action-value gap\n(agent index)','Binary activation\n(agent index)','Usage / pressure']):ax.set_ylabel(label)
fig.subplots_adjust(left=.13,right=.96,top=.95,bottom=.14)
fig.legend(handles=[Line2D([0],[0],color=C['common_0'],label='Active fraction (column color)'),Line2D([0],[0],color='#626970',ls='--',label='Physical pressure')],loc='lower center',bbox_to_anchor=(.55,.064),ncol=2,fontsize=8)
fig.text(.13,.025,f'Matched seed {seed}; agent ordering is fixed by group. Red gaps favor activation; blue gaps favor waiting.\nWhite heatmap cells: service already complete. Vertical line: common full-information follow-up begins.',fontsize=8,color='#555D62')
save(fig,'02_mechanism')

# Design and quality: all competitors retained, including inconvenient controls.
fig,axs=plt.subplots(1,3,figsize=(12,3.9),gridspec_kw={'wspace':.65})
ax=axs[0];title(ax,'A','All tested delivery rules');scatter_frontier(ax,False)
ax.legend(handles=[Line2D([0],[0],marker='o',color='none',markerfacecolor=C[p],markeredgecolor=C[p],label=LABEL[p]) for p in ['common_0','staggered','randomized']],loc='upper right',fontsize=8)
ax=axs[1];title(ax,'B','Additional changes can hurt')
s=f['summary']['0'];rng=np.random.default_rng(195);names={0:'common_0',**{i:f'mask_{i:02d}' for i in range(1,16)}}
for mask in range(16):
 for e in range(4):
  if mask&(1<<e):continue
  k=mask.bit_count();marg=s[names[mask]]['herding']-s[names[mask|1<<e]]['herding']
  ax.scatter(k+rng.uniform(-.16,.16),marg,s=24,color=['#355C7D','#BE6A3C','#238579','#8B589C'][e],edgecolor='white',lw=.4,zorder=3)
ax.axhline(0,color='#666',lw=.85);ax.set(xlabel='Changes already applied',ylabel='Empirical marginal reduction in $H$',xticks=[0,1,2,3]);clean(ax)
ax=axs[2];title(ax,'C','Low buildup is not enough')
ps=['frozen','public_only','full','common_0','staggered','randomized']
for j,p in enumerate(ps):
 d=f['summary_ci']['0'][p]['quality_worst'];ax.errorbar(d['mean'],j,xerr=yerr(d),fmt='o',ms=4.5,color=C[p],capsize=2)
ax.axvline(.04,color='#555',ls='--',lw=.9);ax.set(yticks=range(6),yticklabels=['Frozen','State only','Full','Common','Staggered','Randomized'],xlabel='Worst-agent reference loss');ax.invert_yaxis();clean(ax,False)
fig.subplots_adjust(left=.065,right=.985,top=.84,bottom=.27)
fig.text(.065,.065,'96 held-out seeds. Panel B shows sample-mean interactions, not a submodularity theorem.\nThe gray line in A and dashed line in C are pilot-informed empirical thresholds; eligibility also requires deadline service.',fontsize=8,color='#555D62')
save(fig,'03_design_and_quality')

# Feedback: distinguish performance from identification and data access.
fb=list(csv.DictReader((ROOT/'results/feedback.csv').open()))
fig,axs=plt.subplots(1,3,figsize=(11.2,3.8),gridspec_kw={'wspace':.34})
ax=axs[0];title(ax,'A','Correction on fresh episodes');mismatch(ax,True);ax.legend(fontsize=7.4,loc='upper left')
ax=axs[1];title(ax,'B','Aggregate feedback is ambiguous')
for k,color,label in [('contains_truth','#238579','True memory retained'),('nominal_rejected','#8B589C','Nominal memory rejected'),('empty_set','#BE6A3C','Empty compatibility set')]:
 vals=[]
 for m in memories:
  z=[r for r in fb if int(r['memory'])==m and r['method']=='feedback']; vals.append(np.mean([r[k]=='True' for r in z]))
 v=np.array(vals);n=12;z=1.96;center=(v+z*z/(2*n))/(1+z*z/n);half=z*np.sqrt(v*(1-v)/n+z*z/(4*n*n))/(1+z*z/n)
 ax.errorbar(np.arange(5)+({'contains_truth':-.09,'nominal_rejected':0,'empty_set':.09}[k]),v,yerr=[v-center+half,center+half-v],fmt='o-',color=color,label=label,ms=3,capsize=2,lw=.9)
ax.set(xticks=range(5),xticklabels=['Full\nhistory','24','16','8','4'],xlabel='Actual memory window (rounds)',ylabel='Fraction of probe batches',ylim=(-.04,1.06));clean(ax);ax.legend(fontsize=7.5,loc='center left')
ax=axs[2];title(ax,'C','Corrections introduce differentiation')
for method in ['nominal','feedback','conservative','known_model']:
 d=[f['feedback_ci'][str(m)][method]['delta'] for m in memories];ax.plot(range(5),[v['mean'] for v in d],marker='o',ms=3.5,color=C[method],ls='--' if method=='known_model' else '-',label=LABEL[method])
ax.set(xticks=range(5),xticklabels=['Full\nhistory','24','16','8','4'],xlabel='Actual memory window (rounds)',ylabel='Departure from a common phase',ylim=(-.015,.31));clean(ax)
fig.subplots_adjust(left=.075,right=.99,top=.84,bottom=.28)
fig.text(.075,.055,'Each decision uses four independent probe episodes and is tested on four fresh episodes (12 batches per memory).\nB: 95% Wilson intervals over 12 decisions. Known-memory selection uses training data and the true label; empty sets use the conservative fallback.',fontsize=8,color='#555D62')
save(fig,'04_mismatch_and_feedback')

# Independent robustness/control seeds, paired comparisons.
di=list(csv.DictReader((ROOT/'results/diagnostics.csv').open()));case_order=['main','exact_symmetry','heterogeneity_15pct','zero_pressure_cost','signals_only','population_48'];case_labels=['Main model','Exact symmetry','15% log-cost heterogeneity','Zero pressure cost','No public-state inference','48 agents']
fig,axs=plt.subplots(1,2,figsize=(9.2,4.6),sharey=True,gridspec_kw={'wspace':.25})
for ax,key,letter,head in [(axs[0],'herding','A','Change in excess buildup'),(axs[1],'peak','B','Change in peak activation')]:
 for j,case in enumerate(case_order):
  rr=[r for r in di if r['case']==case];values={(int(r['seed']),r['policy']):float(r[key]) for r in rr}
  for p,dy in [('staggered',-.12),('randomized',.12)]:
   dif=[values[s,p]-values[s,'common_0'] for s in sorted({int(r['seed']) for r in rr})];d=ci(dif)
   ax.errorbar(d['mean'],j+dy,xerr=yerr(d),fmt='o',color=C[p],ms=4,capsize=2,label=LABEL[p] if j==0 else None)
 ax.axvline(0,color='#777',lw=.8,ls='--');ax.set(xlabel='Policy minus common-periodic baseline',yticks=range(6),yticklabels=case_labels);title(ax,letter,head);clean(ax,False)
axs[0].invert_yaxis();fig.legend(*axs[1].get_legend_handles_labels(),fontsize=8,loc='upper center',bbox_to_anchor=(.65,1.02),ncol=2);fig.subplots_adjust(left=.24,right=.985,top=.86,bottom=.21)
fig.text(.24,.04,'64 independent seeds per condition; paired 95% bootstrap intervals. Negative values favor the tested policy.\nZero pressure cost removes the state-to-cost term, not all feedback. The signals-only ablation deliberately ignores public-state inference.',fontsize=8,color='#555D62')
save(fig,'05_controls_and_robustness')
print('Saved five PDF/SVG/PNG figure sets.')
