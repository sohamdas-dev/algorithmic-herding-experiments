from pathlib import Path
import csv,json,itertools
import numpy as np
from policy_selection import read,summarize,choose
from calibrate import batches
ROOT=Path(__file__).resolve().parents[1]

def ci(values,seed=73):
 v=np.array(values);rng=np.random.default_rng(seed)
 boot=v[rng.integers(len(v),size=(4000,len(v)))].mean(1)
 return dict(mean=float(v.mean()),lo=float(np.quantile(boot,.025)),hi=float(np.quantile(boot,.975)),n=len(v))

def main():
 lock=json.loads((ROOT/'configs/locked.json').read_text());sel=json.loads((ROOT/'configs/selection.json').read_text());det=json.loads((ROOT/'configs/detector.json').read_text())
 rows=read('evaluation');summary=summarize(rows);train={int(k):v for k,v in sel['training_summary'].items()};features=json.loads((ROOT/'results/evaluation_features.json').read_text())
 lookup={(r['memory'],r['seed'],r['policy']):r for r in rows}
 nominal=sel['nominal']['policy'];conservative=sel['conservative']['policy'];B=det['batch_size'];modes=lock['memories'];target=lock['empirical_target'];out=[]
 # Probe seeds and performance seeds are disjoint. One selected policy is evaluated
 # on four fresh seeds per model-set decision; inference uses aggregate features only.
 seeds=list(range(*lock['seeds']['evaluation']));cut=len(seeds)//2;probe=seeds[:cut];test=seeds[cut:]
 for m in modes:
  fs=[r for r in features if r['seed'] in probe]
  data=batches(fs,m,nominal,B)
  for j,z in enumerate(data):
   compatible=[];scores={}
   for candidate in modes:
    d=det['models'][str(candidate)];score=float(np.sqrt(np.mean(((z-d['center'])/d['scale'])**2)));scores[candidate]=score
    if score<=d['threshold']:compatible.append(candidate)
   empty=not compatible;use=compatible or modes
   correction=choose(train,use,target)
   for seed in test[j*B:(j+1)*B]:
    for method,p in [('nominal',nominal),('feedback',correction['policy']),('conservative',conservative),('known_model',sel['known_model'][str(m)]['policy'])]:
     row=dict(lookup[m,seed,p]);row.update(method=method,batch=j,compatible=';'.join(map(str,compatible)),empty_set=empty,contains_truth=m in compatible,nominal_rejected=0 not in compatible,selected_training_feasible=correction['training_target_met'] if method=='feedback' else sel[method if method!='known_model' else 'known_model'].get('training_target_met',False) if method!='known_model' else sel['known_model'][str(m)]['training_target_met'])
     out.append(row)
 with (ROOT/'results/feedback.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 # Bootstrap paired seed differences in nominal stage; batch bootstrap for feedback.
 summary_ci={}
 for m in modes:
  summary_ci[m]={}
  for p in summary[m]:
   rr=[r for r in rows if r['memory']==m and r['policy']==p]
   summary_ci[m][p]={k:ci([r[k] for r in rr]) for k in ('herding','peak','quality_worst','mean_cost','deadline_completion','tail_peak')}
 comparisons={}
 for p in ['staggered',nominal,'full','public_only']:
  rr=[lookup[0,s,p]['herding']-lookup[0,s,'common_0']['herding'] for s in seeds]
  comparisons[p]=ci(rr)
 feedback_ci={}
 for m in modes:
  feedback_ci[m]={}
  for method in ('nominal','feedback','conservative','known_model'):
   rr=[r for r in out if r['memory']==m and r['method']==method]
   feedback_ci[m][method]={k:ci([np.mean([r[k] for r in rr if r['batch']==j]) for j in sorted({r['batch'] for r in rr})]) for k in ('herding','peak','quality_worst','delta','deadline_completion')}
 # Representative nominal seed minimizes distance to median paired H change and levels.
 pairs=np.array([[lookup[0,s,'common_0']['herding'],lookup[0,s,'staggered']['herding']] for s in seeds]);distance=np.sum((pairs-np.median(pairs,axis=0))**2,axis=1)
 # Smallest seed among numerical ties, stable under equivalent floating-point arithmetic.
 representative=seeds[int(np.flatnonzero(np.isclose(distance,distance.min(),rtol=0,atol=1e-12))[0])]
 # Empirical diminishing returns violations, recorded for ALL mask pairs.
 names={0:'common_0',**{m:f'mask_{m:02d}' for m in range(1,16)}}
 benefit={m:summary[0]['common_0']['herding']-summary[0][p]['herding'] for m,p in names.items()}
 violations=[]
 for A in range(16):
  for BB in range(16):
   if A&BB!=A:continue
   for e in range(4):
    if BB&(1<<e):continue
    da=benefit[A|1<<e]-benefit[A];db=benefit[BB|1<<e]-benefit[BB]
    violations.append(dict(A=A,B=BB,element=e,marginal_A=da,marginal_B=db,violation=db-da))
 with (ROOT/'results/marginal_interactions.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(violations[0]));w.writeheader();w.writerows(violations)
 findings=dict(summary=summary,summary_ci=summary_ci,paired_H_difference_vs_common0=comparisons,feedback_ci=feedback_ci,representative_seed=representative,
  feedback_observation=dict(probe_episodes_per_decision=B,performance_episodes_per_decision=B,batches_per_memory=len(test)//B,performance_seeds=test),
  empirical_submodularity=dict(max_violation=max(v['violation'] for v in violations),comparisons=len(violations),positive_violations=sum(v['violation']>1e-10 for v in violations)),
  limitations=['Synthetic uncalibrated Markov resource model','No alpha-potential, FP convergence, or analytical herding certificate established','Empirical H includes reference-wave phase shifts and can exceed one','Frozen reference suppresses aggregate-belief adaptation, not all physical feedback','No operating-field validation','Static finite candidate mismatch set; detector coverage is marginal, not simultaneous','Empirical mean eligibility is not per-agent/per-episode certification','Common randomized rule may suffice without differentiation'])
 (ROOT/'results/findings.json').write_text(json.dumps(findings,indent=2));print(json.dumps({k:v for k,v in findings.items() if k not in ('summary','summary_ci','feedback_ci')},indent=2))
if __name__=='__main__':main()
