from model import *
from dataclasses import replace,asdict
from pathlib import Path
import json,time
root=Path(__file__).resolve().parents[1]; rows=[]
for j,(g,h,z,p,per) in enumerate([(g,h,z,p,per) for g in [2.,3.] for h in [.01,.025] for z in [.025,.05] for p in [.4,.6] for per in [5,6]]):
 c=replace(Config(),congestion=g,holding_cost=h,temperature=z,prior_mean=p,period=per)
 pols=[Policy('common'),Policy('staggered',(0,1,3,4)),Policy('randomized',mode='randomized'),Policy('full',mode='full'),Policy('public_only',mode='public_only')]
 rec=[]
 for seed in range(101,113):
  ref=simulate(c,Policy('frozen',mode='frozen'),seed,quality=False)
  for pol in pols:
   rec.append(dict(seed=seed,policy=pol.name,**metrics(c,simulate(c,pol,seed),ref)))
 summary={p.name:{k:float(np.mean([v[k] for v in rec if v['policy']==p.name])) for k in ['herding','peak','quality_worst','mean_cost','deadline_completion']} for p in pols}
 rows.append(dict(config=asdict(c),summary=summary,runs=rec))
 (root/'results/pilot.json').write_text(json.dumps(rows,indent=2))
 print(j, {p:{k:round(v,3) for k,v in d.items()} for p,d in summary.items()},flush=True)
