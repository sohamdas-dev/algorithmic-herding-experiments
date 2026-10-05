import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
from model import *
from study import differentiation
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[1]
def run(seed):
 c=Config(**json.loads((ROOT/'configs/locked.json').read_text())['model']);out=[]
 cases={'main':(c,False),'exact_symmetry':(replace(c,heterogeneity=0),False),'heterogeneity_15pct':(replace(c,heterogeneity=.15),False),'zero_pressure_cost':(replace(c,pressure_cost=0),False),'signals_only':(c,True),'population_48':(replace(c,n=48),False)}
 for case,(cc,ignore) in cases.items():
  f=simulate(cc,Policy('frozen',mode='frozen'),seed,quality=False)
  for p in [Policy('common_0'),Policy('staggered',(0,1,3,4)),Policy('randomized',mode='randomized')]:
   tr=simulate(cc,p,seed,ignore_public=ignore)
   out.append(dict(case=case,seed=seed,policy=p.name,**metrics(cc,tr,f)))
 return out
if __name__=='__main__':
 rows=[]
 with ProcessPoolExecutor(max_workers=2) as pool:
  for j,f in enumerate(as_completed([pool.submit(run,s) for s in range(2001,2065)]),1):
   rows.extend(f.result())
   if j%16==0:print(j,flush=True)
 rows.sort(key=lambda r:(r['case'],r['seed'],r['policy']))
 with (ROOT/'results/diagnostics.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
