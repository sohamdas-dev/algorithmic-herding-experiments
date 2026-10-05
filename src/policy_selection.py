"""Freeze policy-selection rules using TRAINING data only."""
from pathlib import Path
import csv,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def read(stage):
    rows=list(csv.DictReader((ROOT/f'results/{stage}.csv').open()))
    for r in rows:
        for k in r:
            if k not in ('stage','policy'):r[k]=float(r[k])
        r['memory']=int(r['memory']);r['seed']=int(r['seed'])
    return rows

def summarize(rows):
    return {m:{p:{k:float(np.mean([r[k] for r in rows if r['memory']==m and r['policy']==p])) for k in ('herding','delta','quality_worst','deadline_completion','peak','mean_cost')} for p in sorted({r['policy'] for r in rows})} for m in sorted({r['memory'] for r in rows})}

def choose(summary,memories,target):
    allowed=[p for p in summary[0] if p not in ('full','public_only','frozen')]
    good=[p for p in allowed if all(summary[m][p]['quality_worst']<=target['quality_worst_mean'] and summary[m][p]['deadline_completion']>=target['deadline_completion_mean'] for m in memories)]
    feasible=[p for p in good if all(summary[m][p]['herding']<=target['herding_mean'] for m in memories)]
    pool=feasible or good or allowed
    if feasible:
        p=min(pool,key=lambda p:(summary[0][p]['delta'],max(summary[m][p]['herding'] for m in memories),p))
    else:
        p=min(pool,key=lambda p:(max(summary[m][p]['herding'] for m in memories),summary[0][p]['delta'],p))
    return dict(policy=p,training_target_met=bool(feasible),quality_service_eligible=bool(good))

if __name__=='__main__':
 lock=json.loads((ROOT/'configs/locked.json').read_text());s=summarize(read('train'));t=lock['empirical_target']
 selection={'nominal':choose(s,[0],t),'conservative':choose(s,lock['memories'],t),'known_model':{m:choose(s,[m],t) for m in lock['memories']},'training_summary':s}
 common=[p for p in s[0] if p.startswith('common_') or p=='randomized']
 selection['best_common_training']=min(common,key=lambda p:s[0][p]['herding'])
 selection['selection_note']='Training means only; policies outside the quality/service screen are never called eligible. If no H-feasible policy exists, report minimax fallback explicitly. Finite candidate enumeration is not an analytical certificate.'
 (ROOT/'configs/selection.json').write_text(json.dumps(selection,indent=2));print(json.dumps({k:v for k,v in selection.items() if k!='training_summary'},indent=2))
