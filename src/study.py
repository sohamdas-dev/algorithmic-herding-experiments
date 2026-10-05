"""Locked train/calibrate/evaluate study. Run from any working directory."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from model import *
from dataclasses import asdict,replace
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,csv,time,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1]

def policies(c):
    out=[Policy(f'common_{b}',(b,)*c.groups) for b in range(c.period)]
    offsets=(1,2,3,4)
    for mask in range(1,16):
        out.append(Policy(f'mask_{mask:02d}',tuple(offsets[g] if mask&(1<<g) else 0 for g in range(c.groups))))
    out += [Policy('staggered',(0,1,3,4)),Policy('randomized',mode='randomized'),Policy('full',mode='full'),Policy('public_only',mode='public_only')]
    return out

def differentiation(p,c):
    if p.mode!='periodic': return 0.
    # Minimum weighted cyclic phase displacement from any common phase.
    return min(sum(min((v-b)%c.period,(b-v)%c.period) for v in p.phases)/(c.groups*c.period) for b in range(c.period))

def feature(tr,c):
    # Only aggregate resource usage and public physical states, not internal beliefs.
    load=tr['load'];state=tr['state'][:-1]/(c.state_levels-1)
    block=5
    return np.r_[[load[j:j+block].mean() for j in range(0,c.deadline,block)],
                 [state[j:j+block].mean() for j in range(0,c.deadline,block)],
                 np.abs(np.diff(load[:c.deadline])).mean(),load[:c.deadline].max()]

def batch(job):
    stage,seed,memory,save=job
    lock=json.loads((ROOT/'configs/locked.json').read_text());c=Config(**lock['model'])
    ref=simulate(c,Policy('frozen',mode='frozen'),seed,memory=memory)
    rows=[dict(stage=stage,seed=seed,memory=memory,policy='frozen',delta=0.,**metrics(c,ref,ref))]
    raw={}; feats=[]
    menu=policies(c)
    if stage=='calibration':
        selected=json.loads((ROOT/'configs/selection.json').read_text())['nominal']['policy']
        menu=[p for p in menu if p.name==selected]
    for p in menu:
        tr=simulate(c,p,seed,memory=memory)
        rows.append(dict(stage=stage,seed=seed,memory=memory,policy=p.name,delta=differentiation(p,c),**metrics(c,tr,ref)))
        feats.append(dict(seed=seed,memory=memory,policy=p.name,feature=feature(tr,c).tolist()))
        if save:
            for k,v in tr.items(): raw[p.name+'__'+k]=v.astype(np.float32) if v.dtype.kind=='f' else v
    if save:
        for k,v in ref.items():raw['frozen__'+k]=v.astype(np.float32) if v.dtype.kind=='f' else v
        folder=ROOT/'results/raw'/stage;folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/f'm{memory:02d}_s{seed:04d}.npz',**raw)
    return rows,feats

def run(stage,workers=4):
    lock=json.loads((ROOT/'configs/locked.json').read_text())
    seeds=range(*lock['seeds'][stage]);jobs=[(stage,s,m,stage=='evaluation') for m in lock['memories'] for s in seeds]
    rows=[]; feats=[];start=time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs=[pool.submit(batch,j) for j in jobs]
        for j,f in enumerate(as_completed(futs),1):
            r,v=f.result();rows.extend(r);feats.extend(v)
            if j%20==0 or j==len(jobs): print(stage,j,len(jobs),'elapsed',round(time.time()-start,1),flush=True)
    rows.sort(key=lambda x:(x['memory'],x['seed'],x['policy']));feats.sort(key=lambda x:(x['memory'],x['seed'],x['policy']))
    with (ROOT/f'results/{stage}.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (ROOT/f'results/{stage}_features.json').write_text(json.dumps(feats))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['train','calibration','evaluation']);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();run(a.stage,a.workers)
