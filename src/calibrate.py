"""Split-conformal model compatibility using aggregate observations only.
Each calibration unit is a batch of four independent reset episodes.
"""
from pathlib import Path
import json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def batches(features,memory,policy,B):
    v=sorted([r for r in features if r['memory']==memory and r['policy']==policy],key=lambda r:r['seed'])
    out=[]
    for j in range(0,len(v)-B+1,B):out.append(np.mean([r['feature'] for r in v[j:j+B]],axis=0))
    return np.array(out)
if __name__=='__main__':
 lock=json.loads((ROOT/'configs/locked.json').read_text());sel=json.loads((ROOT/'configs/selection.json').read_text());p=sel['nominal']['policy'];B=lock['detector']['batch_size'];alpha=lock['detector']['alpha']
 train=json.loads((ROOT/'results/train_features.json').read_text());cal=json.loads((ROOT/'results/calibration_features.json').read_text()); models={}
 for m in lock['memories']:
  tr=batches(train,m,p,B);ca=batches(cal,m,p,B)
  center=tr.mean(0);scale=np.maximum(tr.std(0,ddof=1),.025)
  score=np.sqrt(np.mean(((ca-center)/scale)**2,axis=1))
  k=math.ceil((len(score)+1)*(1-alpha));threshold=float(np.sort(score)[k-1]) if k<=len(score) else float('inf')
  models[m]=dict(center=center.tolist(),scale=scale.tolist(),threshold=threshold,calibration_scores=score.tolist(),rank=k)
 out=dict(probe_policy=p,batch_size=B,alpha=alpha,models=models,
 note='Marginal split-conformal coverage for fixed candidate memory, exchangeable independent episode batches, and the locked probe policy. Not simultaneous adaptive coverage, not an online change detector, and not a herding certificate. Empty sets trigger full-set fallback.')
 (ROOT/'configs/detector.json').write_text(json.dumps(out,indent=2));print('Detector locked',p,{m:round(v['threshold'],3) for m,v in models.items()})
