"""Replay held-out archives, validate split isolation and saved-data integrity."""
from pathlib import Path
import sys,json,csv
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'))
from model import Config,simulate,Policy
from study import policies

def main():
    lock=json.loads((ROOT/'configs/locked.json').read_text());c=Config(**lock['model'])
    split=[set(range(*v)) for v in lock['seeds'].values()]
    for i in range(len(split)):
        for j in range(i):assert not split[i]&split[j]
    menu={p.name:p for p in policies(c)};menu['frozen']=Policy('frozen',mode='frozen')
    for m,s in [(0,1003),(8,1049),(24,1096)]:
        with np.load(ROOT/f'results/raw/evaluation/m{m:02d}_s{s:04d}.npz') as raw:
            for name in ['common_0','staggered','randomized','frozen']:
                got=simulate(c,menu[name],s,memory=m)
                for k,v in got.items():
                    saved=raw[name+'__'+k]
                    if v.dtype.kind=='f':np.testing.assert_allclose(v,saved,rtol=1e-6,atol=2e-6,equal_nan=True)
                    else:np.testing.assert_array_equal(v,saved)
    files=list((ROOT/'results/raw/evaluation').glob('*.npz'));assert len(files)==480
    with (ROOT/'results/feedback.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert all(1049<=int(r['seed'])<=1096 for r in rows)
    assert len(rows)==5*48*4
    for stage,n in [('train',32*5*26),('calibration',48*5*2),('evaluation',96*5*26)]:
        with (ROOT/f'results/{stage}.csv').open() as stream:r=list(csv.DictReader(stream))
        assert len(r)==n,(stage,len(r),n)
        assert all(float(x['completion'])<=1 and float(x['deadline_completion'])<=1 for x in r)
    print('PASS: 12 trajectory replays, every saved array, 480 archives, split isolation and study row counts.')
if __name__=='__main__':main()
