"""Run the locked study with this interpreter, from any working directory."""
from pathlib import Path
import argparse,os,subprocess,sys
ROOT=Path(__file__).resolve().parent

def run(*args):
    print('+',*args,flush=True)
    subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=os.environ,check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['figures','verify','full'],default='verify');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    os.environ.update(OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1')
    if a.mode=='full':
        run('src/study.py','train','--workers',a.workers)
        run('src/policy_selection.py')
        run('src/study.py','calibration','--workers',a.workers)
        run('src/calibrate.py')
        run('src/study.py','evaluation','--workers',a.workers)
        run('src/diagnostics.py')
    if a.mode in ('figures','full'):
        run('src/analyze.py');run('src/plot.py');run('src/report.py')
        run('src/proposal_figure_mechanism.py')
    run('-m','unittest','discover','-s','tests','-v');run('verify.py')
    if a.mode=='full':run('src/manifest.py')
