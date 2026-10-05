from pathlib import Path
import hashlib,json,platform,sys,importlib.metadata
R=Path(__file__).resolve().parents[1]
out={'python':sys.version,'platform':platform.platform(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','matplotlib','pillow']},'files':{}}
for p in sorted(R.rglob('*')):
    if not p.is_file() or any(x in p.parts for x in ['.venv','__pycache__','.git','.mplconfig']) or p.name=='manifest.json':continue
    out['files'][str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'manifest.json').write_text(json.dumps(out,indent=2)+'\n')
print('Manifest:',len(out['files']),'files')
