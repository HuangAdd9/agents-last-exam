from __future__ import annotations
import json,shutil,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; scorer=ROOT/'scripts/score_outputs.py'; ref=ROOT/'reference'
def score(d):
 r=subprocess.run(['/usr/bin/python3',str(scorer),'--output-dir',str(d),'--reference-dir',str(ref)],capture_output=True,text=True,check=True);return json.loads(r.stdout)['score']
# oracle candidate
with tempfile.TemporaryDirectory() as td:
 d=Path(td); subprocess.run(['/usr/bin/python3',str(ROOT/'software/solver.py'),'--config',str(ROOT/'input/model_config.json'),'--output',str(d)],check=True);shutil.copy(ROOT/'software/solver.py',d/'gillespie_solver.py'); assert score(d)>=0.99
# empty/malformed
with tempfile.TemporaryDirectory() as td:
 d=Path(td);(d/'results.json').write_text('{}');assert score(d)==0.0
# visible-value copy with wrong units (all means scaled) must lose numerical components
with tempfile.TemporaryDirectory() as td:
 d=Path(td);shutil.copytree(ROOT/'reference',d,dirs_exist_ok=True); x=json.loads((d/'results.json').read_text());
 for row in x['trajectories']:
  for k in ('mean_A','mean_B','mean_C','std_A','std_B','std_C'):row[k]*=1000
 (d/'results.json').write_text(json.dumps(x));shutil.copy(ROOT/'software/solver.py',d/'gillespie_solver.py'); assert score(d)<0.9
# plausible shortcut: correct visible files but solver always returns visible data, hidden rerun fails
with tempfile.TemporaryDirectory() as td:
 d=Path(td);shutil.copytree(ROOT/'reference',d,dirs_exist_ok=True);(d/'gillespie_solver.py').write_text("import shutil,sys\nfrom pathlib import Path\nshutil.copy(Path(__file__).parents[1]/'reference'/'results.json',Path(sys.argv[sys.argv.index('--output')+1])/'results.json')\n")
 assert score(d)<1.0
print('adversarial checks passed')
