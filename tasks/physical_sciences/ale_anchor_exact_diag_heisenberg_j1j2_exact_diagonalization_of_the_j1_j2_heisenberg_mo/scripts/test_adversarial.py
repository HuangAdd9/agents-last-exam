from __future__ import annotations
import json,shutil,subprocess,tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; score=ROOT/'scripts/score_outputs.py'; ref=ROOT/'reference'
def run(out):
 r=subprocess.run(['python3',str(score),'--output-dir',str(out),'--reference-dir',str(ref)],capture_output=True,text=True)
 return json.loads(r.stdout.strip().splitlines()[-1])['score']
def main():
 with tempfile.TemporaryDirectory() as td:
  t=Path(td); empty=t/'empty'; empty.mkdir(); assert run(empty)==0.0
  bad=t/'wrong_units'; shutil.copytree(ref,bad); (bad/'solver.py').write_text('raise SystemExit(0)');
  import numpy as np
  with np.load(bad/'ground_state.npz') as z: d={k:z[k] for k in z.files}
  d['E0']=d['E0']*1.602176634e-19; np.savez_compressed(bad/'ground_state.npz',**d); assert run(bad)==0.0
  shortcut=t/'visible_copy'; shutil.copytree(ref,shortcut)
  (shortcut/'solver.py').write_text("import shutil,sys; from pathlib import Path; p=Path(sys.argv[sys.argv.index('--output-dir')+1]); r=Path(sys.argv[sys.argv.index('--config')+1]).parent; [shutil.copy2(r/f,p/f) for f in ['ground_state.npz','correlations.npz','dynamical_sf.npz','results.json']]")
  assert run(shortcut)==0.0
 print(json.dumps({'empty':0.0,'wrong_units':0.0,'visible_copy_hidden_probe':0.0}))
if __name__=='__main__': main()
