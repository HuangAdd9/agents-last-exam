#!/usr/bin/env python3
"""Independent local gates: perfect, empty, wrong-units, copy-visible, hidden-shortcut."""
from __future__ import annotations
import csv,json,shutil,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
TRUE='k_mix = 0.235\nlight_extinction = 0.165\nair_exchange = 0.041\nsediment_flux = 0.014\ninitial_temp = 5.65\n'
def score(p):
 r=subprocess.run(['/usr/bin/python3',str(p/'scripts/score_outputs.py'),'--input-dir',str(p/'input'),'--output-dir',str(p/'output'),'--reference-dir',str(p/'reference')],capture_output=True,text=True,check=True)
 return json.loads(r.stdout.strip().splitlines()[-1])
def run_model(p, cfg=None):
 if cfg is not None: (p/'input/model.nml').write_text(cfg)
 shutil.rmtree(p/'output',ignore_errors=True)
 subprocess.run([str(p/'software/run_model.sh')],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
def main():
 with tempfile.TemporaryDirectory(prefix='lake_adv_') as td:
  p=Path(td)/ROOT.name; shutil.copytree(ROOT,p,ignore=shutil.ignore_patterns('output','__pycache__'))
  run_model(p,TRUE); perfect=score(p)['score']
  shutil.rmtree(p/'output'); empty=score(p)['score']
  run_model(p,TRUE)
  rows=list(csv.DictReader((p/'output/simulation.csv').open()));
  for r in rows: r['temp_c']=f"{float(r['temp_c'])*9/5+32:.9f}"
  with (p/'output/simulation.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['day','depth_m','temp_c']); w.writeheader(); w.writerows(rows)
  wrong_units=score(p)['score']
  run_model(p,TRUE); shutil.copyfile(p/'input/observations_visible.csv',p/'output/simulation.csv'); copied=score(p)['score']
  run_model(p,'k_mix = 0.235\nlight_extinction = 0.050\nair_exchange = 0.041\nsediment_flux = 0.014\ninitial_temp = 5.65\n'); shortcut=score(p)['score']
  result={'oracle':perfect,'empty':empty,'wrong_units':wrong_units,'copy_visible':copied,'hidden_shortcut':shortcut}
  print(json.dumps(result,sort_keys=True))
  assert perfect==1.0 and empty==0.0 and wrong_units<1.0 and copied<1.0 and shortcut<1.0
if __name__=='__main__': main()
