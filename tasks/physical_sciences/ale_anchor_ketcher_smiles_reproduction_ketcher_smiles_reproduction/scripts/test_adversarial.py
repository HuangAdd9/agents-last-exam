#!/usr/bin/env python3
import json, shutil, subprocess, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
VERIFY=ROOT/'scripts/verify_submission.py'; REF=ROOT/'reference/reference.json'; SVG=ROOT/'input/structure.svg'; HIDDEN=ROOT/'reference/hidden_input/structure.svg'
SMI='Cn1cc2cc(c(cc2n1)Cl)NC3=NC(=O)N(C(=O)N3Cc4cc(c(cc4F)F)F)Cc5ncn(n5)C'

def run(case, submission=None, inv=None, solve=None):
 with tempfile.TemporaryDirectory() as td:
  d=Path(td); (d/'output').mkdir()
  if submission is not None:(d/'output/submission.json').write_text(json.dumps(submission))
  if inv is not None:(d/'output/atom_inventory.json').write_text(json.dumps(inv))
  if solve is not None:(d/'output/solve.py').write_text(solve); (d/'output/solve.py').chmod(0o755)
  p=subprocess.run(['python3',str(VERIFY),'--output',str(d/'output'),'--reference',str(REF),'--svg',str(SVG),'--hidden',str(HIDDEN)],capture_output=True,text=True)
  got=json.loads(p.stdout); print(case,got['score'],got['reason']); return got['score']
inv=json.loads((ROOT/'reference/visible_inventory.json').read_text())
sub={'smiles':SMI,'formula':'C22H17ClF3N9O2','heavy_atom_count':37,'ring_count':5,'atom_counts':inv['atom_counts']}
solve='''import json,re,sys\ns=open(sys.argv[1]).read(); print(json.dumps({"atom_counts":{"C":22,"Cl":1,"F":3,"N":9,"O":2},"heavy_atom_count":37}))\n'''
generic='''import json,re,sys
s=open(sys.argv[1]).read(); c={"C":22}
for e in re.findall(r'<text[^>]*>\\s*([A-Z][a-z]?)\\s*</text>',s):
    if e != "H": c[e]=c.get(e,0)+1
print(json.dumps({"atom_counts":c,"heavy_atom_count":sum(c.values())}))
'''
assert run('empty')==0.0
assert run('perfect',sub,inv,generic)==1.0
bad=dict(sub); bad['formula']='C22H17ClF3N9O2kg'; assert run('wrong_units',bad,inv,generic)==0.0
assert run('hardcoded_hidden_shortcut',sub,inv,solve)==0.8
print('adversarial checks passed')
