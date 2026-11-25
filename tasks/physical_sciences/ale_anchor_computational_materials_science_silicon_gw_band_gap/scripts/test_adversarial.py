#!/usr/bin/env python3
import json, shutil, subprocess, tempfile
from pathlib import Path
D=Path(__file__).resolve().parents[1]; py=D/'software/python'; scorer=D/'scripts/score_outputs.py'; ref=D/'reference'; inp=D/'input'; oracle=D/'scripts/oracle.py'
def score(out):
 p=subprocess.run([str(py),str(scorer),'--input-dir',str(inp),'--output-dir',str(out),'--reference-dir',str(ref)],capture_output=True,text=True,check=True); return json.loads(p.stdout)['score']
with tempfile.TemporaryDirectory() as td:
 t=Path(td); empty=t/'empty'; empty.mkdir(); assert score(empty)==0.0
 malformed=t/'malformed'; malformed.mkdir(); (malformed/'mean_field.json').write_text('{}'); assert score(malformed)==0.0
 perfect=t/'perfect'; subprocess.run([str(py),str(oracle),str(inp/'model_spec.json'),str(perfect)],check=True); shutil.copy(D/'scripts/solve_template.py',perfect/'solve.py'); (perfect/'solve.py').chmod(0o755); assert score(perfect)==1.0
 wrong=t/'wrong'; shutil.copytree(perfect,wrong); q=json.loads((wrong/'quasiparticle.json').read_text()); q['units']='Ha'; (wrong/'quasiparticle.json').write_text(json.dumps(q)); assert score(wrong)==0.0
 visible_copy=t/'visible_copy'; shutil.copytree(perfect,visible_copy); (visible_copy/'solve.py').write_text('# visible-only shortcut\n'); (visible_copy/'solve.py').chmod(0o755); assert score(visible_copy)<1.0
 shortcut=t/'shortcut'; shutil.copytree(perfect,shortcut); s=(shortcut/'solve.py').read_text(); s=s.replace("valley=c['valley']", "valley=[0.0,0.0,0.0]"); (shortcut/'solve.py').write_text(s); assert score(shortcut)<1.0
print('adversarial checks passed: empty=0 malformed=0 wrong_units=0 visible_copy<1 shortcut<1 perfect=1')
