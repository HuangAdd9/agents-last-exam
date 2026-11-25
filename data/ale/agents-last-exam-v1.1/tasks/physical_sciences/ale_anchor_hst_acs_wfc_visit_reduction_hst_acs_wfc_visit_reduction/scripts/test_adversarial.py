from __future__ import annotations
import shutil,subprocess,tempfile,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
score=ROOT/'scripts/score_outputs.py'; oracle=ROOT/'scripts/oracle.py'; ref=ROOT/'reference/reference_outputs'
def run(inp,out): subprocess.run(['python3',str(oracle),'--input',str(inp),'--output',str(out)],check=True,stdout=subprocess.DEVNULL)
def get(out):
    return float(json.loads(subprocess.check_output(['python3',str(score),'--output-dir',str(out),'--reference-dir',str(ref)]))['score'])
with tempfile.TemporaryDirectory() as td:
    t=Path(td); allin=t/'all'; allin.mkdir()
    shutil.copytree(ROOT/'input/visit_bluefield',allin/'visit_bluefield')
    shutil.copytree(ROOT/'reference/hidden_input/visit_redfield',allin/'visit_redfield')
    perfect=t/'perfect'; run(allin,perfect); assert get(perfect)==1.0
    empty=t/'empty'; empty.mkdir(); assert get(empty)==0.0
    wrong=t/'wrong'; shutil.copytree(perfect,wrong)
    for vid in ('visit_bluefield','visit_redfield'):
        p=wrong/vid/'drizzled_image.csv'
        p.write_text('\n'.join(','.join(str(float(x)*1000) for x in line.split(',')) for line in p.read_text().splitlines()))
        c=wrong/vid/'source_catalog.csv'; lines=c.read_text().splitlines(); c.write_text(lines[0]+'\n'+'\n'.join(','.join((str(float(x)*1000) if j==5 else x) for j,x in enumerate(line.split(','))) for line in lines[1:])+'\n')
    assert get(wrong)<0.8
    shortcut=t/'shortcut'; shortcut.mkdir(); shutil.copytree(perfect/'visit_bluefield',shortcut/'visit_redfield'); shutil.copytree(perfect/'visit_bluefield',shortcut/'visit_bluefield'); assert get(shortcut)<0.8
print('adversarial checks passed: perfect=1.0 empty=0 wrong_units<0.8 visible_copy<0.8')
