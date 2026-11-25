from pathlib import Path
import json, shutil, tempfile
from oracle import analyze_dir
from score import score

ROOT=Path(__file__).resolve().parents[1]
def write_candidate(out, payload, analyzer=True):
    out.mkdir(parents=True,exist_ok=True)
    (out/'decisions.json').write_text(json.dumps(payload),encoding='utf-8')
    s=dict(payload['summary']); s.update({'units':'angstrom','algorithm':'covalent_graph_v1'})
    (out/'summary.json').write_text(json.dumps(s),encoding='utf-8')
    if analyzer: shutil.copy(ROOT/'scripts'/'oracle.py',out/'analyzer.py')

with tempfile.TemporaryDirectory() as td:
    base=Path(td); perfect=analyze_dir(ROOT/'input'/'xyz_files'); write_candidate(base/'perfect',perfect)
    p=score(base/'perfect',ROOT/'reference',ROOT/'input'/'xyz_files'); assert abs(p['score']-1.0)<1e-12, p
    empty=base/'empty'; empty.mkdir(); assert score(empty,ROOT/'reference',ROOT/'input'/'xyz_files')['score']==0.0
    wrong=json.loads(json.dumps(perfect)); wrong['files']['water_good.xyz']['min_distance_A']*=10; write_candidate(base/'wrong_units',wrong)
    w=score(base/'wrong_units',ROOT/'reference',ROOT/'input'/'xyz_files'); assert 0.45 < w['score'] < 1.0, w
    shortcut=json.loads(json.dumps(perfect)); shortcut['files']['pentahydrogen_carbon_bad.xyz']['problematic']=False; write_candidate(base/'shortcut',shortcut)
    q=score(base/'shortcut',ROOT/'reference',ROOT/'input'/'xyz_files'); assert q['components']['visible_classification'] < 1.0 and q['score'] < 1.0, q
    visible_only=base/'visible_only'; write_candidate(visible_only,perfect,analyzer=False)
    (visible_only/'analyzer.py').write_text('''import json,sys\nfrom pathlib import Path\nPath(sys.argv[2]).write_text(json.dumps({"files":{},"summary":{"problematic_count":0,"file_count":0,"all_finite":True}}))\n''')
    h=score(visible_only,ROOT/'reference',ROOT/'input'/'xyz_files'); assert h['components']['hidden_classification'] == 0.0 and abs(h['score']-0.75)<1e-12, h
print('adversarial checks passed')
