from __future__ import annotations
import csv,json,math
from pathlib import Path

FILES=('band_gaps.csv','spectral_capture.csv','summary.json','invariants.json')
BG=['material_id','band_gap_eV','soc_applied']; SC=['material_id','capture_fraction']
INV=['finite_positive_gaps','soc_only_for_pb','row_order_preserved']
def _num(x):
    try: return math.isfinite(float(x))
    except Exception: return False
def _rows(path,header):
    with path.open(newline='',encoding='utf-8') as f:
        rd=csv.DictReader(f)
        if rd.fieldnames!=header:return None
        rows=list(rd)
    if any(set(r.keys())!=set(header) or any(r[k] is None for k in header) for r in rows):return None
    return rows
def _close(a,b,t):
    try:return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a)-float(b))<=t
    except Exception:return False
def score_dirs(out:Path,ref:Path):
    try:
        for n in FILES:
            if not (out/n).is_file() or not (ref/n).is_file():return {'score':0.0,'components':{}}
        bg,rb=_rows(out/'band_gaps.csv',BG),_rows(ref/'band_gaps.csv',BG)
        sc,rs=_rows(out/'spectral_capture.csv',SC),_rows(ref/'spectral_capture.csv',SC)
        s=json.loads((out/'summary.json').read_text()); sr=json.loads((ref/'summary.json').read_text())
        inv=json.loads((out/'invariants.json').read_text()); meta=json.loads((ref/'materials_meta.json').read_text())
    except Exception:return {'score':0.0,'components':{}}
    if bg is None or rb is None or sc is None or rs is None or len(bg)!=len(rb) or len(sc)!=len(rs) or not bg or not sc:return {'score':0.0,'components':{}}
    ids=[r['material_id'] for r in rb]
    if [r['material_id'] for r in bg]!=ids or [r['material_id'] for r in sc]!=ids:return {'score':0.0,'components':{}}
    if set(meta.get('ids',[]))!=set(ids) or any(k not in meta.get('B',{}) for k in ids):return {'score':0.0,'components':{}}
    if any(not _num(r['band_gap_eV']) or r['soc_applied'] not in ('true','false') for r in bg):return {'score':0.0,'components':{}}
    if any(not _num(r['capture_fraction']) or float(r['capture_fraction'])<0 for r in sc):return {'score':0.0,'components':{}}
    if set(s)!=set(sr) or set(inv)!=set(INV):return {'score':0.0,'components':{}}
    if any(type(inv[k]) is not bool for k in INV):return {'score':0.0,'components':{}}
    finite=all(_num(r['band_gap_eV']) and float(r['band_gap_eV'])>0 for r in bg)
    socok=all((r['soc_applied']=='true')==(meta['B'][r['material_id']]=='Pb') for r in bg)
    order=[r['material_id'] for r in bg]==ids
    inv_expected={'finite_positive_gaps':finite,'soc_only_for_pb':socok,'row_order_preserved':order}
    invscore=float(all(inv[k] is inv_expected[k] for k in INV))
    gap=sum(_close(a['band_gap_eV'],b['band_gap_eV'],2e-5) for a,b in zip(bg,rb))/len(rb)
    policy=sum((a['soc_applied']==b['soc_applied']) and ((a['soc_applied']=='true')==(meta['B'][a['material_id']]=='Pb')) for a,b in zip(bg,rb))/len(rb)
    cap=sum(_close(a['capture_fraction'],b['capture_fraction'],3e-5) for a,b in zip(sc,rs))/len(rs)
    summary=all((s[k]==sr[k] if k in ('n_materials','n_pb_soc','best_material_id') else _close(s.get(k),sr.get(k),3e-5)) for k in sr)
    c={'schema':1.0,'band_gaps':gap,'soc_policy':policy,'spectral_capture':cap,'summary':float(summary),'invariants':invscore}
    return {'score':.10+.40*gap+.20*policy+.15*cap+.10*float(summary)+.05*invscore,'components':c}
