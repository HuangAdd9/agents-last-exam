from __future__ import annotations
import json, math, subprocess, sys, tempfile
from pathlib import Path

RECORD_KEYS={"problematic","reasons","atom_count","bond_count","min_distance_A"}
def _valid_record(r):
    return isinstance(r,dict) and set(r)==RECORD_KEYS and isinstance(r["problematic"],bool) and isinstance(r["reasons"],list) and all(isinstance(x,str) for x in r["reasons"]) and isinstance(r["atom_count"],int) and isinstance(r["bond_count"],int) and isinstance(r["min_distance_A"],(int,float)) and math.isfinite(r["min_distance_A"])

def _diagnostic_match(got, expected):
    return (got["problematic"] == expected["problematic"]
            and got["reasons"] == expected["reasons"]
            and got["atom_count"] == expected["atom_count"]
            and got["bond_count"] == expected["bond_count"]
            and abs(got["min_distance_A"] - expected["min_distance_A"]) <= 1e-6)

def score(output_dir: Path, reference_dir: Path, visible_dir: Path):
    required = {"decisions.json": 1_000_000, "summary.json": 100_000, "analyzer.py": 50_000}
    for name, limit in required.items():
        path = output_dir / name
        if not path.is_file() or path.stat().st_size == 0 or path.stat().st_size > limit:
            return {"score":0.0,"components":{},"reason":"missing_empty_or_oversized:"+name}
    try:
        dec=json.loads((output_dir/"decisions.json").read_text()); summ=json.loads((output_dir/"summary.json").read_text())
    except Exception as e: return {"score":0.0,"components":{},"reason":"missing_or_bad_json:"+str(e)}
    exp=json.loads((reference_dir/"expected.json").read_text()); vf=exp["visible"]["files"]
    if set(dec)!={"files","summary"} or not isinstance(dec["files"],dict) or set(dec["files"])!=set(vf): return {"score":0.0,"components":{},"reason":"decisions_schema"}
    if any(not _valid_record(r) for r in dec["files"].values()): return {"score":0.0,"components":{},"reason":"record_schema"}
    if set(summ)!={"problematic_count","file_count","all_finite","units","algorithm"} or summ["units"]!="angstrom" or summ["algorithm"]!="covalent_graph_v1": return {"score":0.0,"components":{},"reason":"summary_schema"}
    exact=sum(dec["files"][k]["problematic"]==vf[k]["problematic"] for k in vf)/len(vf)
    diag=sum(_diagnostic_match(dec["files"][k],vf[k]) for k in vf)/len(vf)
    inv=float(dec["summary"]==exp["visible"]["summary"] and summ["file_count"]==len(vf) and summ["problematic_count"]==sum(r["problematic"] for r in dec["files"].values()) and summ["all_finite"] is True)
    hidden_score=0.0
    analyzer=output_dir/"analyzer.py"
    if analyzer.is_file():
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/"hidden.json"
            try:
                r=subprocess.run([sys.executable,str(analyzer),str(reference_dir/"hidden_xyz"),str(out)],capture_output=True,text=True,timeout=10)
                got=json.loads(out.read_text()) if r.returncode==0 else None
                hf=exp["hidden"]["files"]
                if isinstance(got,dict) and set(got)=={"files","summary"} and set(got["files"])==set(hf) and all(_valid_record(x) for x in got["files"].values()):
                    hidden_score=sum(got["files"][k]["problematic"]==hf[k]["problematic"] for k in hf)/len(hf)
            except Exception: pass
    score=0.45*exact+0.15*diag+0.25*hidden_score+0.15*inv
    return {"score":score,"components":{"visible_classification":exact,"visible_diagnostics":diag,"hidden_classification":hidden_score,"invariants":inv}}
