#!/usr/bin/env python3
"""Local deterministic scorer for the table-extraction task."""
from __future__ import annotations
import argparse, csv, json, math, subprocess, sys, tempfile
from pathlib import Path

BASE_FIELDS=["Compound_ID","EC50_nM","HLM_predCLint_L_h_kg","LogD"]
FULL_FIELDS=BASE_FIELDS+["pEC50","Potency_rank","HLM_class"]

def _load_csv(p):
    if not p.exists() or p.stat().st_size>100_000: raise ValueError("missing/oversized csv")
    with p.open(newline="",encoding="utf-8-sig") as f:
        r=csv.DictReader(f)
        if r.fieldnames!=FULL_FIELDS: raise ValueError("exact output headers required")
        rows=list(r)
    if not rows or len(rows)>100 or len({x.get("Compound_ID","") for x in rows})!=len(rows): raise ValueError("empty or duplicate rows")
    return rows

def _num(s):
    try:return float(s)
    except (TypeError,ValueError):return None

def _same_num(a,b,tol=1e-9):
    try:return abs(float(a)-float(b))<=tol*max(1.0,abs(float(b)))
    except (TypeError,ValueError):return str(a).strip()==str(b).strip()

def _score_hidden_script(script, hidden_csv, expected):
    if not script.exists() or script.stat().st_size>30_000:return 0.0
    with tempfile.TemporaryDirectory(prefix="lenacapavir_hidden_") as td:
        out=Path(td)/"hidden.json"
        try:
            r=subprocess.run([sys.executable,str(script),"--input",str(hidden_csv),"--output",str(out)],capture_output=True,text=True,timeout=5)
        except Exception:return 0.0
        if r.returncode!=0 or not out.exists():return 0.0
        try: got=json.loads(out.read_text())
        except Exception:return 0.0
    return 1.0 if got==expected else 0.0

def score(output_dir: Path, reference_dir: Path) -> dict:
    result={"score":0.0,"components":{},"reason":""}
    try:
        cand=_load_csv(output_dir/"submission.csv")
        ref=_load_csv(reference_dir/"reference.csv")
    except Exception as e:
        result["reason"]=f"schema_gate:{e}";return result
    if {r["Compound_ID"] for r in cand}!={r["Compound_ID"] for r in ref}:
        result["reason"]="compound_id_set_mismatch";return result
    refmap={r["Compound_ID"]:r for r in ref}; base=0; derived=0
    for r in cand:
        q=refmap[r["Compound_ID"]]
        if all(str(r[k]).strip()==str(q[k]).strip() for k in BASE_FIELDS):base+=1
        if all(str(r[k]).strip()==str(q[k]).strip() for k in FULL_FIELDS[4:]):derived+=1
    n=len(ref)
    try:
        analysis=json.loads((output_dir/"analysis.json").read_text())
        expected=json.loads((reference_dir/"reference_analysis.json").read_text())
        analysis_ok=analysis==expected
    except Exception: analysis_ok=False
    hidden=_score_hidden_script(output_dir/"analyze_table.py",reference_dir/"hidden_table.csv",json.loads((reference_dir/"hidden_analysis.json").read_text()))
    result["components"]={"schema":1.0,"extraction":base/n,"derived":derived/n,"analysis":1.0 if analysis_ok else 0.0,"hidden_generalization":hidden}
    result["score"]=round(0.10+0.35*base/n+0.20*derived/n+0.25*(1.0 if analysis_ok else 0.0)+0.10*hidden,10)
    result["reason"]="ok" if result["score"]==1.0 else "component_mismatch"
    result["matched_rows"]=base;result["total_rows"]=n
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--reference-dir",type=Path,required=True);a=ap.parse_args();print(json.dumps(score(a.output_dir,a.reference_dir),sort_keys=True))
if __name__=="__main__":main()
