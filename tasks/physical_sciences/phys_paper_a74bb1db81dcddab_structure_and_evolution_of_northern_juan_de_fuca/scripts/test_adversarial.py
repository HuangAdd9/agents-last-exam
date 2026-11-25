"""Local grader gate: perfect and adversarial submissions."""
from pathlib import Path
import csv, json, shutil, tempfile, sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from oracle import read_picks
from verify_outputs import score_submission

ROOT=Path(__file__).resolve().parents[1]
REF=json.loads((ROOT/"reference/reference.json").read_text())
ROWS=read_picks(ROOT/"input/picks.csv")

def perfect(d):
    d.mkdir(); (d/"model.json").write_text(json.dumps(REF["model"]))
    with (d/"predictions.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["phase","offset_km","predicted_time_s","residual_ms"])
        for r,p in zip(ROWS,REF["visible_predictions"]): w.writerow([r["phase"],r["x"],p,(p-r["t"])*1000])
    with (d/"inversion_trace.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["iteration","objective"]); w.writerows([[0,REF["objective"]*1.02],[1,REF["objective"]*1.01],[2,REF["objective"]]])
    (d/"diagnostics.json").write_text(json.dumps({"rms_ms":REF["rms_ms"],"reduced_chi2":REF["reduced_chi2"],"n_picks":len(ROWS),"invariant_ok":True}))

def score(d): return score_submission(d,ROOT/"input/picks.csv",ROOT/"reference/hidden_picks.csv",ROOT/"reference/reference.json")

with tempfile.TemporaryDirectory() as td:
    base=Path(td)/"perfect"; perfect(base); p=score(base); assert abs(p["score"]-1.0)<1e-12, p
    empty=Path(td)/"empty"; empty.mkdir(); e=score(empty); assert e["score"]==0.0, e
    copied=Path(td)/"copied"; shutil.copytree(base,copied)
    with (copied/"predictions.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["phase","offset_km","predicted_time_s","residual_ms"])
        for r in ROWS:w.writerow([r["phase"],r["x"],r["t"],0.0])
    c=score(copied); assert c["components"]["visible"]<0.9 and c["score"]<0.98, c
    wrong=Path(td)/"wrong_units"; shutil.copytree(base,wrong)
    m=REF["model"].copy(); m.update({k:v*1000 for k,v in m.items()}); (wrong/"model.json").write_text(json.dumps(m))
    u=score(wrong); assert u["score"]<0.45 and u["components"]["hidden"]==0.0, u
    inv=Path(td)/"bad_invariant"; shutil.copytree(base,inv)
    m=REF["model"].copy(); m["v1_km_s"],m["v2_km_s"]=m["v2_km_s"],m["v1_km_s"]; (inv/"model.json").write_text(json.dumps(m))
    (inv/"diagnostics.json").write_text(json.dumps({"rms_ms":REF["rms_ms"],"reduced_chi2":REF["reduced_chi2"],"n_picks":len(ROWS),"invariant_ok":False}))
    b=score(inv); assert b["components"]["invariant_and_diagnostics"]==0.0 and b["score"]<0.5, b
print("adversarial checks passed; perfect=1.0")
