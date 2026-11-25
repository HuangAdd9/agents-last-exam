"""Local adversarial checks; expected component scores are asserted."""
from __future__ import annotations
import csv, json, shutil, subprocess, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERIFY = ROOT / "scripts" / "verify_submission.py"
REF = ROOT / "reference" / "reference.csv"
AUD = ROOT / "reference" / "reference_audit.json"
HID = ROOT / "reference" / "hidden_transform_case.json"

def run(case: str, maker):
    with tempfile.TemporaryDirectory(prefix="egt710_case_") as td:
        d = Path(td); out = d / "output"; out.mkdir(); maker(out)
        p = subprocess.run(["python3", str(VERIFY), "--agent", str(out / "submission.csv"), "--ref", str(REF), "--reference-audit", str(AUD), "--hidden-case", str(HID), "--output-dir", str(out)], capture_output=True, text=True)
        payload = json.loads(p.stdout)
        print(case, payload)
        return payload

def perfect(out):
    shutil.copy2(REF, out / "submission.csv"); shutil.copy2(AUD, out / "audit.json")
    (out / "transform.py").write_text('''import argparse,json,math,sys\nfrom pathlib import Path\nsys.path.insert(0,''' + repr(str(ROOT / "scripts")) + ''')\nfrom chem_utils import bookkeeping\nap=argparse.ArgumentParser();ap.add_argument("--input");ap.add_argument("--output");a=ap.parse_args();d=json.loads(Path(a.input).read_text()); result={"Compound_ID":str(d["Compound_ID"]),"pIC50":9-math.log10(float(d["IC50_nM"]))}; result.update(bookkeeping(d["SMILES"])); Path(a.output).write_text(json.dumps(result))\n''')

def empty(out): pass
def wrong_units(out):
    perfect(out)
    rows=list(csv.DictReader((out/"submission.csv").open())); rows[0]["IC50_nM"]="0.0048"
    with (out/"submission.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def shortcut(out):
    perfect(out); rows=list(csv.DictReader((out/"submission.csv").open())); rows[1]["SMILES"]=rows[0]["SMILES"]
    with (out/"submission.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def hidden_shortcut(out):
    perfect(out)
    (out / "transform.py").write_text('''import argparse,json\nfrom pathlib import Path\nap=argparse.ArgumentParser();ap.add_argument("--input");ap.add_argument("--output");a=ap.parse_args();Path(a.output).write_text(json.dumps({"Compound_ID":"1","pIC50":5.318509567,"heavy_atoms":24,"hetero_atoms":6,"fluorine_atoms":4,"chiral_marks":0}))\n''')

if __name__ == "__main__":
    p=run("perfect",perfect); assert p["score"] == 1.0 and p["components"]["hidden_transform"] == 1.0
    e=run("empty",empty); assert e["score"] == 0.0
    u=run("wrong_units",wrong_units); assert u["components"]["assay_units"] < 1.0 and 0.0 < u["score"] < 1.0
    s=run("shortcut",shortcut); assert s["components"]["smiles_identity"] < 1.0 and 0.0 < s["score"] < 1.0
    h=run("hidden_shortcut",hidden_shortcut); assert h["components"]["hidden_transform"] == 0.0 and h["score"] == 0.95
