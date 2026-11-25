"""Independent dependency-free verifier for the transformed SAR extraction."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from .chem_utils import bookkeeping
except ImportError:  # direct execution: python3 scripts/verify_submission.py
    from chem_utils import bookkeeping

COLS = ["Compound_ID", "SMILES", "IC50_nM", "Solubility_uM", "pIC50"]
IDS = {str(i) for i in range(1, 10)}


def _rows(path: Path) -> dict[str, dict[str, str]]:
    if path.suffix.lower() != ".csv" or path.stat().st_size > 1_000_000:
        raise ValueError("bad or oversized CSV")
    with path.open(newline="", encoding="utf-8-sig") as f:
        r = csv.DictReader(f)
        if r.fieldnames != COLS:
            raise ValueError(f"headers must be exactly {COLS}")
        out = {}
        for row in r:
            cid = (row.get("Compound_ID") or "").strip()
            if cid in out or not cid:
                raise ValueError("blank or duplicate Compound_ID")
            out[cid] = {k: (row.get(k) or "").strip() for k in COLS}
    if set(out) != IDS:
        raise ValueError("Compound_ID set must be 1..9")
    return out


def _num(s: str) -> float:
    return float(s.replace("μ", "u").strip())


def _sol(s: str) -> tuple[str, float]:
    s = s.strip()
    if s.startswith("<"):
        return "<", float(s[1:])
    return "=", float(s)


def _run_hidden_transform(output_dir: Path, hidden_case: Path) -> bool:
    script = output_dir / "transform.py"
    if not script.is_file() or script.stat().st_size > 200_000:
        return False
    with tempfile.TemporaryDirectory(prefix="egt710_hidden_") as td:
        out = Path(td) / "result.json"
        try:
            p = subprocess.run(
                [sys.executable, str(script), "--input", str(hidden_case), "--output", str(out)],
                cwd=str(output_dir), capture_output=True, text=True, timeout=5,
            )
        except Exception:
            return False
        if p.returncode != 0 or not out.is_file():
            return False
        try:
            got = json.loads(out.read_text(encoding="utf-8"))
            case = json.loads(hidden_case.read_text())
            expected_pic50 = 9.0 - math.log10(float(case["IC50_nM"]))
            expected_audit = bookkeeping(case["SMILES"])
            return (
                abs(float(got["pIC50"]) - expected_pic50) <= 1e-6
                and got.get("Compound_ID") == "10"
                and all(got.get(key) == value for key, value in expected_audit.items())
            )
        except Exception:
            return False


def verify(agent: Path, reference: Path, reference_audit: Path, hidden_case: Path, output_dir: Path) -> dict:
    result = {"score": 0.0, "components": {}, "reason": ""}
    try:
        a = _rows(agent); r = _rows(reference)
        matched_smiles = sum(a[k]["SMILES"] == r[k]["SMILES"] for k in IDS) / 9
        numeric = 0
        pic = 0
        for k in IDS:
            try:
                if abs(_num(a[k]["IC50_nM"]) - _num(r[k]["IC50_nM"])) <= 1e-6:
                    sa, sr = _sol(a[k]["Solubility_uM"]), _sol(r[k]["Solubility_uM"])
                    if sa[0] == sr[0] and abs(sa[1] - sr[1]) <= 1e-6:
                        numeric += 1
                if abs(_num(a[k]["pIC50"]) - _num(r[k]["pIC50"])) <= 1e-6:
                    pic += 1
            except Exception:
                pass
        numeric /= 9; pic /= 9
        audit_path = output_dir / "audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
        ref_audit = json.loads(reference_audit.read_text(encoding="utf-8"))
        audit_ok = 0
        for k in IDS:
            if audit.get(k) == ref_audit.get(k) and audit.get(k, {}).get("heavy_atoms", 0) >= audit.get(k, {}).get("hetero_atoms", 0):
                audit_ok += 1
        audit_ok /= 9
        hidden = 1.0 if _run_hidden_transform(output_dir, hidden_case) else 0.0
        components = {"row_schema": 1.0, "smiles_identity": matched_smiles, "assay_units": numeric, "pIC50": pic, "audit_invariants": audit_ok, "hidden_transform": hidden}
        result["components"] = components
        result["score"] = 0.15 + 0.35 * matched_smiles + 0.25 * numeric + 0.10 * pic + 0.10 * audit_ok + 0.05 * hidden
        result["reason"] = "ok" if result["score"] == 1.0 else "component_mismatch"
    except Exception as exc:
        result["reason"] = f"hard_gate:{type(exc).__name__}:{exc}"
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True); ap.add_argument("--ref", required=True)
    ap.add_argument("--reference-audit", required=True); ap.add_argument("--hidden-case", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    print(json.dumps(verify(Path(args.agent), Path(args.ref), Path(args.reference_audit), Path(args.hidden_case), Path(args.output_dir)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
