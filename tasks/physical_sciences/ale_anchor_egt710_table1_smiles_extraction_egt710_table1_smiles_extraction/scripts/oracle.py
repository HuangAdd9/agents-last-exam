"""Deterministic hidden-reference generator for the EGT710 extraction task."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

try:
    from .chem_utils import bookkeeping
except ImportError:  # direct execution: python3 scripts/oracle.py
    from chem_utils import bookkeeping

ROOT = Path(__file__).resolve().parents[1]
SI = ROOT / "oracle_data" / "source_si.csv"
TABLE = ROOT / "oracle_data" / "table1_reported.json"
REF = ROOT / "reference"


def _si_rows() -> dict[str, dict[str, str]]:
    with SI.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    # The first two lines are a two-level header.  Column 0 is the compound id,
    # column 1 is the canonical SMILES, and column 2 is the geometric-mean
    # RapidFire AC50 in micromolar.
    out = {}
    for row in rows[2:]:
        if len(row) >= 3 and row[0].strip() in {str(i) for i in range(1, 10)}:
            out[row[0].strip()] = {"smiles": row[1].strip(), "ic50_uM": row[2].strip()}
    if set(out) != {str(i) for i in range(1, 10)}:
        raise RuntimeError(f"SI rows missing: {sorted(out)}")
    return out


def _measurement(value: str) -> tuple[str, float]:
    value = value.strip().replace("μ", "u")
    if value.startswith("<"):
        return "<", float(value[1:].strip())
    return "=", float(value)


def generate() -> dict:
    si = _si_rows()
    table = json.loads(TABLE.read_text(encoding="utf-8"))
    rows = []
    for cid in map(str, range(1, 10)):
        # Grade the rounded assay values that are actually visible in Table 1.
        # The SI supplies the canonical structure strings, but its higher
        # precision assay measurements are not visible to the solver and must
        # not become an impossible hidden target.
        ic50_um = float(table[cid]["ic50_uM"])
        sol_op, sol_mm = _measurement(table[cid]["solubility_mM"])
        ic50_nm = ic50_um * 1000.0
        p_ic50 = 9.0 - math.log10(ic50_nm)
        rows.append({
            "Compound_ID": cid,
            "SMILES": si[cid]["smiles"],
            "IC50_nM": f"{ic50_nm:.9g}",
            "Solubility_uM": f"{sol_op}{sol_mm * 1000.0:.9g}" if sol_op == "<" else f"{sol_mm * 1000.0:.9g}",
            "pIC50": f"{p_ic50:.9f}",
        })
    REF.mkdir(exist_ok=True)
    with (REF / "reference.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Compound_ID", "SMILES", "IC50_nM", "Solubility_uM", "pIC50"])
        w.writeheader(); w.writerows(rows)
    audit = {r["Compound_ID"]: bookkeeping(r["SMILES"]) for r in rows}
    (REF / "reference_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    hidden = {"Compound_ID": "10", "SMILES": "C#Cc2ncc1ccccc1c2N4C(=O)Nc3ccc(cc3C4=O)C(F)(F)F", "IC50_nM": "9.701408", "Solubility_uM": "NA"}
    (REF / "hidden_transform_case.json").write_text(json.dumps(hidden, indent=2) + "\n", encoding="utf-8")
    return {"rows": len(rows), "reference": str(REF / "reference.csv")}


if __name__ == "__main__":
    print(json.dumps(generate(), sort_keys=True))
