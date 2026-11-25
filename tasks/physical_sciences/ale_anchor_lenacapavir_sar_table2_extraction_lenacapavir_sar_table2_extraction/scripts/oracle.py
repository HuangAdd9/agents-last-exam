#!/usr/bin/env python3
"""Deterministically build the hidden reference and hidden generalization fixture."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

CSV_FIELDS = ["Compound_ID","EC50_nM","HLM_predCLint_L_h_kg","LogD","pEC50","Potency_rank","HLM_class"]

def _num(s):
    try: return float(s)
    except (TypeError, ValueError): return None

def _rank(values):
    order = sorted(range(len(values)), key=lambda i: (values[i], i))
    out = [0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]: end += 1
        r = (pos + 1 + end) / 2.0
        for j in order[pos:end]: out[j] = r
        pos = end
    return out

def summarize(rows):
    ec = [float(r["EC50_nM"]) for r in rows]
    logd = [float(r["LogD"]) for r in rows if _num(r.get("LogD")) is not None]
    ec_for_logd = [float(r["EC50_nM"]) for r in rows if _num(r.get("LogD")) is not None]
    def ranks(v): return _rank(v)
    a, b = ranks(ec_for_logd), ranks(logd)
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    den = math.sqrt(sum((x-ma)**2 for x in a) * sum((y-mb)**2 for y in b))
    rho = sum((x-ma)*(y-mb) for x,y in zip(a,b))/den if den else 0.0
    best = min(rows, key=lambda r: (float(r["EC50_nM"]), r["Compound_ID"]))["Compound_ID"]
    med = sorted(ec)[(len(ec)-1)//2] if len(ec)%2 else (sorted(ec)[len(ec)//2-1]+sorted(ec)[len(ec)//2])/2
    return {"n_rows":len(rows),"best_potency_id":best,"median_ec50_nM":med,
            "numeric_hlm_count":sum(_num(r["HLM_predCLint_L_h_kg"]) is not None for r in rows),
            "censored_hlm_count":sum(str(r["HLM_predCLint_L_h_kg"]).startswith("<") for r in rows),
            "not_tested_hlm_count":sum(r["HLM_predCLint_L_h_kg"]=="NT" for r in rows),
            "spearman_ec50_logd":round(rho,10)}

def enrich(rows):
    ranks = _rank([float(r["EC50_nM"]) for r in rows])
    out=[]
    for r, rank in zip(rows, ranks):
        ec=float(r["EC50_nM"])
        h=str(r["HLM_predCLint_L_h_kg"])
        cls="not_tested" if h=="NT" else "censored_low" if h.startswith("<") else "measured"
        out.append({**r,"pEC50":f"{9-math.log10(ec):.6f}","Potency_rank":f"{rank:.1f}" if rank%1 else str(int(rank)),"HLM_class":cls})
    return out

def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=CSV_FIELDS); w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--source", type=Path, required=True); ap.add_argument("--out", type=Path, required=True); a=ap.parse_args()
    data=json.loads(a.source.read_text(encoding="utf-8")); rows=data["rows"]
    a.out.mkdir(parents=True, exist_ok=True)
    enriched=enrich(rows); write_csv(a.out/"reference.csv", enriched)
    (a.out/"reference_analysis.json").write_text(json.dumps(summarize(rows),sort_keys=True,indent=2)+"\n",encoding="utf-8")
    hidden=[]
    for i,r in enumerate(rows[:7],1):
        ec=float(r["EC50_nM"])*(1.0+0.05*i); h=r["HLM_predCLint_L_h_kg"]
        if _num(h) is not None: h=f"{float(h)*(1+0.1*i):.6g}"
        hidden.append({"Compound_ID":f"H{i}","EC50_nM":f"{ec:.6g}","HLM_predCLint_L_h_kg":h,"LogD":("NT" if r["LogD"]=="NT" else f"{float(r['LogD'])+0.1*((i%3)-1):.6g}")})
    write_csv(a.out/"hidden_table.csv", hidden)
    (a.out/"hidden_analysis.json").write_text(json.dumps(summarize(hidden),sort_keys=True,indent=2)+"\n",encoding="utf-8")
    (a.out/"oracle_manifest.json").write_text(json.dumps({"source_sha256":data["source"]["sha256"],"rows":len(rows),"hidden_rows":len(hidden)},sort_keys=True,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__": main()
