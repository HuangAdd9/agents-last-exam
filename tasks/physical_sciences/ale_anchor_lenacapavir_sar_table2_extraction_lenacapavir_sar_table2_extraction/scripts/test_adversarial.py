#!/usr/bin/env python3
from __future__ import annotations
import csv, json, shutil, tempfile
from pathlib import Path
from score_outputs import FULL_FIELDS, score

ROOT=Path(__file__).resolve().parents[1]; REF=ROOT/"reference"
def run():
    with tempfile.TemporaryDirectory(prefix="lenacapavir_adv_") as td:
        t=Path(td); perfect=t/"perfect";perfect.mkdir()
        shutil.copy2(REF/"reference.csv",perfect/"submission.csv")
        # A correct general-purpose analysis script is supplied as a candidate fixture below.
        shutil.copy2(ROOT/"scripts/reference_analysis_impl.py",perfect/"analyze_table.py")
        shutil.copy2(REF/"reference_analysis.json",perfect/"analysis.json")
        print("perfect",score(perfect,REF)["score"])
        empty=t/"empty";empty.mkdir();(empty/"submission.csv").write_text("bad\n")
        print("empty",score(empty,REF)["score"])
        wrong=t/"wrong_units";shutil.copytree(perfect,wrong);p=wrong/"submission.csv"
        with p.open(newline="") as f: rows=list(csv.DictReader(f))
        for row in rows: row["EC50_nM"]=str(float(row["EC50_nM"])/1000.0)
        with p.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=FULL_FIELDS);w.writeheader();w.writerows(rows)
        print("wrong_units",score(wrong,REF)["score"])
        shortcut=t/"shortcut";shutil.copytree(perfect,shortcut);(shortcut/"analyze_table.py").write_text("import argparse,json; a=argparse.ArgumentParser();a.add_argument('--input');a.add_argument('--output');x=a.parse_args();json.dump({'n_rows':17},open(a.output,'w'))")
        print("hidden_shortcut",score(shortcut,REF)["score"])
        assert score(perfect,REF)["score"]==1.0
        assert score(empty,REF)["score"]==0.0
        assert score(wrong,REF)["score"]==0.65
        assert score(shortcut,REF)["score"]==0.9
if __name__=="__main__":run()
