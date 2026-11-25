#!/usr/bin/env python3
"""Deterministically derive evaluator artifacts from the staged SVG and source record."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path

def atom_counts(svg: str) -> dict[str,int]:
    vals = re.findall(r'<text[^>]*>\s*([A-Z][a-z]?)\s*</text>', svg)
    out: dict[str,int] = {}
    for e in vals:
        if e == 'H': continue
        out[e] = out.get(e,0)+1
    # SVG labels omit carbon; carbon count is sourced from the verified formula.
    return out

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--svg',required=True); ap.add_argument('--record',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args(); svg_path=Path(a.svg); rec=json.loads(Path(a.record).read_text())
    raw=svg_path.read_bytes(); txt=raw.decode('utf-8'); counts=atom_counts(txt)
    full=dict(rec['atom_counts']); full.update(counts); full['C']=rec['atom_counts']['C']
    inv={'atom_counts':full,'heavy_atom_count':sum(full.values()),'source_sha256':hashlib.sha256(raw).hexdigest()}
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    (out/'reference.json').write_text(json.dumps({**rec,'visible_source_sha256':inv['source_sha256']},sort_keys=True,indent=2)+'\n')
    (out/'visible_inventory.json').write_text(json.dumps(inv,sort_keys=True,indent=2)+'\n')
    hidden=txt.replace('>  F </text>','>  Br </text>',1)
    hc=atom_counts(hidden); hfull=dict(full); hfull['F']=hfull.get('F',0)-1; hfull['Br']=hfull.get('Br',0)+1
    (out/'hidden_input').mkdir(exist_ok=True)
    (out/'hidden_input/structure.svg').write_text(hidden)
    (out/'hidden_expected.json').write_text(json.dumps({'atom_counts':hfull,'heavy_atom_count':sum(hfull.values())},sort_keys=True,indent=2)+'\n')
if __name__=='__main__': main()
