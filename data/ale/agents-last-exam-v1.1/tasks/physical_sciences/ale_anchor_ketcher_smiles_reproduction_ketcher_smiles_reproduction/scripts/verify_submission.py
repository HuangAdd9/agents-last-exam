#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, re, subprocess, sys
from pathlib import Path

REQUIRED_SUB={'smiles','formula','heavy_atom_count','ring_count','atom_counts'}
REQUIRED_INV={'atom_counts','heavy_atom_count','source_sha256'}

def load_json(p, keys):
    x=json.loads(Path(p).read_text(encoding='utf-8'))
    if not isinstance(x,dict) or set(x)!=keys: raise ValueError(f'exact keys required: {sorted(keys)}')
    return x

def smiles_atoms(s):
    toks=re.findall(r'Br|Cl|[BCNOFPSI]|[cnops]',s)
    out={}
    for t in toks:
        t={'c':'C','n':'N','o':'O','p':'P','s':'S'}.get(t,t)
        out[t]=out.get(t,0)+1
    return out

def ring_closures(s):
    # Every ring label pair (including %nn) is one closure.
    labels=re.findall(r'%\d{2}|\d',s)
    return len(labels)//2

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',required=True); ap.add_argument('--reference',required=True); ap.add_argument('--svg',required=True); ap.add_argument('--hidden',required=True); a=ap.parse_args()
    payload={'score':0.0,'components':{},'reason':''}
    try:
        out=Path(a.output); ref=Path(a.reference); svg=Path(a.svg); hidden=Path(a.hidden)
        sub=load_json(out/'submission.json',REQUIRED_SUB); inv=load_json(out/'atom_inventory.json',REQUIRED_INV)
        r=json.loads(ref.read_text()); he=json.loads((ref.parent/'hidden_expected.json').read_text())
        if hashlib.sha256(svg.read_bytes()).hexdigest()!=inv['source_sha256']: raise ValueError('visible SVG hash mismatch')
        if inv['heavy_atom_count']!=sum(int(v) for v in inv['atom_counts'].values()): raise ValueError('inventory heavy count invariant')
        if sub['heavy_atom_count']!=sum(int(v) for v in sub['atom_counts'].values()): raise ValueError('submission heavy count invariant')
        if sub['atom_counts']!=r['atom_counts']: raise ValueError('atom inventory does not match SVG/source')
        if sub['formula']!=r['formula'] or sub['heavy_atom_count']!=r['heavy_atom_count'] or sub['ring_count']!=r['ring_count']: raise ValueError('formula/invariant mismatch')
        if sub['smiles'] not in {r['smiles_openeye'],r['smiles_cactvs']}: raise ValueError('SMILES is not a sourced 7YY representation')
        if smiles_atoms(sub['smiles'])!=r['atom_counts'] or ring_closures(sub['smiles'])!=r['ring_count']: raise ValueError('SMILES atom/ring cross-check failed')
        inv_score=0.15; identity=0.35; phys=0.30; hidden_score=0.20
        payload['components']={'schema_and_hash':inv_score,'chemical_identity':identity,'physical_invariants':phys,'hidden_generalization':0.0}
        script=out/'solve.py'
        if not script.exists(): raise ValueError('missing solve.py')
        src=script.read_text(encoding='utf-8')
        if r['smiles_openeye'] in src or r['smiles_cactvs'] in src: raise ValueError('solve.py embeds answer SMILES')
        payload['score']=inv_score+identity+phys
        payload['reason']='visible checks passed; hidden SVG mutation failed'
        try:
            run=subprocess.run([sys.executable,str(script),str(hidden)],capture_output=True,text=True,timeout=5)
            got=json.loads(run.stdout)
            if run.returncode==0 and set(got)=={'atom_counts','heavy_atom_count'} and got==he:
                payload['components']['hidden_generalization']=hidden_score
                payload['score']=round(payload['score']+hidden_score,10)
                payload['reason']='ok'
        except Exception as hidden_error:
            payload['hidden_error']=f'{type(hidden_error).__name__}:{hidden_error}'
    except Exception as e:
        payload['reason']=f'{type(e).__name__}:{e}'
    print(json.dumps(payload,sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
