#!/usr/bin/env python3
"""Reference implementation used only by local adversarial validation."""
from __future__ import annotations
import argparse,csv,json,math
def num(x):
    try:return float(x)
    except:return None
def ranks(v):
    o=sorted(range(len(v)),key=lambda i:(v[i],i)); out=[0]*len(v);i=0
    while i<len(o):
        j=i+1
        while j<len(o) and v[o[j]]==v[o[i]]:j+=1
        r=(i+1+j)/2
        for k in o[i:j]:out[k]=r
        i=j
    return out
def main():
    a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--output',required=True);x=a.parse_args()
    with open(x.input,newline='') as f: rows=list(csv.DictReader(f))
    ec=[float(r['EC50_nM']) for r in rows]; pairs=[r for r in rows if num(r.get('LogD')) is not None]; le=[float(r['LogD']) for r in pairs]; ee=[float(r['EC50_nM']) for r in pairs]; ra,rb=ranks(ee),ranks(le); ma=sum(ra)/len(ra);mb=sum(rb)/len(rb);den=math.sqrt(sum((z-ma)**2 for z in ra)*sum((z-mb)**2 for z in rb));rho=sum((u-ma)*(v-mb) for u,v in zip(ra,rb))/den if den else 0.0
    out={'n_rows':len(rows),'best_potency_id':min(rows,key=lambda r:(float(r['EC50_nM']),r['Compound_ID']))['Compound_ID'],'median_ec50_nM':(sorted(ec)[len(ec)//2] if len(ec)%2 else (sorted(ec)[len(ec)//2-1]+sorted(ec)[len(ec)//2])/2),'numeric_hlm_count':sum(num(r['HLM_predCLint_L_h_kg']) is not None for r in rows),'censored_hlm_count':sum(r['HLM_predCLint_L_h_kg'].startswith('<') for r in rows),'not_tested_hlm_count':sum(r['HLM_predCLint_L_h_kg']=='NT' for r in rows),'spearman_ec50_logd':round(rho,10)}
    with open(x.output,'w') as f:json.dump(out,f,sort_keys=True,indent=2);f.write('\n')
if __name__=='__main__':main()
