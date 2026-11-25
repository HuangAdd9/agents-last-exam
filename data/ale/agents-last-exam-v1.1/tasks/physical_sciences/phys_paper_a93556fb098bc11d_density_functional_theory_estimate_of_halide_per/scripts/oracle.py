"""Deterministic oracle for a bounded orbital/SOC band-edge model."""
from __future__ import annotations
import csv, json, math
from pathlib import Path

KB=8.617333262e-5; TS=5778.0
GRID=tuple(-math.pi+2*math.pi*i/8 for i in range(9))
REQUIRED_INPUT=['material_id','A','B','X','a_A','c_A','tilt_deg']
ALLOWED={'A':{'Cs','FA','MA'},'B':{'Ge','Sn','Pb'},'X':{'Cl','Br','I'}}
A_V={'Cs':0.00,'FA':0.05,'MA':0.08}; B_V={'Ge':-0.10,'Sn':-0.04,'Pb':0.06}
B_C={'Ge':0.18,'Sn':0.08,'Pb':-0.02}; X_V={'Cl':-0.18,'Br':-0.05,'I':0.16}
X_C={'Cl':0.22,'Br':0.06,'I':-0.12}; LAMBDA={'Ge':0.0,'Sn':0.0,'Pb':0.32}

def validate_rows(rows):
    if not rows or list(rows[0].keys())!=REQUIRED_INPUT: raise ValueError('exact input header required')
    ids=[]
    for r in rows:
        if list(r.keys())!=REQUIRED_INPUT or not r['material_id'] or r['material_id'] in ids: raise ValueError('bad id/schema')
        ids.append(r['material_id'])
        for k in ('A','B','X'):
            if r[k] not in ALLOWED[k]: raise ValueError('unknown chemistry')
        for k in ('a_A','c_A','tilt_deg'):
            if not math.isfinite(float(r[k])): raise ValueError('nonfinite geometry')
        if not (2.5<=float(r['a_A'])<=7.5 and 2.5<=float(r['c_A'])<=7.5 and 0<=float(r['tilt_deg'])<=35): raise ValueError('geometry range')
    return rows

def read_rows(path):
    with open(path,newline='',encoding='utf-8') as f: return validate_rows(list(csv.DictReader(f)))

def _edges(r):
    a,c,tilt=float(r['a_A']),float(r['c_A']),float(r['tilt_deg'])
    ev0=-0.80+A_V[r['A']]+B_V[r['B']]+X_V[r['X']]+0.15*(a-5.8)-0.05*(c-5.8)
    ec0=1.40+0.30*A_V[r['A']]+B_C[r['B']]+X_C[r['X']]-0.08*(a-5.8)+0.12*(c-5.8)
    tv=-0.18*(1+tilt/90); tc=0.22*(1-tilt/120)
    bv=(-float('inf'),None); bc=(float('inf'),None)
    for ix,kx in enumerate(GRID):
      for iy,ky in enumerate(GRID):
       for iz,kz in enumerate(GRID):
        q=(math.cos(kx)+math.cos(ky)+math.cos(kz))/3
        ev=ev0+2*tv*q; ec=ec0+2*tc*q
        if ev>bv[0]: bv=(ev,(ix,iy,iz))
        if ec<bc[0]: bc=(ec,(ix,iy,iz))
    soc=r['B']=='Pb'; ix,iy,iz=bv[1]; q=(math.cos(GRID[ix])+math.cos(GRID[iy])+math.cos(GRID[iz]))/3
    # Diagonalise the explicit 2x2 valence SOC block [[Ev,lambda*g],[lambda*g,Ev]].
    coupling=LAMBDA[r['B']]*((1-q)/2) if soc else 0.0
    vbm=bv[0]+math.sqrt(coupling*coupling); cbm=bc[0]
    return cbm-vbm,vbm,cbm,soc,bv[1],bc[1]

def capture(gap):
    if not math.isfinite(gap) or gap<0.001 or gap>8.0: return 0.0
    es=[0.001*i for i in range(1,8001)]; phi=[e*e/math.expm1(e/(KB*TS)) for e in es]
    den=sum((es[i+1]-es[i])*(es[i]*phi[i]+es[i+1]*phi[i+1])/2 for i in range(len(es)-1))
    sel=[i for i,e in enumerate(es) if e>=gap]
    if len(sel)<2:return 0.0
    num=sum((es[i+1]-es[i])*(phi[i]+phi[i+1])/2 for i in sel if i+1<len(es))
    return gap*num/den

def compute(rows):
    validate_rows(rows); bands=[]; caps=[]
    for r in rows:
        gap,vbm,cbm,soc,vk,ck=_edges(r); bands.append((r['material_id'],gap,soc,vbm,cbm,vk,ck,r['B'])); caps.append((r['material_id'],capture(gap)))
    return {'band_gaps':bands,'spectral_capture':caps,
      'summary':{'n_materials':len(rows),'n_pb_soc':sum(int(x[2]) for x in bands),'mean_gap_eV':sum(x[1] for x in bands)/len(bands),'mean_capture_fraction':sum(x[1] for x in caps)/len(caps),'best_material_id':max(enumerate(caps),key=lambda z:z[1][1])[1][0]},
      'invariants':{'finite_positive_gaps':all(math.isfinite(x[1]) and x[1]>0 for x in bands),'soc_only_for_pb':all(x[2]==(x[7]=='Pb') for x in bands),'row_order_preserved':[x[0] for x in bands]==[r['material_id'] for r in rows]}}

def write_reference(csv_path,out_dir):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True); d=compute(read_rows(csv_path))
    with open(out/'band_gaps.csv','w',newline='',encoding='utf-8') as f:
      w=csv.writer(f); w.writerow(['material_id','band_gap_eV','soc_applied'])
      for mid,g,soc,*_ in d['band_gaps']: w.writerow([mid,f'{g:.9f}',str(soc).lower()])
    with open(out/'spectral_capture.csv','w',newline='',encoding='utf-8') as f:
      w=csv.writer(f); w.writerow(['material_id','capture_fraction'])
      for mid,c in d['spectral_capture']: w.writerow([mid,f'{c:.11f}'])
    for name,key in [('summary.json','summary'),('invariants.json','invariants')]: (out/name).write_text(json.dumps(d[key],sort_keys=True,indent=2)+'\n',encoding='utf-8')
    (out/'materials_meta.json').write_text(json.dumps({'ids':[x[0] for x in d['band_gaps']],'B':{x[0]:x[7] for x in d['band_gaps']}},sort_keys=True,indent=2)+'\n')
    return d

if __name__=='__main__':
    import argparse; p=argparse.ArgumentParser(); p.add_argument('csv'); p.add_argument('out'); a=p.parse_args(); write_reference(a.csv,a.out)
