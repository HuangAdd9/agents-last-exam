#!/usr/bin/env python3
import csv, json, math, sys
from pathlib import Path

def pd(x,y):
 d=abs(float(x)-float(y)); return min(d,1.0-d)
def solve(case):
 g=case['grid']; n=int(g['n']); v=case['valence']; c=case['conduction']; gw=case['gw_self_energy']; valley=c['valley']
 rows=[]
 for ix in range(n):
  for iy in range(n):
   for iz in range(n):
    k=[ix/n,iy/n,iz/n]; s=[math.sin(math.pi*x) for x in k]; q=sum(x*x for x in s)
    ev=v['e0']-v['curvature']*q+v['warping']*math.sin(2*math.pi*k[0])*math.sin(2*math.pi*k[1])*math.sin(2*math.pi*k[2])
    d=[pd(k[j],valley[j]) for j in range(3)]; ec=c['e0']+sum(c['curvature'][j]*d[j]*d[j] for j in range(3))+c['warping']*math.cos(2*math.pi*k[0])*math.cos(2*math.pi*k[1])*math.cos(2*math.pi*k[2])
    f=math.exp(-math.sqrt(sum(x*x for x in d))/gw['screening_length'])
    evq=ev+gw['valence_shift']-gw['nonlocal_strength']*(1-f); ecq=ec+gw['conduction_shift']+gw['nonlocal_strength']*(1-f)
    rows.append((k,ev,ec,evq,ecq))
 iv=max(range(len(rows)),key=lambda i:rows[i][1]); ic=min(range(len(rows)),key=lambda i:rows[i][2]); iqv=max(range(len(rows)),key=lambda i:rows[i][3]); iqc=min(range(len(rows)),key=lambda i:rows[i][4])
 def ex(i,b): return {'k':[float(x) for x in rows[i][0]],'energy_eV':float(rows[i][b])}
 dgap=rows[ic][2]-rows[iv][1]; qgap=rows[iqc][4]-rows[iqv][3]
 mf={'units':'eV','grid_shape':[n,n,n],'vbm':ex(iv,1),'cbm':ex(ic,2),'indirect_gap_eV':float(dgap)}
 qp={'units':'eV','vbm':ex(iqv,3),'cbm':ex(iqc,4),'indirect_gap_eV':float(qgap),'gap_opening_eV':float(qgap-dgap),'direct_gamma_gap_eV':float(rows[0][4]-rows[0][3]),'valley_declared':[float(x) for x in valley]}
 p=case['path']; pts=p['points']; seg=int(p['segments']); path=[]; dist=0.0
 for si in range(len(pts)-1):
  a,b=pts[si],pts[si+1]
  for j in range(seg+1):
   if si and j==0: continue
   t=j/seg; k=[(1-t)*a[z]+t*b[z] for z in range(3)]
   s=[math.sin(math.pi*x) for x in k]; q=sum(x*x for x in s); ev=v['e0']-v['curvature']*q+v['warping']*math.sin(2*math.pi*k[0])*math.sin(2*math.pi*k[1])*math.sin(2*math.pi*k[2]); d=[pd(k[z],valley[z]) for z in range(3)]; ec=c['e0']+sum(c['curvature'][z]*d[z]*d[z] for z in range(3))+c['warping']*math.cos(2*math.pi*k[0])*math.cos(2*math.pi*k[1])*math.cos(2*math.pi*k[2]); f=math.exp(-math.sqrt(sum(x*x for x in d))/gw['screening_length']); evq=ev+gw['valence_shift']-gw['nonlocal_strength']*(1-f); ecq=ec+gw['conduction_shift']+gw['nonlocal_strength']*(1-f)
   if path: dist += math.sqrt(sum((k[z]-path[-1]['k'][z])**2 for z in range(3)))
   path.append({'segment':si,'index':j,'distance':dist,'ev_dft':ev,'ec_dft':ec,'ev_qp':evq,'ec_qp':ecq,'k':k})
 diag={'units':'eV','grid_shape':[n,n,n],'structure_atom_count':2,'vbm_at_gamma':math.sqrt(sum(x*x for x in rows[iv][0]))<=1/n+1e-12,'cbm_near_declared_valley':math.sqrt(sum(pd(rows[ic][0][j],valley[j])**2 for j in range(3)))<=math.sqrt(3)/n+1e-12,'qp_gap_opening_positive':qgap>dgap,'direct_gamma_gap_eV':path[0]['ec_qp']-path[0]['ev_qp'],'path_rows':len(path)}
 return {'mean_field':mf,'quasiparticle':qp,'diagnostics':diag,'path':path}
def write(case,out):
 o=Path(out); o.mkdir(parents=True,exist_ok=True); r=solve(case)
 for name,key in [('mean_field.json','mean_field'),('quasiparticle.json','quasiparticle'),('diagnostics.json','diagnostics')]: (o/name).write_text(json.dumps(r[key],sort_keys=True,indent=2)+'\n')
 with (o/'band_path.csv').open('w',newline='') as f:
  w=csv.writer(f); w.writerow(['segment','index','distance','ev_dft','ec_dft','ev_qp','ec_qp']); [w.writerow([x[k] for k in ['segment','index','distance','ev_dft','ec_dft','ev_qp','ec_qp']]) for x in r['path']]
if __name__=='__main__':
 if len(sys.argv)!=3: raise SystemExit('usage: solve.py CASE.json RESULT.json')
 out=sys.argv[2]
 if out.endswith('.json'):
  tmp=Path(out).with_suffix(''); write(json.loads(Path(sys.argv[1]).read_text()),tmp); payload={k:json.loads((tmp/(k+'.json')).read_text()) for k in ['mean_field','quasiparticle','diagnostics']}; payload['band_path_csv']=(tmp/'band_path.csv').read_text(); Path(out).write_text(json.dumps(payload,sort_keys=True))
 else: write(json.loads(Path(sys.argv[1]).read_text()),out)
