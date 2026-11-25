"""Authoritative standard-library oracle for the EMC interpolation fixture."""
import argparse,csv,json,math
FIELDS=["path_id","segment_index","length_km","theta_deg","azimuth_deg","phase_normal_deg","model_lat_deg","model_lon_deg","model_depth_km","v_qp_km_s","v_qsv_km_s","v_qsh_km_s","dt_qp_s","dt_qsv_s","dt_qsh_s"]
def br(x,a):
 if x<=a[0]:return 0,0.
 if x>=a[-1]:return len(a)-2,1.
 i=next(i for i in range(len(a)-1) if a[i]<=x<=a[i+1]);return i,(x-a[i])/(a[i+1]-a[i])
def mat(p,c):
 g=p['model_grid'];il,fl=br(c[0],g['lat']);io,fo=br(c[1],g['lon']);iz,fz=br(c[2],g['depth_km']);o={}
 for n in ('vpv','vph','vsv','vsh','eta'):
  a=g['values'][n];v=0.
  for z,wz in ((0,1-fz),(1,fz)):
   for y,wy in ((0,1-fl),(1,fl)):
    for x,wx in ((0,1-fo),(1,fo)):v+=wz*wy*wx*a[iz+z][il+y][io+x]
  o[n]=v
 return o
def phase(s):
 c33,c11,c44,c66=s['vpv']**2,s['vph']**2,s['vsv']**2,s['vsh']**2; c13=math.sqrt((c33-c44)**2+2*s['delta']*c33*(c33-c44))-c44;t=math.radians(s['phase_deg']);x,z=math.sin(t),math.cos(t);a=c11*x*x+c44*z*z;d=c44*x*x+c33*z*z;b=(c13+c44)*x*z;q=math.hypot(a-d,2*b);return {'qp':math.sqrt((a+d+q)/2),'qsv':math.sqrt((a+d-q)/2),'qsh':math.sqrt(c66*x*x+c44*z*z)}
def sup(s,mode,t):
 def f(ph):
  q=dict(s);q['phase_deg']=math.degrees(ph);c=math.cos(ph-t);return c/phase(q)[mode] if c>1e-10 else -math.inf
 lo=max(-math.pi/2+1e-7,t-math.pi/2+1e-7);hi=min(math.pi/2-1e-7,t+math.pi/2-1e-7);xs=[lo+(hi-lo)*i/96 for i in range(97)];k=max(range(97),key=lambda i:f(xs[i]));a,b=xs[max(0,k-1)],xs[min(96,k+1)];g=(math.sqrt(5)-1)/2;c,d=b-g*(b-a),a+g*(b-a)
 for _ in range(48):
  if f(c)>f(d):b,d=d,c;c=b-g*(b-a)
  else:a,c=c,d;d=a+g*(b-a)
 ph=(a+b)/2;return f(ph),math.degrees(ph)
def calc(p):
 out=[];tr=[];gap=1e99;xmax=-1e99
 for r in p['rays']:
  tt=[0.,0.,0.];sv=sh=0.;ns=r['nodes_xyz_km'];mn=r['model_nodes']
  for i,s0 in enumerate(r['segments']):
   a,b=ns[i],ns[i+1];d=[b[k]-a[k] for k in range(3)];L=math.sqrt(sum(x*x for x in d));th=math.degrees(math.acos(d[2]/L));az=math.degrees(math.atan2(d[1],d[0]))%360;mid=[(mn[i][k]+mn[i+1][k])/2 for k in range(3)];m=mat(p,mid);s={**m,'delta':s0['delta']};v=phase({**s,'phase_deg':th});ss=[];pn=0
   for mode in ('qp','qsv','qsh'):
    q,pp=sup(s,mode,math.radians(th));ss.append(q);pn=pp if mode=='qp' else pn
   ds=[L*x for x in ss];tt=[tt[j]+ds[j] for j in range(3)];sv+=ds[1];sh+=ds[2];gap=min(gap,v['qp']-v['qsv']);xmax=max(xmax,(m['vsh']/m['vsv'])**2)
   tr.append({'path_id':r['path_id'],'segment_index':i,'length_km':L,'theta_deg':th,'azimuth_deg':az,'phase_normal_deg':pn,'model_lat_deg':mid[0],'model_lon_deg':mid[1],'model_depth_km':mid[2],'v_qp_km_s':v['qp'],'v_qsv_km_s':v['qsv'],'v_qsh_km_s':v['qsh'],'dt_qp_s':ds[0],'dt_qsv_s':ds[1],'dt_qsh_s':ds[2]})
  out.append({'path_id':r['path_id'],'t_qp_s':tt[0],'t_qsv_s':tt[1],'t_qsh_s':tt[2],'xi_eff':(sh/sv)**-2})
 return {'schema_version':'spiral_emc_v3','visible':out,'invariant':{'min_qp_minus_qsv_km_s':gap,'max_xi_material':xmax}},tr
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--input',required=True);a.add_argument('--result',required=True);a.add_argument('--trace',required=True);x=a.parse_args();o,t=calc(json.load(open(x.input)));json.dump(o,open(x.result,'w'),indent=2);w=csv.DictWriter(open(x.trace,'w',newline=''),fieldnames=FIELDS);w.writeheader();w.writerows(t)
