"""Evaluator for the SPiRaL/EMC model-interpolation tomography exercise."""
from __future__ import annotations
import asyncio, csv, io, json, math, os, resource, shutil, signal, subprocess, sys, sysconfig, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

DOMAIN_NAME="physical_sciences"; TASK_NAME="phys_paper_c2a4326b1e9d33e0_spiral_a_multiresolution_global_tomography_model"; VARIANT_NAME="base"
_setup=BaseTaskSetup()
FIELDS=["path_id","segment_index","length_km","theta_deg","azimuth_deg","phase_normal_deg","model_lat_deg","model_lon_deg","model_depth_km","v_qp_km_s","v_qsv_km_s","v_qsh_km_s","dt_qp_s","dt_qsv_s","dt_qsh_s"]

@dataclass
class SpiralVTIConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME
    @property
    def problem_file(self): return f"{self.input_dir}/problem.json"
    @property
    def spec_file(self): return f"{self.input_dir}/spec.md"
    @property
    def result_file(self): return f"{self.remote_output_dir}/result.json"
    @property
    def trace_file(self): return f"{self.remote_output_dir}/trace.csv"
    @property
    def solver_file(self): return f"{self.remote_output_dir}/solver.py"
    @property
    def task_description(self): return f"Read `{self.spec_file}` and `{self.problem_file}`. Implement a reusable standard-library-only SPiRaL/EMC model-interpolation ray workflow. Interpolate the embedded licensed EMC extract at each segment midpoint, derive 3-D length/theta/azimuth from nodes, apply Christoffel and stationary group-slowness physics, and write `{self.result_file}`, `{self.trace_file}`, and `{self.solver_file}`. The evaluator reruns your solver on visible and two independent hidden coarse/fine model mutations; do not hard-code answers, use internet/NumPy/SciPy, or modify input/."
    def to_metadata(self):
        m=super().to_metadata(); m.update({"task_id":f"{DOMAIN_NAME}/{TASK_NAME}","problem_file":self.problem_file,"spec_file":self.spec_file,"result_file":self.result_file,"trace_file":self.trace_file,"solver_file":self.solver_file}); return m

def _bracket(x,a):
    if x<=a[0]: return 0,0.0
    if x>=a[-1]: return len(a)-2,1.0
    i=next(i for i in range(len(a)-1) if a[i]<=x<=a[i+1]); return i,(x-a[i])/(a[i+1]-a[i])
def _material(problem,coord):
    g=problem["model_grid"]; lat,lon,dep=map(float,coord); il,fl=_bracket(lat,g["lat"]); io,fo=_bracket(lon,g["lon"]); iz,fz=_bracket(dep,g["depth_km"]); out={}
    for name in ("vpv","vph","vsv","vsh","eta"):
        a=g["values"][name]; val=0.
        for dz,wz in ((0,1-fz),(1,fz)):
          for dy,wy in ((0,1-fl),(1,fl)):
            for dx,wx in ((0,1-fo),(1,fo)): val+=wz*wy*wx*float(a[iz+dz][il+dy][io+dx])
        out[name]=val
    return out
def _stiffness(s):
    c33,c11,c44,c66=s["vpv"]**2,s["vph"]**2,s["vsv"]**2,s["vsh"]**2; d=s["delta"]; c13=math.sqrt((c33-c44)**2+2*d*c33*(c33-c44))-c44; return c11,c33,c44,c66,c13
def _phase(s):
    c11,c33,c44,c66,c13=_stiffness(s); t=math.radians(s["phase_deg"]); x,z=math.sin(t),math.cos(t); aa=c11*x*x+c44*z*z; dd=c44*x*x+c33*z*z; bb=(c13+c44)*x*z; di=math.hypot(aa-dd,2*bb); return {"qp":math.sqrt((aa+dd+di)/2),"qsv":math.sqrt((aa+dd-di)/2),"qsh":math.sqrt(c66*x*x+c44*z*z)}
def _support(s,mode,theta):
    def f(ph):
        q=dict(s); q["phase_deg"]=math.degrees(ph); c=math.cos(ph-theta); return c/_phase(q)[mode] if c>1e-10 else -math.inf
    lo=max(-math.pi/2+1e-7,theta-math.pi/2+1e-7); hi=min(math.pi/2-1e-7,theta+math.pi/2-1e-7); n=96; xs=[lo+(hi-lo)*i/n for i in range(n+1)]; k=max(range(n+1),key=lambda i:f(xs[i])); a,b=xs[max(0,k-1)],xs[min(n,k+1)]; gr=(math.sqrt(5)-1)/2; c,d=b-gr*(b-a),a+gr*(b-a)
    for _ in range(48):
        if f(c)>f(d): b,d=d,c; c=b-gr*(b-a)
        else: a,c=c,d; d=a+gr*(b-a)
    ph=(a+b)/2; return f(ph),math.degrees(ph)
def _segment(seg,problem,midpoint,theta):
    m=_material(problem,midpoint); s={**m,"delta":float(seg["delta"])}; p=_phase({**s,"phase_deg":theta}); out={"v_qp_km_s":p["qp"],"v_qsv_km_s":p["qsv"],"v_qsh_km_s":p["qsh"],"material":m}
    for mode,key in (("qp","s_qp"),("qsv","s_qsv"),("qsh","s_qsh")): out[key],pn=_support(s,mode,math.radians(theta)); out.setdefault("phase_normal_deg",pn)
    return out
def _angle(dx,dy,dz,L): return math.degrees(math.acos(max(-1,min(1,dz/L)))),math.degrees(math.atan2(dy,dx))%360
def _compute(problem):
    visible=[]; trace=[]; gap=math.inf; xmax=-math.inf
    for ray in problem["rays"]:
        nodes=ray["nodes_xyz_km"]; mn=ray["model_nodes"]; segs=ray["segments"]
        if len(nodes)!=len(segs)+1 or len(mn)!=len(nodes): raise ValueError("node count")
        if any(abs(float(ray["source_xyz_km"][k])-float(nodes[0][k]))>1e-7 for k in range(3)) or any(abs(float(ray["receiver_xyz_km"][k])-float(nodes[-1][k]))>1e-7 for k in range(3)): raise ValueError("endpoints")
        t=[0.,0.,0.]; sv=sh=0.
        for i,seg in enumerate(segs):
            a,b=nodes[i],nodes[i+1]; dx,dy,dz=[float(b[k])-float(a[k]) for k in range(3)]; L=math.sqrt(dx*dx+dy*dy+dz*dz); th,az=_angle(dx,dy,dz,L)
            if abs(L-float(seg["length_km"]))>2e-6*max(1,L) or abs(th-float(seg["theta_deg"]))>2e-5 or abs(((az-float(seg["azimuth_deg"])+180)%360)-180)>2e-5: raise ValueError("geometry heading")
            mid=[(float(mn[i][k])+float(mn[i+1][k]))/2 for k in range(3)]; v=_segment(seg,problem,mid,th); dts=[L*v[x] for x in ("s_qp","s_qsv","s_qsh")]; t=[t[j]+dts[j] for j in range(3)]; sv+=dts[1]; sh+=dts[2]; m=v["material"]; gap=min(gap,v["v_qp_km_s"]-v["v_qsv_km_s"]); xmax=max(xmax,(m["vsh"]/m["vsv"])**2)
            trace.append({"path_id":ray["path_id"],"segment_index":i,"length_km":L,"theta_deg":th,"azimuth_deg":az,"phase_normal_deg":v["phase_normal_deg"],"model_lat_deg":mid[0],"model_lon_deg":mid[1],"model_depth_km":mid[2],"v_qp_km_s":v["v_qp_km_s"],"v_qsv_km_s":v["v_qsv_km_s"],"v_qsh_km_s":v["v_qsh_km_s"],"dt_qp_s":dts[0],"dt_qsv_s":dts[1],"dt_qsh_s":dts[2]})
        visible.append({"path_id":ray["path_id"],"t_qp_s":t[0],"t_qsv_s":t[1],"t_qsh_s":t[2],"xi_eff":(sh/sv)**-2})
    return {"schema_version":"spiral_emc_v3","visible":visible,"invariant":{"min_qp_minus_qsv_km_s":gap,"max_xi_material":xmax}},trace
def _num(a,b,t):
    try: return max(0.,1.-abs(float(a)-float(b))/t) if math.isfinite(float(a)) and math.isfinite(float(b)) else 0.
    except: return 0.
def _score(out,tr,ref):
    if set(out)!={"schema_version","visible","invariant"} or out.get("schema_version")!="spiral_emc_v3" or len(out.get("visible",[]))!=len(ref.get("visible",[])): return 0.
    vals=[]
    for a,b in zip(out["visible"],ref["visible"]):
      if a.get("path_id")!=b.get("path_id"): return 0.
      vals += [_num(a[k],b[k],5e-5) for k in ("t_qp_s","t_qsv_s","t_qsh_s","xi_eff")]
    inv=out.get("invariant",{}); ri=ref.get("invariant",{}); iv=[_num(inv.get(k),ri.get(k),2e-5) for k in ("min_qp_minus_qsv_km_s","max_xi_material")]
    try:
      if float(inv["min_qp_minus_qsv_km_s"])<0 or not .8<=float(inv["max_xi_material"])<=1.3:return 0.
      rows=list(csv.DictReader(io.StringIO(tr))); rr=ref["trace"]
      if len(rows)!=len(rr) or (rows and list(rows[0])!=FIELDS):return 0.
      tv=[]
      for a,b in zip(rows,rr):
       if a["path_id"]!=b["path_id"] or int(a["segment_index"])!=int(b["segment_index"]):return 0.
       tv += [_num(a[k],b[k],2e-5) for k in FIELDS[2:]]
      return .5*sum(vals)/len(vals)+.15*sum(iv)/2+.35*sum(tv)/len(tv)
    except:return 0.
def _independent_case_score(problem,out,tr):
    try: expected,rows=_compute(problem); return _score(out,tr,{**expected,"trace":rows})
    except:return 0.
def _cmd(stage,inp,res,tr):
    b=shutil.which("bwrap"); exe=Path(sys.executable).resolve(); std=Path(sysconfig.get_paths().get("stdlib","")).resolve()
    if not b or not exe.is_file() or not std.is_dir():return None
    c=[b,"--die-with-parent","--new-session","--unshare-all","--tmpfs","/"]
    for x in (exe,std,Path("/lib"),Path("/lib64"),Path("/usr/lib/x86_64-linux-gnu")):
      if x.exists():c += ["--ro-bind",str(x),str(x)]
    return c+["--proc","/proc","--dev","/dev","--bind",str(stage),"/sandbox","--chdir","/sandbox","--clearenv","--setenv","PATH","/usr/bin:/bin","--setenv","PYTHONNOUSERSITE","1","--setenv","PYTHONDONTWRITEBYTECODE","1","--",str(exe),"-I","-S","/sandbox/solver.py","--input",f"/sandbox/{inp}","--result",f"/sandbox/{res}","--trace",f"/sandbox/{tr}"]
async def _run(stage,inp,res,tr):
    c=_cmd(stage,inp,res,tr)
    if c is None:return None
    def lim():
      resource.setrlimit(resource.RLIMIT_CPU,(8,8));resource.setrlimit(resource.RLIMIT_AS,(256*1024*1024,)*2);resource.setrlimit(resource.RLIMIT_FSIZE,(4*1024*1024,)*2);resource.setrlimit(resource.RLIMIT_NPROC,(32,32));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    p=await asyncio.to_thread(subprocess.Popen,c,cwd=str(stage),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True,preexec_fn=lim)
    try:o,e=await asyncio.to_thread(p.communicate,timeout=12);return subprocess.CompletedProcess(c,p.returncode,o,e)
    except subprocess.TimeoutExpired:
      try:os.killpg(p.pid,signal.SIGKILL)
      except ProcessLookupError:pass
      await asyncio.to_thread(p.communicate);return None
config=SpiralVTIConfig()
@cb.tasks_config(split="train")
def load():
    c=SpiralVTIConfig();return [cb.Task(description=c.task_description,metadata=c.to_metadata(),computer={"provider":"computer","setup_config":{"os_type":"linux"}})]
@cb.setup_task(split="train")
async def start(task_cfg,session):await _setup(task_cfg,session)
@cb.evaluate_task(split="train")
async def evaluate(task_cfg,session):
  m=task_cfg.metadata
  try:
    solver=await session.read_file(m["solver_file"]);inp=await session.read_file(m["problem_file"])
    if len(solver.encode())>100000 or len(inp.encode())>1024*1024:return [0.]
    cases=[inp,await session.read_file(f'{m["reference_dir"]}/hidden_problem.json'),await session.read_file(f'{m["reference_dir"]}/hidden_problem_2.json')];scores=[]
    for idx,text in enumerate(cases):
      with tempfile.TemporaryDirectory(prefix="spiral_case_") as td:
       p=Path(td);(p/"solver.py").write_text(solver);(p/"case.json").write_text(text);r=p/"result.json";t=p/"trace.csv";proc=await _run(p,"case.json","result.json","trace.csv");names={x.name for x in p.iterdir()}
       if proc is None or proc.returncode!=0 or not r.exists() or not t.exists() or names!={"solver.py","case.json","result.json","trace.csv"} or r.stat().st_size>256*1024 or t.stat().st_size>4*1024*1024:return [0.]
       out=json.loads(r.read_text());tr=t.read_text();scores.append(_score(out,tr,json.loads(await session.read_file(f'{m["reference_dir"]}/reference.json'))) if idx==0 else _independent_case_score(json.loads(text),out,tr))
    if scores[1]<.9 or scores[2]<.9:return [max(0.,min(.79,scores[0]*.79))]
    return [max(0.,min(1.,scores[0]))]
  except:return [0.]
