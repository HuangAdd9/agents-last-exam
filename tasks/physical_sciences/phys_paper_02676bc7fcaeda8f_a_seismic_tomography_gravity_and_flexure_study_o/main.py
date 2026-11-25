"""ALE task adapter for the Ka'ena-style gravity/flexure inversion."""
from __future__ import annotations
import csv, importlib.util, json, shlex, shutil, stat, sys, tempfile, types, uuid
from dataclasses import dataclass
from pathlib import Path
import numpy as np
try:
    import cua_bench as cb
    from tasks.linux_runtime import LinuxTaskConfig
except ModuleNotFoundError:
    class _CB:
        tasks_config=staticmethod(lambda **k:(lambda f:f)); setup_task=staticmethod(lambda **k:(lambda f:f)); evaluate_task=staticmethod(lambda **k:(lambda f:f))
        Task=lambda *a,**k:type('Task',(),k)
    cb=_CB()
    @dataclass
    class LinuxTaskConfig:
        DOMAIN_NAME:str='physical_sciences'; TASK_NAME:str='task'; VARIANT_NAME:str='base'; input_dir:str='input'; remote_output_dir:str='output'; reference_dir:str='reference'
        def to_metadata(self): return {'input_dir':self.input_dir,'remote_output_dir':self.remote_output_dir,'reference_dir':self.reference_dir}

TASK='phys_paper_02676bc7fcaeda8f_a_seismic_tomography_gravity_and_flexure_study_o'
UNITS={"velocity":"km/s","density":"kg/m^3","gravity":"mGal","flexure":"m","te":"km"}
KEYS={'density_kg_m3','gravity_mgal','flexure_m','te_km','gravity_rms_mgal','flexure_rms_m'}
# These limits are deliberately much larger than the reference artifacts but
# small enough that a candidate cannot make the host ingest an unbounded file.
PULL_LIMITS={'result.npz':16*1024*1024,'trace.csv':1*1024*1024,'summary.json':64*1024}

def _oracle():
    path=Path(__file__).resolve().parent/'scripts'/'oracle.py'; spec=importlib.util.spec_from_file_location('_kaena_private_oracle',path)
    if spec is None or spec.loader is None: raise RuntimeError('oracle unavailable')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def _npz(path,ref):
    try:
        z=np.load(path,allow_pickle=False)
        if set(z.files)!=KEYS: return None
        out={k:z[k] for k in z.files}
        for k,v in out.items():
            if not np.issubdtype(v.dtype,np.number) or not np.all(np.isfinite(v)): return None
            if v.ndim>2 or v.size>100000: return None
        if any(out[k].shape!=ref[k].shape for k in KEYS): return None
        return out
    except Exception: return None

def _trace(path,expected):
    try:
        with path.open(newline='') as f: rows=list(csv.reader(f))
        if not rows or rows[0]!=['te_km','flexure_rms_m'] or len(rows)!=len(expected)+1: return None
        vals=[]
        for row,e in zip(rows[1:],expected):
            if len(row)!=2: return None
            a,b=float(row[0]),float(row[1]);
            if not np.isfinite(a+b) or abs(a-e[0])>1e-9 or abs(b-e[1])>2e-5: return None
            vals.append((a,b))
        return vals
    except Exception: return None

def _summary(path,inv):
    try:
        d=json.loads(path.read_text())
        return set(d)=={'units','min_qp_minus_qsv'} and d['units']==UNITS and np.isfinite(float(d['min_qp_minus_qsv'])) and abs(float(d['min_qp_minus_qsv'])-inv)<=2e-7 and float(d['min_qp_minus_qsv'])>=0
    except Exception: return False

def score(local,ref):
    got=_npz(local/'result.npz',ref)
    if got is None: return {'score':0.0,'components':{'schema':0.0,'physical':0.0,'trace':0.0,'invariant':0.0}}
    tol={'density_kg_m3':2e-5,'gravity_mgal':2e-5,'flexure_m':2e-5,'te_km':1e-8,'gravity_rms_mgal':2e-5,'flexure_rms_m':2e-5}
    positivity=bool(np.all(got['density_kg_m3']>0) and np.all(got['te_km']>0))
    physical_arrays=bool(all(np.max(np.abs(got[k]-ref[k]))<=tol[k] for k in KEYS))
    # Positivity is a validity condition, not an independent shortcut point.
    physical=float(physical_arrays and positivity)
    trace=float(_trace(local/'trace.csv',ref['trace']) is not None)
    invariant=float(_summary(local/'summary.json',ref['invariant']))
    # No credit is available for a trace/invariant unless the physical arrays
    # themselves are correct. This prevents positive dummy arrays plus the
    # easily recomputed invariant from earning a fixed score.
    score=0.85*physical + 0.15*float(physical and trace and invariant)
    return {'score':score,'components':{'schema':1.0,'physical':physical,'trace':trace,'invariant':invariant,'positivity':float(positivity)}}

async def _run(session,cmd,timeout=30):
    r=await session.run_command(cmd,timeout=timeout,check=False); rc=r.get('returncode',0) if isinstance(r,dict) else getattr(r,'returncode',0)
    if rc!=0: raise RuntimeError(f'candidate failed ({rc})')

async def _isolated(session,solve_text,input_text,stage):
    q=shlex.quote(stage); await _run(session,f'rm -rf {q} && mkdir -m 700 {q}',30)
    await session.write_file(stage+'/solve.py',solve_text); await session.write_file(stage+'/problem.json',input_text)
    cmd=(f"command -v bwrap >/dev/null 2>&1 || exit 127; mkdir -m 700 {q}/run; "
         f"bwrap --die-with-parent --unshare-all --ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 --ro-bind /bin /bin --ro-bind /etc /etc --dev /dev --proc /proc --tmpfs /tmp --bind {q} /work --chdir /work /bin/sh -c 'ulimit -v 262144; exec /usr/bin/python3 /work/solve.py --input /work/problem.json --output /work/run'")
    await _run(session,cmd,30)
    for n in ('result.npz','trace.csv','summary.json'):
        if not await session.file_exists(stage+'/run/'+n): raise RuntimeError('missing '+n)

async def _pull(session,stage,local):
    """Copy candidate artifacts only after an lstat/size check.

    ``read_bytes`` follows links in the guest, so checking with lstat (rather
    than a symlink-following ``stat``) is essential.  The size cap also rejects
    sparse/stream-like files before any bytes cross the evaluator boundary.
    """
    for n in ('result.npz','trace.csv','summary.json'):
        remote=stage+'/run/'+n; q=shlex.quote(remote)
        probe=await session.run_command(
            f"/usr/bin/python3 -c 'import os,stat,sys; s=os.lstat(sys.argv[1]); "
            f"print(int(s.st_mode), int(s.st_size))' {q}", timeout=10, check=False)
        rc=probe.get('returncode',probe.get('return_code',0)) if isinstance(probe,dict) else getattr(probe,'returncode',getattr(probe,'return_code',0))
        if rc!=0: raise RuntimeError(f'missing or inaccessible {n}')
        text=probe.get('stdout',probe.get('output','')) if isinstance(probe,dict) else getattr(probe,'stdout',getattr(probe,'output',''))
        if isinstance(text,bytes): text=text.decode('ascii','strict')
        try: mode_s,size_s=str(text).strip().split()[-2:]; mode,size=int(mode_s),int(size_s)
        except Exception as exc: raise RuntimeError(f'bad lstat response for {n}') from exc
        if not stat.S_ISREG(mode) or size<0 or size>PULL_LIMITS[n]:
            raise RuntimeError(f'unsafe candidate artifact {n}')
        blob=await session.read_bytes(remote)
        if len(blob)>PULL_LIMITS[n] or len(blob)!=size: raise RuntimeError(f'unbounded read for {n}')
        (local/n).write_bytes(blob)

@dataclass
class Config(LinuxTaskConfig):
    VARIANT_NAME:str='base'; DOMAIN_NAME:str='physical_sciences'; TASK_NAME:str=TASK
    def to_metadata(self):
        m=dict(super().to_metadata()); m['task_id']=f'{self.DOMAIN_NAME}/{self.TASK_NAME}'; return m
    @property
    def task_description(self):
        return f"Read `{self.input_dir}/spec.md` and `{self.input_dir}/problem.json`. Implement the reusable NumPy solver `{self.remote_output_dir}/solve.py` with `--input PATH --output DIR`, producing result.npz, trace.csv, and summary.json exactly as specified. The evaluator reruns unchanged code on two hidden Ka'ena-style velocity geometries; do not hard-code visible values or access reference/."

@cb.tasks_config(split='train')
def load():
    c=Config(); return [cb.Task(description=c.task_description,metadata=c.to_metadata(),computer={'provider':'computer','setup_config':{'os_type':'linux'}})]

@cb.setup_task(split='train')
async def start(task_cfg,session):
    out=(task_cfg.metadata or {}).get('remote_output_dir','output'); await session.run_command(f'mkdir -p {shlex.quote(out)}',check=False)

@cb.evaluate_task(split='train')
async def evaluate(task_cfg,session):
    stages=[]; locals_=[]
    try:
        meta=task_cfg.metadata; out=meta['remote_output_dir']; inp=meta['input_dir']; refdir=meta['reference_dir']
        if not await session.file_exists(out+'/solve.py'): return [0.0]
        solve_text=await session.read_file(out+'/solve.py'); vis_text=await session.read_file(inp+'/problem.json'); vis=json.loads(vis_text)
        oracle=_oracle(); vref=oracle.solve_reference(vis); local=Path(tempfile.mkdtemp()); locals_.append(local); stage='/tmp/kaena-'+uuid.uuid4().hex; stages.append(stage)
        await _isolated(session,solve_text,vis_text,stage); await _pull(session,stage,local)
        vals=[score(local,vref)]
        for name in ('hidden_case.json','hidden_geometry.json'):
            htxt=await session.read_file(f'{refdir}/{name}'); h=json.loads(htxt); hs='/tmp/kaena-'+uuid.uuid4().hex; stages.append(hs)
            await _isolated(session,solve_text,htxt,hs); hlocal=Path(tempfile.mkdtemp()); locals_.append(hlocal); await _pull(session,hs,hlocal)
            vals.append(score(hlocal,oracle.solve_reference(h)))
        return [float(np.mean([v['score'] for v in vals]))]
    except Exception:
        return [0.0]
    finally:
        for stage in stages:
            try: await _run(session,'rm -rf '+shlex.quote(stage),30)
            except Exception: pass
        for local in locals_:
            shutil.rmtree(local,ignore_errors=True)
