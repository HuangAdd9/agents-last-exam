"""Authoritative evaluator for the period-dependent Rayleigh inversion task."""
from __future__ import annotations
import json, shlex, tempfile, sys, types, uuid, importlib.util
from pathlib import Path
from dataclasses import dataclass
import numpy as np
if __name__ not in sys.modules: sys.modules[__name__] = types.ModuleType(__name__)
try:
    import cua_bench as cb
    from tasks.linux_runtime import LinuxTaskConfig
except ModuleNotFoundError:
    class _CB:
        Task=lambda *a,**k:type('Task',(),k); tasks_config=staticmethod(lambda **k:(lambda f:f))
        setup_task=staticmethod(lambda **k:(lambda f:f)); evaluate_task=staticmethod(lambda **k:(lambda f:f))
    cb=_CB()
    class LinuxTaskConfig:
        DOMAIN_NAME='physical_sciences'; TASK_NAME='task'; VARIANT_NAME='base'; input_dir='input'; remote_output_dir='output'; reference_dir='reference'
        def to_metadata(self): return {'input_dir':self.input_dir,'remote_output_dir':self.remote_output_dir,'reference_dir':self.reference_dir}
UNITS={"velocity":"km/s","slowness":"s/km","rms_residual":"s/km","roughness":"(s/km)^2"}

def _differences(nx,ny):
    n=nx*ny; rows=[]
    for y in range(ny):
        for x in range(nx-1):
            r=np.zeros(n); r[y*nx+x]=1.; r[y*nx+x+1]=-1.; rows.append(r)
    for y in range(ny-1):
        for x in range(nx):
            r=np.zeros(n); r[y*nx+x]=1.; r[(y+1)*nx+x]=-1.; rows.append(r)
    # A 1x1 (or 1xN/ Nx1) grid has no finite-difference rows.  Keep the
    # matrix two-dimensional so the normal equations remain well-defined.
    return np.asarray(rows,dtype=float).reshape((-1,n)) if rows else np.empty((0,n),dtype=float)

def _solve(inp:dict)->dict:
    """Convenience duplicate; grading references are produced by scripts/oracle.py."""
    K=np.asarray(inp['ray_matrix'],float); d=np.asarray(inp['observed_slowness_s_per_km'],float)
    ny,nx=int(inp['grid']['ny']),int(inp['grid']['nx']); n=nx*ny; L=_differences(nx,ny)
    var=np.asarray(inp['data_variance_s2_per_km2'],float); prior=np.asarray(inp['model_variance_s2_per_km2'],float)
    m0=np.asarray(inp['initial_model_slowness_s_per_km'],float); lam=float(inp['regularization_lambda']); mu=float(inp['prior_weight'])
    steps=int(inp['iterations']); alpha=float(inp['update_damping']); out=[]; pred=[]; rms=[]; rough=[]
    for p in range(len(d)):
        A=K[p]; W=np.diag(1./var[p]); C=np.diag(1./prior); m=m0.copy(); H=A.T@W@A+lam*(L.T@L)+mu*C+1e-10*np.eye(n)
        for _ in range(steps):
            g=A.T@W@(d[p]-A@m)-mu*C@(m-m0)-lam*L.T@(L@m); m += alpha*np.linalg.solve(H,g)
        q=A@m; out.append((1./m).reshape(ny,nx)); pred.append(q); rms.append(np.sqrt(np.mean((q-d[p])**2))); rough.append(float(np.mean((L@m)**2)) if L.shape[0] else 0.0)
    return {'velocity_km_s':np.asarray(out),'predicted_slowness_s_per_km':np.asarray(pred),'rms_residual_s_per_km':np.asarray(rms),'roughness_slowness':np.asarray(rough)}

def _validate_payload(p:Path, expected:dict|None=None):
    z=np.load(p,allow_pickle=False); req={'velocity_km_s','predicted_slowness_s_per_km','rms_residual_s_per_km','roughness_slowness'}
    if set(z.files)!=req: raise ValueError('wrong npz keys')
    out={k:z[k] for k in z.files}
    if expected is not None and any(out[k].shape!=tuple(expected[k]) for k in req): raise ValueError('wrong shapes')
    if not all(np.all(np.isfinite(v)) for v in out.values()): raise ValueError('nonfinite')
    return out
def _diagnostics(path:Path, expected_max:float):
    try:
        d=json.loads(path.read_text())
        return (set(d)=={'max_abs_residual_s_per_km','units'}
                and np.isfinite(float(d['max_abs_residual_s_per_km']))
                and abs(float(d['max_abs_residual_s_per_km'])-expected_max)<=2e-7
                and d['units']==UNITS)
    except Exception:
        return False

def _oracle_max_residual(reference:dict, observations):
    """Expected diagnostic derived from the private oracle, never candidate arrays."""
    return float(np.max(np.abs(np.asarray(reference['predicted_slowness_s_per_km'])
                            - np.asarray(observations))))

def score_dir(outdir:Path, ref:dict, hidden_ref:dict|None=None, obs=None, hidden_obs=None):
    try:
        expected={k:ref[k].shape for k in ('velocity_km_s','predicted_slowness_s_per_km','rms_residual_s_per_km','roughness_slowness')}; got=_validate_payload(outdir/'result.npz',expected)
        if obs is None: raise ValueError('missing observations')
        # Diagnostics are checked against the independently recomputed oracle
        # residual, rather than the candidate's self-reported prediction.
        diagnostics=float(_diagnostics(outdir/'diagnostics.json',_oracle_max_residual(ref,obs)))
        vis=float(np.mean([np.max(np.abs(got[k]-ref[k])) <= (1e-7 if k.startswith('velocity') else 2e-7) for k in got]))
        invariant=float(np.all(got['velocity_km_s']>0) and np.all(got['rms_residual_s_per_km']>=0) and np.all(got['roughness_slowness']>=0))
    except Exception: return {'score':0.0,'components':{'schema':0.0,'visible':0.0,'diagnostics':0.0,'invariant':0.0,'hidden':0.0}}
    hidden=0.0
    if hidden_ref is not None:
        try:
            h=_validate_payload(outdir/'hidden_result.npz',{k:hidden_ref[k].shape for k in hidden_ref})
            hidden=float(all(np.max(np.abs(h[k]-hidden_ref[k])) < (1e-7 if k.startswith('velocity') else 2e-7) for k in h) and _diagnostics(outdir/'hidden_diagnostics.json',_oracle_max_residual(hidden_ref,hidden_obs)))
        except Exception: hidden=0.0
    return {'score':0.35*vis+0.15*diagnostics+0.15*invariant+0.35*hidden,'components':{'schema':1.0,'visible':vis,'diagnostics':diagnostics,'invariant':invariant,'hidden':hidden}}

@dataclass
class Config(LinuxTaskConfig):
    VARIANT_NAME:str='base'; DOMAIN_NAME:str='physical_sciences'; TASK_NAME:str='phys_paper_30c3c559c4ab8bd3_imaging_the_seismic_velocity_structure_of_the_cr'
    def to_metadata(self):
        m=dict(super().to_metadata()); m['task_id']=f'{self.DOMAIN_NAME}/{self.TASK_NAME}'; return m
    @property
    def task_description(self):
        return f"""Implement the period-dependent, covariance-weighted iterative Rayleigh-wave tomography proxy in `{self.input_dir}/spec.md`. Read the complete spec and visible JSON. Write reusable `{self.remote_output_dir}/solve.py` accepting `--input PATH --output DIRECTORY --diagnostics PATH`; execute it on the visible input. The evaluator reruns the same program on hidden period-specific kernels, changed dimensions, covariance, and damping. Use only Python and NumPy, derive every shape from JSON, and emit exactly the required NPZ arrays and diagnostics."""
@cb.tasks_config(split='train')
def load():
    c=Config(); return [cb.Task(description=c.task_description,metadata=c.to_metadata(),computer={'provider':'computer','setup_config':{'os_type':'linux'}})]
@cb.setup_task(split='train')
async def start(task_cfg,session):
    out=(task_cfg.metadata or {}).get('remote_output_dir','output'); await session.run_command(f'mkdir -p {shlex.quote(out)}',check=False)
async def _run(session,cmd,timeout=30):
    r=await session.run_command(cmd,timeout=timeout,check=False); rc=r.get('returncode',0) if isinstance(r,dict) else getattr(r,'returncode',0)
    if rc!=0: raise RuntimeError(f'command failed: {rc}')

async def _run_candidate_isolated(session, solve_text, input_text, stage):
    """Run an untrusted submission in a private mount namespace.

    bwrap mounts only the staged solve.py/input.json directory at /work,
    with read-only system libraries and private /proc,/tmp.  In particular,
    the evaluator's reference directory and hidden fixtures are not visible
    or writable during candidate execution.  The stage is recreated for every
    visible/hidden run and removed after artifacts are copied out.
    """
    qstage=shlex.quote(stage)
    await _run(session, f'rm -rf {qstage} && mkdir -m 700 {qstage}', 30)
    await session.write_file(f'{stage}/solve.py', solve_text)
    await session.write_file(f'{stage}/input.json', input_text)
    sandbox=(
        f"command -v bwrap >/dev/null 2>&1 || exit 127; "
        f"mkdir -m 700 {qstage}/run && "
        f"bwrap --die-with-parent --unshare-all "
        f"--ro-bind /usr /usr --ro-bind /lib /lib --ro-bind /lib64 /lib64 "
        f"--ro-bind /bin /bin --ro-bind /sbin /sbin --ro-bind /etc /etc "
        f"--dev /dev --proc /proc --tmpfs /tmp --bind {qstage} /work --chdir /work "
        f"/usr/bin/python3 /work/solve.py --input /work/input.json "
        f"--output /work/run --diagnostics /work/run/diagnostics.json"
    )
    await _run(session, sandbox, 30)
    for name in ('result.npz','diagnostics.json'):
        if not await session.file_exists(f'{stage}/run/{name}'):
            raise RuntimeError(f'missing candidate artifact: {name}')

async def _copy_stage_artifacts(session, stage, local):
    for name in ('result.npz','diagnostics.json'):
        (local/name).write_bytes(await session.read_bytes(f'{stage}/run/{name}'))
@cb.evaluate_task(split='train')
async def evaluate(task_cfg,session):
    try:
        meta=task_cfg.metadata; out=meta['remote_output_dir']; inp=meta['input_dir']; refdir=meta['reference_dir']
        if not await session.file_exists(f'{out}/solve.py'): return [0.0]
        local=Path(tempfile.mkdtemp()); solve_text=await session.read_file(f'{out}/solve.py')
        vis_text=await session.read_file(f'{inp}/tomography_input.json'); vis=json.loads(vis_text)
        # Reference code is imported and executed only in the evaluator process;
        # it is never copied into the candidate VM or its output directory.
        oracle_path=Path(__file__).resolve().parent/'scripts'/'oracle.py'
        spec=importlib.util.spec_from_file_location('_evaluator_private_tomography_oracle',oracle_path)
        if spec is None or spec.loader is None: raise RuntimeError('oracle import failed')
        oracle_mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle_mod)
        stage=f'/tmp/ale-tomography-{uuid.uuid4().hex}'
        await _run_candidate_isolated(session,solve_text,vis_text,stage)
        await _copy_stage_artifacts(session,stage,local)
        await _run(session,f'rm -rf {shlex.quote(stage)}',30)
        # Compute the visible reference only after the candidate has exited.
        ref=oracle_mod.solve_reference(vis)
        scores=[]
        for name in ('hidden_case.json','hidden_dimensions_case.json'):
            hidden_text=await session.read_file(f'{refdir}/{name}'); hid=json.loads(hidden_text)
            hidden_stage=f'/tmp/ale-tomography-{uuid.uuid4().hex}'
            await _run_candidate_isolated(session,solve_text,hidden_text,hidden_stage)
            (local/'hidden_result.npz').write_bytes(await session.read_bytes(f'{hidden_stage}/run/result.npz'))
            (local/'hidden_diagnostics.json').write_bytes(await session.read_bytes(f'{hidden_stage}/run/diagnostics.json'))
            await _run(session,f'rm -rf {shlex.quote(hidden_stage)}',30)
            # Hidden reference generation is likewise evaluator-private and
            # deliberately follows the candidate process.
            href=oracle_mod.solve_reference(hid)
            scores.append(score_dir(local,ref,href,np.asarray(vis['observed_slowness_s_per_km']),np.asarray(hid['observed_slowness_s_per_km']))['score'])
        return [float(np.mean(scores))]
    except Exception: return [0.0]
