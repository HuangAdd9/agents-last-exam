"""ALE hooks for the Jimmu Guyot reduced straight-ray tomography task."""
from __future__ import annotations
import asyncio, importlib.util, json, logging, os, shutil, stat, subprocess, sys, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cua_bench as cb
    from tasks.common_setup import BaseTaskSetup
    from tasks.linux_runtime import LinuxTaskConfig
except Exception:  # importable in the authoring checkout too
    cb=None
    class LinuxTaskConfig:
        REMOTE_ROOT_DIR="/media/user/data/agenthle"; DOMAIN_NAME=""; TASK_NAME=""; VARIANT_NAME="base"; REMOTE_OUTPUT_DIR="output"
        @property
        def task_dir(self): return f"{self.REMOTE_ROOT_DIR}/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}"
        @property
        def input_dir(self): return self.task_dir+"/input"
        @property
        def reference_dir(self): return self.task_dir+"/reference"
        @property
        def remote_output_dir(self): return self.task_dir+"/output"
    class BaseTaskSetup:
        async def __call__(self,*a,**k): return None

ROOT=Path(__file__).resolve().parent; SCRIPTS=ROOT/"scripts"; log=logging.getLogger(__name__)
spec=importlib.util.spec_from_file_location("jimmu_verify",SCRIPTS/"verify_outputs.py"); verify=importlib.util.module_from_spec(spec); spec.loader.exec_module(verify)
oracle_spec=importlib.util.spec_from_file_location("jimmu_oracle",SCRIPTS/"oracle.py"); oracle=importlib.util.module_from_spec(oracle_spec); oracle_spec.loader.exec_module(oracle)
TASK_NAME="phys_paper_08211215490f0dca_a_seismic_tomography_gravity_and_flexure_study_o"; VARIANT_NAME="base"

@dataclass
class JimmuConfig(LinuxTaskConfig):
    DOMAIN_NAME: str="physical_sciences"; TASK_NAME: str=TASK_NAME; VARIANT_NAME: str=VARIANT_NAME
    @property
    def visible_case(self): return f"{self.input_dir}/tomography_case.json"
    @property
    def solver_output(self): return f"{self.remote_output_dir}/model.json"
    def to_metadata(self):
        try: m=super().to_metadata()
        except Exception: m={}
        m.update({"task_id":f"{self.DOMAIN_NAME}/{self.TASK_NAME}","input_dir":self.input_dir,"reference_dir":self.reference_dir,"remote_output_dir":self.remote_output_dir,"visible_case":self.visible_case,"solver_output":self.solver_output,"hidden_case":f"{self.reference_dir}/hidden_case.json"})
        return m

config=JimmuConfig()

def _run_isolated(solver: Path, case: Path, output: Path, timeout: int = 30):
    """Run an untrusted solver in a mount/network/pid-isolated bubblewrap jail.

    Only the solver, the current case, and a writable output directory are
    mounted at /work.  In particular, the evaluator package and private oracle
    never exist in the candidate's namespace.
    """
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap is required for isolated evaluation")
    output.mkdir(parents=True, exist_ok=True)
    # The declared cpu-free-ubuntu runtime does not grant a network namespace;
    # unshare only the namespaces available there and enforce the no-network
    # policy by mounting a minimal filesystem and supplying no credentials.
    cmd = [bwrap, "--unshare-pid", "--unshare-uts", "--unshare-ipc", "--unshare-cgroup", "--die-with-parent", "--new-session",
           "--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin",
           "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
           "--ro-bind", "/etc", "/etc", "--proc", "/proc", "--dev", "/dev",
           "--tmpfs", "/tmp", "--ro-bind", str(solver), "/work/tomography_solver.py",
           "--ro-bind", str(case), "/work/CASE.json", "--bind", str(output), "/work/out",
           "--chdir", "/work", sys.executable, "/work/tomography_solver.py",
           "--input", "/work/CASE.json", "--output", "/work/out"]
    env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "PYTHONNOUSERSITE": "1",
           "HOME": "/tmp", "PYTHONHASHSEED": "0"}
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)

def _description():
    return f"""You are a geophysicist reducing a compact active-source P-wave tomography benchmark inspired by the Jimmu Guyot OBS line. The staged case is {config.visible_case}. Read the companion input/problem_spec.md: it defines the weighted straight-ray forward model, ridge objective, row-major indexing, deterministic rectangular segment clipping, tolerances, and auditable iterative trace contract. The visible fixture is a reduced surrogate for the paper's P-wave observable; hidden cases change endpoints, crossing-cell assignments, ray count, uncertainties, and grid dimensions.\n\nDerive every ray's cell lengths from its boundary endpoints by clipping against grid lines and independently check the staged vector before inversion. Implement the specified diagonally preconditioned steepest-descent method with exact quadratic line search (at least one non-zero update; do not print a fabricated table) to recover cell slowness. Write ONLY under {config.remote_output_dir}: model.json, trace.csv, report.json, and a reusable tomography_solver.py. The solver must accept `python tomography_solver.py --input CASE.json --output DIR` and work on an evaluator-only changed case.\n\nmodel.json exact keys: grid, velocity_km_s, slowness_s_per_km, rms_s, ray_predictions_s, with dimensions taken from CASE and row-major flattening. Use km/s, s/km, seconds. trace.csv headers exactly iteration,objective,rms_s,step,s_0,...,s_{{n-1}}; every row must contain its intermediate row-major slowness state. The grader recomputes geometry, objective/RMS, and the prescribed gradient-line-search update from each state, checks Euclidean state-to-state steps, and requires the final state/objective to match the submitted model. report.json keys exactly method,units,case_id,invariant; set invariant to forward_time_sum_matches_reported_rms only after independently checking predictions. Use each provided uncertainty_s as a weight, enforce positive velocities, and include the specified ridge prior. Runtime target is under 30 s and no network or undeclared packages may be used.\n"""

if cb:
    @cb.tasks_config(split="train")
    def load():
        return [cb.Task(description=_description(),metadata=config.to_metadata(),computer={"provider":"computer","setup_config":{"os_type":"linux"}})]
    @cb.setup_task(split="train")
    async def start(task_cfg,session): await BaseTaskSetup()(task_cfg,session)
    async def _pull(session,remote,local,limit=verify.MAX_ARTIFACT_BYTES):
        local.parent.mkdir(parents=True,exist_ok=True)
        # Use the runtime's metadata primitive when available so symlinks and large
        # remote files are rejected before their contents are fetched.
        for meth in ("lstat", "stat"):
            probe=getattr(session,meth,None)
            if probe is not None:
                info=await probe(remote)
                size=info.get("size") if isinstance(info,dict) else getattr(info,"size",None)
                mode=info.get("mode") if isinstance(info,dict) else getattr(info,"mode",None)
                if size is not None and int(size)>limit: raise ValueError("remote artifact too large")
                if mode is not None and not stat.S_ISREG(int(mode)): raise ValueError("remote artifact is not regular")
                break
        blob=await session.read_bytes(remote)
        if len(blob)>limit: raise ValueError("remote artifact too large")
        local.write_bytes(blob)
        verify._regular(local,limit)
    @cb.evaluate_task(split="train")
    async def evaluate(task_cfg,session):
        meta=task_cfg.metadata
        with tempfile.TemporaryDirectory(prefix="jimmu_eval_") as td:
            root=Path(td); out=root/"output"; ref=root/"reference"; out.mkdir(); ref.mkdir()
            try:
                for name in ("model.json","trace.csv","report.json","tomography_solver.py"): await _pull(session,f"{meta['remote_output_dir']}/{name}",out/name)
                for name,remote in (("visible_case.json",meta["visible_case"]),("hidden_case.json",meta["hidden_case"])):
                    await _pull(session,remote,ref/name)
                # Candidate-side hidden input deliberately omits the fixture's
                # sensitivity rows.  Geometry must be reconstructed from the
                # boundary endpoints; the private oracle retains its full case.
                hidden=json.loads((ref/"hidden_case.json").read_text())
                for ray in hidden.get("rays", []): ray.pop("path_lengths_km", None)
                (ref/"hidden_candidate_case.json").write_text(json.dumps(hidden))
            except Exception: return [0.0]
            # Keep oracle models in evaluator memory; never place a hidden answer in
            # the candidate-readable temporary directory.
            visible_ref=oracle.reference(str(ref/"visible_case.json"))
            hidden_ref=oracle.reference(str(ref/"hidden_case.json"))
            base=verify.score_dir(out,ref/"visible_case.json",visible_ref)
            hidden_out=root/"hidden_out"; hidden_out.mkdir()
            try:
                p=_run_isolated(out/"tomography_solver.py", ref/"hidden_candidate_case.json", hidden_out, timeout=30)
                if p.returncode!=0: return [float(base["score"])]
                if not verify.validate_run(hidden_out,ref/"hidden_candidate_case.json"): return [float(base["score"])]
                hs=verify.score_hidden(hidden_out,hidden_ref,ref/"hidden_candidate_case.json")
            except Exception: hs={"score":0.0}
            return [float(max(0.0,min(1.0,base["score"]+hs["score"]))) ]
else:
    def load(): return []
    async def start(task_cfg,session): return None
    async def evaluate(task_cfg,session): return [0.0]
