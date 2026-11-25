"""Active-source seismic travel-time inversion benchmark."""
from __future__ import annotations
import importlib.util, logging, sys, tempfile, shlex, stat
from dataclasses import dataclass
from pathlib import Path
import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

_setup=BaseTaskSetup(); logger=logging.getLogger(__name__)
if __name__ not in sys.modules:
    sys.modules[__name__]=sys.modules.get(__name__,type(sys)(__name__))
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"scripts"))
spec=importlib.util.spec_from_file_location("jdf_verify",ROOT/"scripts/verify_outputs.py")
VERIFY=importlib.util.module_from_spec(spec); sys.modules[spec.name]=VERIFY; spec.loader.exec_module(VERIFY)

@dataclass
class JDFConfig(LinuxTaskConfig):
    DOMAIN_NAME: str="physical_sciences"
    TASK_NAME: str="phys_paper_a74bb1db81dcddab_structure_and_evolution_of_northern_juan_de_fuca"
    VARIANT_NAME: str="base"
    VARIANT_LABEL: str="Layered P-wave refraction inversion"
    @property
    def problem_spec(self)->str:return f"{self.input_dir}/problem_spec.md"
    @property
    def picks_file(self)->str:return f"{self.input_dir}/picks.csv"
    @property
    def output_dir(self)->str:return self.remote_output_dir
    @property
    def task_description(self)->str:
        return f'''You are working on a Linux VM with no network access. Reconstruct the 2-D four-cell P-wave velocity field from active-source first-arrival rays, following the ray-path equations and prescribed regularized damped Gauss–Newton inversion in `{self.problem_spec}`. Read `{self.picks_file}` and write exactly `model.json`, `predictions.csv`, `inversion_trace.csv`, and `diagnostics.json` under `{self.output_dir}`. Use kilometres, km/s, and seconds; compute geometric ray lengths from each source/receiver pair, apply sigma weighting, and record the fixed initial state plus every prescribed refinement state. Hidden rays have changed observations, geometry, and heteroscedastic uncertainties and are inverted independently.'''
    def to_metadata(self):
        m=super().to_metadata(); m.update({"task_id":f"{self.DOMAIN_NAME}/{self.TASK_NAME}","evaluator_reference_dir":m.get("reference_dir"),"problem_spec":self.problem_spec,"picks_file":self.picks_file,"required_output_files":["model.json","predictions.csv","inversion_trace.csv","diagnostics.json"]}); return m

CONFIG=JDFConfig()

@cb.tasks_config(split="train")
def load():
    return [cb.Task(description=CONFIG.task_description,metadata=CONFIG.to_metadata(),computer={"provider":"computer","setup_config":{"os_type":"linux"}})]

@cb.setup_task(split="train")
async def start(task_cfg,session:cb.DesktopSession):
    await _setup(task_cfg,session)
    await session.interface.create_dir(task_cfg.metadata["remote_output_dir"])

@cb.evaluate_task(split="train")
async def evaluate(task_cfg,session:cb.DesktopSession)->list[float]:
    meta=task_cfg.metadata; names=meta["required_output_files"]
    limits={"model.json":64*1024,"predictions.csv":2*1024*1024,"inversion_trace.csv":2*1024*1024,"diagnostics.json":64*1024}
    async def pull(remote: str, cap: int) -> bytes:
        q=shlex.quote(remote)
        probe=await session.run_command(f"/usr/bin/python3 -c 'import os,stat,sys; s=os.lstat(sys.argv[1]); print(int(s.st_mode),int(s.st_size))' {q}",timeout=10,check=False)
        rc=probe.get("returncode",probe.get("return_code",0)) if isinstance(probe,dict) else getattr(probe,"returncode",getattr(probe,"return_code",0))
        text=probe.get("stdout",probe.get("output","")) if isinstance(probe,dict) else getattr(probe,"stdout",getattr(probe,"output",""))
        try: mode_s,size_s=str(text).strip().split()[-2:]; mode,size=int(mode_s),int(size_s)
        except Exception: raise RuntimeError("invalid lstat")
        if rc!=0 or not stat.S_ISREG(mode) or size<0 or size>cap: raise RuntimeError("unsafe artifact")
        blob=await session.read_bytes(remote)
        if len(blob)!=size or len(blob)>cap: raise RuntimeError("bounded read failed")
        return blob
    with tempfile.TemporaryDirectory(prefix="jdf_eval_") as td:
        out=Path(td)/"output"; ref=Path(td)/"reference"; out.mkdir(); ref.mkdir()
        for n in names:
            op=f'{meta["remote_output_dir"]}/{n}'
            if not await session.file_exists(op): return [0.0]
            try: (out/n).write_bytes(await pull(op,limits[n]))
            except Exception: return [0.0]
        refdir=meta.get("evaluator_reference_dir",meta.get("reference_dir"))
        rp=f'{refdir}/reference.json'; hp=f'{refdir}/hidden_picks.csv'
        if not await session.file_exists(rp) or not await session.file_exists(hp): return [0.0]
        (ref/"reference.json").write_bytes(await pull(rp,64*1024)); (ref/"hidden_picks.csv").write_bytes(await pull(hp,2*1024*1024))
        visible=ref/"visible_picks.csv"
        if not await session.file_exists(meta["picks_file"]): return [0.0]
        visible.write_bytes(await pull(meta["picks_file"],2*1024*1024))
        try:
            result=VERIFY.score_submission(out,visible,ref/"hidden_picks.csv",ref/"reference.json")
        except Exception as exc:
            logger.exception("grader failure: %s",exc); return [0.0]
    return [float(result.get("score",0.0))]
