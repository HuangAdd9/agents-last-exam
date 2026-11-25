"""Selective-SOC perovskite band-gap benchmark."""
from __future__ import annotations
import csv, json, logging, sys, tempfile
from dataclasses import dataclass
from pathlib import Path
import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
try:
    from .scripts.oracle import write_reference
    from .scripts.grader import score_dirs
except ImportError:
    from scripts.oracle import write_reference
    from scripts.grader import score_dirs

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))
logger=logging.getLogger(__name__); _setup=BaseTaskSetup()
VARIANTS=[('base','Selective spin-orbit correction and band-gap screening')]
FILES=('band_gaps.csv','spectral_capture.csv','summary.json','invariants.json')

@dataclass
class Config(LinuxTaskConfig):
    DOMAIN_NAME: str='physical_sciences'; TASK_NAME: str='phys_paper_a93556fb098bc11d_density_functional_theory_estimate_of_halide_per'; VARIANT_NAME: str=''; VARIANT_LABEL: str=''
    @property
    def task_description(self):
        return f'''You are working on a Linux VM. Read {self.input_dir}/problem_spec.md and {self.input_dir}/materials.csv. Implement the specified 9x9x9 k-resolved two-orbital Hamiltonian, diagonalise the Pb-only SOC valence term, extract VBM/CBM, and compute the spectral-capture cross-check. Write exactly band_gaps.csv, spectral_capture.csv, summary.json, and invariants.json under {self.remote_output_dir}. No internet is available; keep all outputs there.'''
    def to_metadata(self):
        m=super().to_metadata(); m.update({'task_id':f'{self.DOMAIN_NAME}/{self.TASK_NAME}','variant_label':self.VARIANT_LABEL,'input_problem_spec':f'{self.input_dir}/problem_spec.md','input_materials':f'{self.input_dir}/materials.csv','required_output_files':list(FILES)}); return m

@cb.tasks_config(split='train')
def load():
    return [cb.Task(description=Config(VARIANT_NAME=n,VARIANT_LABEL=l).task_description, metadata=Config(VARIANT_NAME=n,VARIANT_LABEL=l).to_metadata(), computer={'provider':'computer','setup_config':{'os_type':'linux'}}) for n,l in VARIANTS]

@cb.setup_task(split='train')
async def start(task_cfg, session):
    await _setup(task_cfg,session)
    await session.interface.create_dir(task_cfg.metadata["remote_output_dir"])

@cb.evaluate_task(split='train')
async def evaluate(task_cfg, session):
    meta=task_cfg.metadata; tag=meta.get('variant_name','base')
    with tempfile.TemporaryDirectory(prefix='soc_eval_') as td:
        root=Path(td); out=root/'out'; ref=root/'ref'; out.mkdir(); ref.mkdir()
        for n in FILES:
            try:
                if not await session.file_exists(f'{meta["remote_output_dir"]}/{n}'): return [0.0]
                (out/n).write_bytes(await session.read_bytes(f'{meta["remote_output_dir"]}/{n}'))
            except Exception: return [0.0]
        try:
            hidden=root/'hidden.csv'; hidden.write_bytes(await session.read_bytes(f'{meta["reference_dir"]}/materials_hidden.csv'))
            write_reference(hidden,ref)
            return [float(score_dirs(out,ref)['score'])]
        except Exception:
            logger.exception('evaluation failure for %s',tag); return [0.0]
