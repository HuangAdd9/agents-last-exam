from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
try:
 import cua_bench as cb
 from tasks.common_setup import BaseTaskSetup
 from tasks.linux_runtime import LinuxTaskConfig
 _setup=BaseTaskSetup()
except ImportError:
 class _CB:
  class DesktopSession: pass
  class Task:
   def __init__(self,**kw): self.__dict__.update(kw)
  @staticmethod
  def tasks_config(split='train'): return lambda f:f
  @staticmethod
  def setup_task(split='train'): return lambda f:f
  @staticmethod
  def evaluate_task(split='train'): return lambda f:f
 cb=_CB()
 class LinuxTaskConfig:
  OS_TYPE='linux'; task_dir='base'; input_dir='base/input'; software_dir='base/software'; remote_output_dir='base/output'; reference_dir='base/reference'
  def to_metadata(self): return {'task_dir':self.task_dir,'input_dir':self.input_dir,'software_dir':self.software_dir,'remote_output_dir':self.remote_output_dir,'reference_dir':self.reference_dir}
 class BaseTaskSetup:
  async def __call__(self,*a,**k): return None
 _setup=BaseTaskSetup()
ROOT=Path(__file__).resolve().parent
TASK_NAME='ale_anchor_exact_diag_heisenberg_j1j2_exact_diagonalization_of_the_j1_j2_heisenberg_mo'
INPUT_FILES=('problem_spec.md','model_config.json')
@dataclass
class Config(LinuxTaskConfig):
 DOMAIN_NAME: str='physical_sciences'; TASK_NAME: str=TASK_NAME; VARIANT_NAME: str='base'
 def to_metadata(self):
  d=super().to_metadata() if hasattr(super(),'to_metadata') else {}; d.update({'task_id':f'physical_sciences/{TASK_NAME}','input_files':list(INPUT_FILES),'python_entry':f'{self.software_dir}/python','eval_tmp_dir':f'{self.task_dir}/_eval_tmp'}); return d
 @property
 def task_description(self):
     return json.loads(Path(__file__).with_name("task_card.json").read_text(encoding="utf-8"))["taskPrompt"]
config=Config()
@cb.tasks_config(split='train')
def load(): return [cb.Task(description=config.task_description,metadata=config.to_metadata(),computer={'provider':'computer','setup_config':{'os_type':'linux'}})]
@cb.setup_task(split='train')
async def start(task_cfg, session):
    await _setup(task_cfg, session)
    await session.interface.create_dir(task_cfg.metadata["remote_output_dir"])
@cb.evaluate_task(split='train')
async def evaluate(task_cfg,session):
 m=task_cfg.metadata; td=m.get('eval_tmp_dir',f"{m['task_dir']}/_eval_tmp"); await session.interface.create_dir(td); verifier=f'{td}/score_outputs.py'; await session.write_file(verifier,(ROOT/'scripts/score_outputs.py').read_text())
 r=await session.run_command(f'"{m["python_entry"]}" "{verifier}" --output-dir "{m["remote_output_dir"]}" --reference-dir "{m["reference_dir"]}"',check=False)
 if not isinstance(r,dict) or r.get('return_code',1)!=0:return [0.0]
 for line in reversed(str(r.get('stdout','')).splitlines()):
  try:return [float(json.loads(line).get('score',0.0))]
  except Exception: pass
 return [0.0]
