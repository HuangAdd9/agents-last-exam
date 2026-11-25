"""ALE task: offline 7YY structure reconstruction with hidden SVG mutation."""
from __future__ import annotations
import json, shlex, subprocess, sys, tempfile
from dataclasses import dataclass
from pathlib import Path
try:
 import cua_bench as cb
 from tasks.common_setup import BaseTaskSetup
 from tasks.linux_runtime import LinuxTaskConfig
 _setup=BaseTaskSetup()
except ImportError:
 class _CB:
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
  task_dir='.'; input_dir='input'; software_dir='software'; remote_output_dir='output'; reference_dir='reference'
  def to_metadata(self): return self.__dict__.copy()
 class BaseTaskSetup:
  async def __call__(self,*a,**k): return None
 _setup=BaseTaskSetup()
ROOT=Path(__file__).resolve().parent
@dataclass
class Config(LinuxTaskConfig):
 DOMAIN_NAME: str='physical_sciences'; TASK_NAME: str='ale_anchor_ketcher_smiles_reproduction_ketcher_smiles_reproduction'; VARIANT_NAME: str='7yy'
 def to_metadata(self):
  d=super().to_metadata(); d.update({'task_id':'physical_sciences/'+self.TASK_NAME,'eval_tmp_dir':f'{self.task_dir}/_eval_tmp'}); return d
 @property
 def task_description(self):
     return json.loads(Path(__file__).with_name("task_card.json").read_text(encoding="utf-8"))["taskPrompt"]
config=Config()
@cb.tasks_config(split='train')
def load():
 return [cb.Task(description=config.task_description,metadata=config.to_metadata(),computer={'provider':'computer','setup_config':{'os_type':'linux'}})]
@cb.setup_task(split='train')
async def start(task_cfg, session):
    await _setup(task_cfg, session)
    await session.interface.create_dir(task_cfg.metadata["remote_output_dir"])
@cb.evaluate_task(split='train')
async def evaluate(task_cfg,session):
 m=task_cfg.metadata
 out=m['remote_output_dir']; ref=m['reference_dir']; inp=m['input_dir']
 try:
  with tempfile.TemporaryDirectory(prefix='ale_7yy_eval_') as td:
   root=Path(td); local_out=root/'output'; local_ref=root/'reference'; hidden=local_ref/'hidden_input'
   local_out.mkdir(); hidden.mkdir(parents=True); local_input=root/'input'; local_input.mkdir()
   for name in ('submission.json','atom_inventory.json','solve.py'):
    remote=f'{out}/{name}'
    if not await session.file_exists(remote): return [0.0]
    (local_out/name).write_bytes(await session.read_bytes(remote))
   for name in ('reference.json','hidden_expected.json'):
    (local_ref/name).write_bytes(await session.read_bytes(f'{ref}/{name}'))
   (hidden/'structure.svg').write_bytes(await session.read_bytes(f'{ref}/hidden_input/structure.svg'))
   (local_input/'structure.svg').write_bytes(await session.read_bytes(f'{inp}/structure.svg'))
   r=subprocess.run([sys.executable,str(ROOT/'scripts/verify_submission.py'),'--output',str(local_out),'--reference',str(local_ref/'reference.json'),'--svg',str(local_input/'structure.svg'),'--hidden',str(hidden/'structure.svg')],capture_output=True,text=True,timeout=15,check=False)
   for line in reversed(r.stdout.splitlines()):
    try: return [float(json.loads(line).get('score',0.0))]
    except Exception: pass
 except Exception:
  return [0.0]
 return [0.0]
