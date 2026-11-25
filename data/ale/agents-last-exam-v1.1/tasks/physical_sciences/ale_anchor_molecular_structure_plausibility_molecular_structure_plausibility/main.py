from __future__ import annotations
import importlib.util
import json
import sys
import tempfile
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
  task_dir='base'; input_dir='base/input'; software_dir='base/software'; remote_output_dir='base/output'; reference_dir='base/reference'
  def to_metadata(self): return {'task_dir':self.task_dir,'input_dir':self.input_dir,'software_dir':self.software_dir,'remote_output_dir':self.remote_output_dir,'reference_dir':self.reference_dir}
 class BaseTaskSetup:
  async def __call__(self,*a,**k): return None
 _setup=BaseTaskSetup()
ROOT=Path(__file__).resolve().parent

def _load_script(name):
 module_name=f'{__name__}._{name}'
 spec=importlib.util.spec_from_file_location(module_name,ROOT/'scripts'/f'{name}.py')
 if spec is None or spec.loader is None: raise ImportError(f'cannot load task script: {name}')
 module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module;spec.loader.exec_module(module);return module

@dataclass
class Config(LinuxTaskConfig):
 DOMAIN_NAME: str='physical_sciences'; TASK_NAME: str='ale_anchor_molecular_structure_plausibility_molecular_structure_plausibility'; VARIANT_NAME: str='base'
 def to_metadata(self):
  d=super().to_metadata(); d.update({'task_id':f'{self.DOMAIN_NAME}/{self.TASK_NAME}','input_files':['task_brief.md','xyz_files'],'output_dir':self.remote_output_dir}); return d
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
 m=task_cfg.metadata
 try:
  with tempfile.TemporaryDirectory(prefix='ale_msp_eval_') as td:
   root=Path(td); out=root/'output'; ref=root/'reference'; vis=root/'visible'
   for p in (out,ref,vis,ref/'hidden_xyz'): p.mkdir(parents=True,exist_ok=True)
   for name in ('decisions.json','summary.json','analyzer.py'):
    if await session.file_exists(f"{m['output_dir']}/{name}"): (out/name).write_bytes(await session.read_bytes(f"{m['output_dir']}/{name}"))
   (ref/'expected.json').write_bytes(await session.read_bytes(f"{m['reference_dir']}/expected.json"))
   # Hidden fixtures are copied only at evaluation time.
   for n in await session.list_dir(f"{m['reference_dir']}/hidden_xyz"): (ref/'hidden_xyz'/n).write_bytes(await session.read_bytes(f"{m['reference_dir']}/hidden_xyz/{n}"))
   score=_load_script('score').score
   return [float(score(out,ref,vis).get('score',0.0))]
 except Exception: return [0.0]
