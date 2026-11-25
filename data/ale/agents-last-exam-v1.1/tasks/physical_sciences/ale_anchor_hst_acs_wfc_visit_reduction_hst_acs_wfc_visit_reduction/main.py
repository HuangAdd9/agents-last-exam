"""ALE hooks for the benchmark-authored ACS/WFC reduction task."""
from __future__ import annotations
import json
import logging
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
   def __init__(self,**kw):self.__dict__.update(kw)
  @staticmethod
  def tasks_config(split='train'):return lambda f:f
  @staticmethod
  def setup_task(split='train'):return lambda f:f
  @staticmethod
  def evaluate_task(split='train'):return lambda f:f
 cb=_CB()
 class LinuxTaskConfig:
  task_dir='.';input_dir='input';software_dir='software';remote_output_dir='output';reference_dir='reference'
  def to_metadata(self):return self.__dict__.copy()
 class BaseTaskSetup:
  async def __call__(self,*a,**k):return None
 _setup=BaseTaskSetup()
ROOT=Path(__file__).resolve().parent; TASK='ale_anchor_hst_acs_wfc_visit_reduction_hst_acs_wfc_visit_reduction'
logger = logging.getLogger(__name__)
@dataclass
class Config(LinuxTaskConfig):
 DOMAIN_NAME: str='physical_sciences';TASK_NAME: str=TASK;VARIANT_NAME: str='base'
 def to_metadata(self):
  d=super().to_metadata();d.update({'task_id':f'physical_sciences/{TASK}','eval_tmp_dir':f'{self.task_dir}/_eval_tmp'});return d
 @property
 def task_description(self):
     return json.loads(Path(__file__).with_name("task_card.json").read_text(encoding="utf-8"))["taskPrompt"]
config=Config()
@cb.tasks_config(split='train')
def load():return [cb.Task(description=config.task_description,metadata=config.to_metadata(),computer={'provider':'computer','setup_config':{'os_type':'linux'}})]
@cb.setup_task(split='train')
async def start(task_cfg, session):
    await _setup(task_cfg, session)
    await session.interface.create_dir(task_cfg.metadata["remote_output_dir"])
@cb.evaluate_task(split='train')
async def evaluate(task_cfg,session):
 m = task_cfg.metadata
 tmp = m.get('eval_tmp_dir', f"{m['task_dir']}/_eval_tmp")
 cmd = (
  f"rm -rf {tmp}; mkdir -p {tmp}/input {tmp}/output; "
  f"cp -a {m['input_dir']}/visit_bluefield {tmp}/input/; "
  f"cp -a {m['reference_dir']}/hidden_input/visit_redfield {tmp}/input/; "
  f"python3 {m['remote_output_dir']}/reduce_visit.py --input {tmp}/input --output {tmp}/output"
 )
 result = await session.run_command(cmd, check=False)
 if result.get('return_code', result.get('returncode', 1)) != 0:
  logger.warning('HST reduction failed: %s', result.get('stderr', ''))
  return [0.0]

 await session.write_file(f'{tmp}/score_outputs.py', (ROOT/'scripts/score_outputs.py').read_text())
 score_cmd = (
  f"python3 {tmp}/score_outputs.py --output-dir {tmp}/output "
  f"--reference-dir {m['reference_dir']}/reference_outputs"
 )
 result = await session.run_command(score_cmd, check=False)
 if result.get('return_code', result.get('returncode', 1)) != 0:
  logger.warning('HST scoring failed: %s', result.get('stderr', ''))
  return [0.0]
 stdout = result.get('stdout', result.get('output', ''))
 if isinstance(stdout, bytes):
  stdout = stdout.decode('utf-8', errors='replace')
 for line in reversed(str(stdout).splitlines()):
  try:
   return [float(json.loads(line)['score'])]
  except (ValueError, KeyError, TypeError):
   continue
 logger.warning('HST scoring returned no score: %s', str(stdout)[-500:])
 return [0.0]
