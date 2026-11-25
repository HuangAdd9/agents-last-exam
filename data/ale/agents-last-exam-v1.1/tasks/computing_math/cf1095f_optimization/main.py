"""ALE native Linux task: optimize exact prefix connection costs."""
from dataclasses import dataclass
import asyncio
import json
import logging
import math
from pathlib import Path
import shlex

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

logger=logging.getLogger(__name__)


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str='computing_math'
    TASK_NAME: str='cf1095f_optimization'
    VARIANT_NAME: str='base'

    @property
    def task_description(self):
        return f'''Optimize the connection replay service without changing its correct results.
The handover starts at {self.input_dir}/README.md; resolve the deployed contract
from the visible release records, not from file timestamps or draft documents.
Your working baseline is {self.remote_output_dir}/solution.cpp. Submit this single
C++17 source; internal implementation may be replaced. Deployment quotas and
execution-based scoring are in the handover. Run bash run_tests.sh from the output
directory for public regressions. Native Linux CPU CLI; no desktop is required.'''

    def to_metadata(self):
        meta=super().to_metadata()
        meta['output_file']=f'{self.remote_output_dir}/solution.cpp'
        meta['task_id']='computing_math/cf1095f_optimization'
        meta['task_version']=6
        meta['contract_release']='R52'
        return meta


class CleanSetup(BaseTaskSetup):
    async def setup(self,task_cfg,session):
        m=task_cfg.metadata
        for path in (f"{m['input_dir']}/README.md",f"{m['input_dir']}/solution.cpp",
                     f"{m['software_dir']}/run_tests.sh",
                     *(f"{m['input_dir']}/{p}" for p in (
                         'ops/selected.ini','ops/quota.txt','handover/export.csv',
                         'archive/contract.md','archive/next.txt','transport/reader.notes',
                         'tickets/settlement.txt','tickets/capacity.txt'))):
            if not await session.file_exists(path):
                raise RuntimeError(f'Public task file missing: {path}')
        # Only these two exact task-local directories are reset on a reused sandbox.
        for key in ('remote_output_dir','reference_dir'):
            path=Path(m[key])
            if path.parent!=Path(m['task_dir']) or path.name in ('','input','software'):
                raise RuntimeError('Invalid task-local reset path')
        out=shlex.quote(m['remote_output_dir']); ref=shlex.quote(m['reference_dir'])
        source=shlex.quote(f"{m['input_dir']}/solution.cpp")
        helper=shlex.quote(f"{m['software_dir']}/run_tests.sh")
        result=await session.run_command(
            f'rm -rf -- {out} {ref} && mkdir -p -- {out} && cp -- {source} {out}/solution.cpp && cp -- {helper} {out}/run_tests.sh',
            check=False)
        if result.get('return_code')!=0:
            raise RuntimeError(f'Workspace initialization failed: {result}')
        if await session.directory_exists(m['reference_dir']) or await session.file_exists(m['reference_dir']):
            raise RuntimeError('Private reference visible at start')


config=TaskConfig()
_setup=CleanSetup()


@cb.tasks_config(split='train')
def load():
    cfg=TaskConfig()
    return [cb.Task(description=cfg.task_description,metadata=cfg.to_metadata(),
                    computer={'provider':'computer','setup_config':{'os_type':'linux'}})]


@cb.setup_task(split='train')
async def start(task_cfg,session: cb.DesktopSession):
    await _setup(task_cfg,session)


@cb.evaluate_task(split='train')
async def evaluate(task_cfg,session: cb.DesktopSession)->list[float]:
    m=task_cfg.metadata
    for name in ('judge.py','manifest.json'):
        if not await session.file_exists(f"{m['reference_dir']}/{name}"):
            raise RuntimeError(f'Private grading file missing: {name}')
    wrapper=(Path(__file__).parent/'scripts/run_judge.sh').read_text()
    result=await session.run_command('mktemp -d /tmp/cf1095f-eval.XXXXXX',check=False)
    if result.get('return_code')!=0:
        raise RuntimeError(f'Cannot allocate evaluation workspace: {result}')
    work=result['stdout'].strip()
    if not work.startswith('/tmp/cf1095f-eval.') or '\n' in work:
        raise RuntimeError('Invalid evaluation workspace response')
    await session.write_file(f'{work}/run.sh',wrapper)
    command=' '.join(shlex.quote(s) for s in ['bash',f'{work}/run.sh',m['output_file'],m['reference_dir'],work])
    result=await session.run_command(f'nohup {command} > {shlex.quote(work)}/launcher.log 2>&1 < /dev/null &',check=False)
    if result.get('return_code')!=0:
        raise RuntimeError(f'Cannot launch evaluator: {result}')
    for _ in range(360):
        if await session.file_exists(f'{work}/rc'): break
        await asyncio.sleep(2)
    else:
        raise RuntimeError('Evaluator infrastructure exceeded 720 seconds')
    rc=(await session.read_file(f'{work}/rc')).strip()
    if rc!='0':
        diagnostic=await session.read_file(f'{work}/stderr')
        raise RuntimeError(f'Evaluator infrastructure rc={rc}: {diagnostic}')
    try:
        report=json.loads(await session.read_file(f'{work}/report.json'))
        score=float(report['score'])
        assert math.isfinite(score) and 0<=score<=1
    except (ValueError,KeyError,AssertionError,TypeError) as exc:
        raise RuntimeError('Malformed evaluator report') from exc
    await session.write_file(f"{m['remote_output_dir']}/evaluation_report.json",json.dumps(report,indent=2))
    logger.info('CF1095F evaluation: %s',json.dumps(report))
    return [score]
