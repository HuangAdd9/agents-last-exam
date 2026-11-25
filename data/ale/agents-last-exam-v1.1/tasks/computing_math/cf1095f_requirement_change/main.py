"""Native Linux CPU task: recover and implement the deployed cost-worker contract."""
import asyncio
from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
import shlex
import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

logger=logging.getLogger(__name__)

async def run_remote(session,command):
    # This CUA version can default a missing transport return_code to zero.
    # Read the actual child status, not that transport fallback.
    wrapper=('import json,subprocess,sys; '
             'p=subprocess.run(sys.argv[1],shell=True,capture_output=True,text=True); '
             'print(json.dumps(dict(return_code=p.returncode,stdout=p.stdout,stderr=p.stderr)))')
    result=await session.run_command(shlex.join(['python3','-c',wrapper,command]),check=False)
    try:
        data=json.loads(result['stdout'])
        if type(data['return_code']) is not int or not isinstance(data['stdout'],str) or not isinstance(data['stderr'],str):
            raise ValueError('Invalid child status')
        return data
    except (ValueError,KeyError,TypeError) as exc:
        raise RuntimeError(f'Remote command status unavailable: {result}') from exc

@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str='computing_math'
    TASK_NAME: str='cf1095f_requirement_change'
    VARIANT_NAME: str='base'

    @property
    def task_description(self):
        return f'''Complete the cost-worker migration using the handoff workspace at
{self.input_dir}. Recover the deployed contract from the visible materials and
deliver C++17 source at {self.remote_output_dir}/solution.cpp.
Run bash run_tests.sh from {self.remote_output_dir} for the available replay checks.
Only the submitted source is evaluated. This is a native Linux CPU CLI task;
no desktop is needed.'''

    def to_metadata(self):
        m=super().to_metadata()
        m.update(task_id='computing_math/cf1095f_requirement_change',task_version=10,
                 output_file=f'{self.remote_output_dir}/solution.cpp')
        return m

class CleanSetup(BaseTaskSetup):
    async def setup(self,task_cfg,session):
        m=task_cfg.metadata
        for name in ('README.md','solution.cpp','checks/replay.json','ops/worker.conf',
                     'queue/spool.notes','queue/consumer-r7.txt','handover/CR-27.txt',
                     'archive/tariff-2018.txt','support/thread-42.txt','archive/batch-v1-export.csv',
                     'deployment/current.env','queue/credit.note','ops/runtime-r8.conf','support/rollout-0914.log'):
            if not await session.file_exists(f"{m['input_dir']}/{name}"):raise RuntimeError('Missing public input: '+name)
        for name in ('run_tests.sh','check_public.py','stream_runtime.py'):
            if not await session.file_exists(f"{m['software_dir']}/{name}"):raise RuntimeError('Missing public helper: '+name)
        for key in ('remote_output_dir','reference_dir'):
            p=Path(m[key])
            if p.parent!=Path(m['task_dir']) or p.name not in ('output','reference'):
                raise RuntimeError('Invalid task-local reset path')
        out=shlex.quote(m['remote_output_dir']);ref=shlex.quote(m['reference_dir'])
        source=shlex.quote(m['input_dir']+'/solution.cpp');helper=shlex.quote(m['software_dir']+'/run_tests.sh')
        obsolete=shlex.quote(m['input_dir']+'/public_cases.json')
        result=await run_remote(session,
            f'command -v g++ && command -v python3 && rm -f -- {obsolete} && rm -rf -- {out} {ref} && mkdir -p {out} && cp {source} {out}/solution.cpp && cp {helper} {out}/run_tests.sh')
        if result.get('return_code')!=0:raise RuntimeError(f'Workspace initialization failed: {result}')
        if await session.directory_exists(m['reference_dir']) or await session.file_exists(m['reference_dir']):
            raise RuntimeError('Private reference visible at start')

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
    for name in ('judge.py','manifest.json','stream_runtime.py'):
        if not await session.file_exists(f"{m['reference_dir']}/{name}"):raise RuntimeError('Missing private grading file: '+name)
    result=await run_remote(session,'mktemp -d /tmp/cf1095f-requirement-eval.XXXXXX')
    if result.get('return_code')!=0:raise RuntimeError(f'Evaluation workspace failed: {result}')
    work=result['stdout'].strip()
    if not work.startswith('/tmp/cf1095f-requirement-eval.') or '\n' in work:raise RuntimeError('Invalid evaluation path')
    await session.write_file(work+'/run.sh',(Path(__file__).parent/'scripts/run_judge.sh').read_text())
    command=shlex.join(['bash',work+'/run.sh',m['output_file'],m['reference_dir'],work])
    result=await run_remote(session,f'nohup {command} > {shlex.quote(work)}/launcher.log 2>&1 < /dev/null &')
    if result.get('return_code')!=0:raise RuntimeError(f'Evaluation launch failed: {result}')
    for _ in range(900):
        if await session.file_exists(work+'/rc'):break
        await asyncio.sleep(2)
    else:raise RuntimeError('Evaluation infrastructure exceeded 1800 seconds')
    if (await session.read_file(work+'/rc')).strip()!='0':
        raise RuntimeError('Evaluation infrastructure failed: '+await session.read_file(work+'/stderr'))
    try:
        report=json.loads(await session.read_file(work+'/report.json'));score=report['score']
        if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1:raise ValueError('Invalid score')
        if report['status'] not in ('judged','missing_source','compile_timeout','compile_error'):raise ValueError('Invalid status')
        if report['status']=='judged':
            weights={'A':10,'B':20,'C':20,'D':25,'E':25}
            if set(report['groups'])!=set(weights):raise ValueError('Missing score groups')
            points=0
            for g,w in weights.items():
                item=report['groups'][g]
                if type(item['passed']) is not bool or item['points']!=(w if item['passed'] else 0):raise ValueError('Group mismatch')
                points+=item['points']
            if score!=points/100:raise ValueError('Score mismatch')
        elif score!=0:raise ValueError('Nonzero invalid submission score')
    except (ValueError,KeyError,TypeError) as exc:raise RuntimeError('Malformed evaluation report') from exc
    await session.write_file(m['remote_output_dir']+'/evaluation_report.json',json.dumps(report,indent=2)+'\n')
    logger.info('CF1095F_REQUIREMENT_CHANGE_REPORT %s',json.dumps(report))
    return [float(score)]
