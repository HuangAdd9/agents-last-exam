"""ALE native CPU task: receipt replay after the score distribution change."""
import asyncio
from dataclasses import dataclass
import json
import logging
import math
import secrets
from pathlib import Path
import shlex
import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

LOG=logging.getLogger(__name__)
WEIGHTS=dict(A=10,B=15,C=20,D=20,E=20,F=15)

@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str='computing_math'
    TASK_NAME: str='cf1305e_requirement_change'
    VARIANT_NAME: str='base'

    @property
    def task_description(self):
        return f'''接手 score distribution 的回放消费者改造。交接人留下的导出在
{self.input_dir}，工作副本在 {self.remote_output_dir}。
按照现行调用方的要求完成消费者，交付
{self.remote_output_dir}/src/dynamic_checker.cpp（单个 C++17 翻译单元）。
按盘面留下的调用关系和部署状态完成改造。
公开冒烟工具：python3 {self.software_dir}/smoke.py <源文件路径>。
运行限制和验收分组也在导出中。无需桌面操作或安装依赖。'''

    def to_metadata(self):
        m=super().to_metadata()
        m.update(task_id='computing_math/cf1305e_requirement_change',task_version=5,
                 output_file=self.remote_output_dir+'/src/dynamic_checker.cpp')
        return m

class ReplaySetup(BaseTaskSetup):
    async def setup(self,cfg,session):
        m=cfg.metadata
        for name in ('msg-17','0905.reply','consumer.hh','quota.export','quota.export.42','dispatch.state','admit.inc','build.rows',
                     'gate.tsv','src/dynamic_checker.cpp','restore.0/checker.py','return.stdin',
                     'lost+found/44.carbon','tally.slot','17b.stdin','17b.stdout','runtime.rows','r.44.stdin','r.44.stdout'):
            if not await session.file_exists(m['input_dir']+'/'+name):
                raise RuntimeError('Missing task input: '+name)
        if not await session.file_exists(m['software_dir']+'/smoke.py'):
            raise RuntimeError('Missing public smoke tool')
        for key in ('remote_output_dir','reference_dir'):
            p=Path(m[key])
            if p.parent!=Path(m['task_dir']) or p.name not in ('output','reference'):
                raise RuntimeError('Invalid task-local setup path')
        out=shlex.quote(m['remote_output_dir']);ref=shlex.quote(m['reference_dir'])
        src=shlex.quote(m['input_dir']+'/src/dynamic_checker.cpp')
        r=await session.run_command(f'command -v g++ && command -v python3 && sudo -n true && rm -rf -- {out} {ref} && mkdir -p {out}/src && cp {src} {out}/src/dynamic_checker.cpp',check=False)
        if r.get('return_code')!=0: raise RuntimeError('Workspace initialization failed: '+str(r))
        # Reused sandboxes must not expose a previous evaluation's private report.
        cleanup="from pathlib import Path; import shutil; " \
                "[shutil.rmtree(p) for p in Path('/tmp').glob('cf1305e-replay-eval.*') " \
                "if p.is_dir() and not p.is_symlink() and p.resolve().parent==Path('/tmp')]"
        r=await session.run_command(shlex.join(['python3','-c',cleanup]),check=False)
        if r.get('return_code')!=0: raise RuntimeError('Stale evaluation cleanup failed: '+str(r))
        if await session.directory_exists(m['reference_dir']) or await session.file_exists(m['reference_dir']):
            raise RuntimeError('Private reference visible at setup')

_setup=ReplaySetup()

async def check_reference(session,root,locks):
    script=(Path(__file__).parent/'scripts/probe_reference.py').read_text()
    nonce=secrets.token_hex(16)
    command=shlex.join(['python3','-I','-c',script,root,json.dumps(list(locks)),nonce])
    attempts=[]
    for attempt in range(3):
        try:
            result=await asyncio.wait_for(session.run_command(command,check=False),timeout=20)
        except Exception as exc:
            attempts.append({'transport_exception':type(exc).__name__})
        else:
            if result.get('return_code')==0:
                try:
                    report=json.loads(result['stdout'])
                    if report.get('protocol')!='reference-probe-v1' or report.get('root')!=root or report.get('nonce')!=nonce:
                        raise ValueError('probe identity')
                    if set(report['files'])!=set(locks):raise ValueError('probe file set')
                except (ValueError,KeyError,TypeError) as exc:
                    raise RuntimeError('Malformed reference staging diagnostic') from exc
                for name,digest in locks.items():
                    state=report['files'][name]
                    if state['status']=='missing':
                        raise RuntimeError('Confirmed private grading file missing: '+name+
                                           '; nested reference entries='+str(report.get('nested')))
                    if state['status']!='present':
                        raise RuntimeError('Private staging I/O failure: '+json.dumps(report))
                    if state['sha256']!=digest:
                        raise RuntimeError('Private grading file corrupted: '+name)
                return
            attempts.append({'return_code':result.get('return_code')})
        if attempt<2:await asyncio.sleep(1)
    raise RuntimeError('Sandbox transport/command health failed; file absence was NOT established: '+json.dumps(attempts))

@cb.tasks_config(split='train')
def load():
    cfg=TaskConfig()
    return [cb.Task(description=cfg.task_description,metadata=cfg.to_metadata(),
                    computer={'provider':'computer','setup_config':{'os_type':'linux'}})]

@cb.setup_task(split='train')
async def start(cfg,session: cb.DesktopSession):
    await _setup(cfg,session)

@cb.evaluate_task(split='train')
async def evaluate(cfg,session: cb.DesktopSession)->list[float]:
    m=cfg.metadata
    locks=json.loads((Path(__file__).parent/'scripts/reference-lock.json').read_text())
    await check_reference(session,m['reference_dir'],locks)
    r=await session.run_command('mktemp -d /tmp/cf1305e-replay-eval.XXXXXX',check=False)
    if r.get('return_code')!=0: raise RuntimeError('Evaluation workspace creation failed')
    work=r['stdout'].strip()
    if not work.startswith('/tmp/cf1305e-replay-eval.') or '\n' in work: raise RuntimeError('Invalid eval path')
    await session.write_file(work+'/run.sh',(Path(__file__).parent/'scripts/run_judge.sh').read_text())
    command=shlex.join(['bash',work+'/run.sh',m['output_file'],m['reference_dir'],work])
    r=await session.run_command(f'nohup {command} > {shlex.quote(work)}/launcher.log 2>&1 < /dev/null &',check=False)
    if r.get('return_code')!=0: raise RuntimeError('Evaluation launch failed')
    for _ in range(180):
        if await session.file_exists(work+'/rc'): break
        await asyncio.sleep(2)
    else: raise RuntimeError('Evaluation infrastructure exceeded 360 seconds')
    if (await session.read_file(work+'/rc')).strip()!='0':
        raise RuntimeError('Evaluation infrastructure failed: '+await session.read_file(work+'/stderr'))
    try:
        report=json.loads(await session.read_file(work+'/report.json'));score=report['score']
        if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1: raise ValueError('score')
        if report['status']=='judged':
            if set(report['groups'])!=set(WEIGHTS): raise ValueError('groups')
            points=0
            for g,w in WEIGHTS.items():
                item=report['groups'][g]
                if type(item['passed']) is not bool or item['points']!=(w if item['passed'] else 0): raise ValueError('group')
                points+=item['points']
            if score!=points/100: raise ValueError('total')
        elif report['status'] not in ('missing_source','invalid_source','compile_timeout','compile_error') or score!=0:
            raise ValueError('status')
    except (ValueError,KeyError,TypeError) as exc: raise RuntimeError('Malformed grading report') from exc
    await session.write_file(m['remote_output_dir']+'/evaluation_report.json',json.dumps(report,indent=2)+'\n')
    LOG.info('CF1305E_REPLAY_REPORT %s',json.dumps(report))
    return [float(score)]
