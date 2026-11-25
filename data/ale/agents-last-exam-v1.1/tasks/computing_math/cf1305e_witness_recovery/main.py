"""ALE receipt witness recovery task."""
import asyncio
import hashlib
import json
import logging
import math
import shlex
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

LOG = logging.getLogger(__name__)
WEIGHTS = {'A':10, 'B':15, 'C':20, 'D':20, 'E':25, 'F':10}


async def shell(session, command):
    # The installed remote session can discard the underlying process status.
    marker = '__CF1305E_RECEIPT_RC__='
    inner = ('set +e; bash -c ' + shlex.quote(command)
             + '; receipt_rc=$?; printf "\\n' + marker + '%s\\n" "$receipt_rc"')
    result = await session.run_command('bash -c ' + shlex.quote(inner), check=False)
    before, sep, after = result.get('stdout', '').rpartition('\n' + marker)
    if not sep or not after.strip().isdigit():
        raise RuntimeError('Remote shell status missing: ' + repr(result))
    if int(after.strip()) != 0:
        raise RuntimeError('Remote shell failed: ' + repr(result))
    return before.strip()


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = 'computing_math'
    TASK_NAME: str = 'cf1305e_witness_recovery'
    VARIANT_NAME: str = 'base'

    @property
    def task_description(self):
        return f'''接手 score distribution 报价服务的 receipt 改造。现有程序只能吐出报价，
现在调用方要拿到达到这个报价的实物凭据。导出散在 {self.input_dir}，
工作副本在 {self.remote_output_dir}。
交付：{self.remote_output_dir}/src/main.cpp，单个 C++17 源文件。
不要交某次请求的静态输出。调用范围、资源和验收批次在导出里。
传输冒烟入口：python3 {self.software_dir}/smoke.py <源文件路径>。
无需联网或桌面操作。'''

    def to_metadata(self):
        m = super().to_metadata()
        m.update(task_id='computing_math/cf1305e_witness_recovery', task_version=3,
                 output_file=self.remote_output_dir+'/src/main.cpp')
        return m


class ReceiptSetup(BaseTaskSetup):
    async def setup(self, cfg, session):
        m = cfg.metadata
        for key, suffix in [('input_dir','src/main.cpp'), ('input_dir','ops/active.tab'),
                            ('input_dir','wire/tail.py'), ('input_dir','ops/quota.env'),
                            ('input_dir','wire/metric.py'), ('input_dir','captures/route.log'),
                            ('software_dir','smoke.py')]:
            if not await session.file_exists(m[key]+'/'+suffix):
                raise RuntimeError('Missing public material: '+suffix)
        for key, expected in [('remote_output_dir','output'), ('reference_dir','reference')]:
            p = Path(m[key])
            if p.parent != Path(m['task_dir']) or p.name != expected:
                raise RuntimeError('Invalid task-local reset target')
        out, ref, src = map(shlex.quote, (m['remote_output_dir'],m['reference_dir'],m['input_dir']))
        await shell(session, f'command -v g++ && command -v python3 && command -v ldd && sudo -n true && '
                    f'sudo -n rm -rf -- {out} {ref} && mkdir -p -- {out} && cp -R -- {src}/. {out}/')
        cleanup = ("from pathlib import Path; import shutil; "
                   "[shutil.rmtree(p) for p in Path('/tmp').glob('cf1305e-witness-eval.*') "
                   "if p.is_dir() and not p.is_symlink() and p.resolve().parent==Path('/tmp')]")
        await shell(session, shlex.join(['sudo','-n','python3','-c',cleanup]))
        if await session.directory_exists(m['reference_dir']) or await session.file_exists(m['reference_dir']):
            raise RuntimeError('Private reference visible at start')


_setup = ReceiptSetup()


@cb.tasks_config(split='train')
def load():
    cfg = TaskConfig()
    return [cb.Task(description=cfg.task_description, metadata=cfg.to_metadata(),
                    computer={'provider':'computer','setup_config':{'os_type':'linux'}})]


@cb.setup_task(split='train')
async def start(cfg, session: cb.DesktopSession):
    await _setup(cfg, session)


@cb.evaluate_task(split='train')
async def evaluate(cfg, session: cb.DesktopSession) -> list[float]:
    m = cfg.metadata
    scripts = Path(__file__).parent/'scripts'
    locks = json.loads((scripts/'reference-lock.json').read_text())
    for name, digest in locks.items():
        path = m['reference_dir']+'/'+name
        if not await session.file_exists(path):
            raise RuntimeError('Missing private grading file: '+name)
        if hashlib.sha256((await session.read_file(path)).encode()).hexdigest() != digest:
            raise RuntimeError('Corrupted private grading file: '+name)
    work = await shell(session, 'mktemp -d /tmp/cf1305e-witness-eval.XXXXXX')
    if not work.startswith('/tmp/cf1305e-witness-eval.') or '\n' in work:
        raise RuntimeError('Invalid evaluation directory')
    await session.write_file(work+'/run.sh', (scripts/'run_judge.sh').read_text())
    command = shlex.join(['bash',work+'/run.sh',m['output_file'],m['reference_dir'],work])
    await shell(session, f'nohup {command} > {shlex.quote(work)}/launcher.log 2>&1 < /dev/null &')
    for _ in range(300):
        if await session.file_exists(work+'/rc'):
            break
        await asyncio.sleep(2)
    else:
        raise RuntimeError('Evaluation infrastructure exceeded 600 seconds')
    if (await session.read_file(work+'/rc')).strip() != '0':
        raise RuntimeError('Evaluation infrastructure failed: '+await session.read_file(work+'/stderr'))
    try:
        report = json.loads(await session.read_file(work+'/report.json'))
        score = report['score']
        if type(score) not in (float, int) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid score')
        if report['status'] == 'judged':
            if set(report['groups']) != set(WEIGHTS):
                raise ValueError('Invalid groups')
            points = 0
            for group, weight in WEIGHTS.items():
                g = report['groups'][group]
                if type(g['passed']) is not bool or g['points'] != (weight if g['passed'] else 0):
                    raise ValueError('Invalid group result')
                points += g['points']
            if score != points/100:
                raise ValueError('Invalid total')
        elif report['status'] not in ('missing_source','invalid_source','compile_error','compile_timeout') or score != 0:
            raise ValueError('Invalid source result')
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError('Malformed grading report') from exc
    await session.write_file(m['remote_output_dir']+'/evaluation_report.json', json.dumps(report,indent=2)+'\n')
    LOG.info('CF1305E_WITNESS_REPORT %s', json.dumps(report))
    return [float(score)]
