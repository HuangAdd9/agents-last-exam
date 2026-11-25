"""ALE entry point for the frame-1 native EEVEE render pipeline repair."""
import asyncio
import json
import logging
import math
import os
import shlex
import uuid
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb
from tasks.linux_runtime import LinuxTaskConfig

PACKAGE = Path(__file__).resolve().parent
TASK_NAME = 'vm_0005_blender_native_workf_blender_workflow_eevee_race_render_pipeline_repair'
DEFAULT_BASE = f'/media/user/data/agenthle/visual_media/{TASK_NAME}/base'
logger = logging.getLogger(__name__)
# The bundled OpenEXR wheel targets CPython 3.10.  The ALE helper venv is
# currently CPython 3.12, so use the image's system 3.10 interpreter unless a
# caller explicitly supplies another evaluator.
EVALUATOR_PYTHON = os.environ.get('ALE_EVALUATOR_PYTHON', '/usr/bin/python3')


@dataclass
class RacePipelineConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = 'visual_media'
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = 'base'
    OS_TYPE: str = 'linux'

    @property
    def task_description(self):
        prompt = json.loads((PACKAGE / 'task_card.json').read_text())['taskPrompt']
        return prompt.replace(DEFAULT_BASE + '/output', self.remote_output_dir).replace(DEFAULT_BASE, self.task_dir)


@cb.tasks_config(split='train')
def load():
    config = RacePipelineConfig()
    return [cb.Task(description=config.task_description, metadata=config.to_metadata(),
                    computer={'provider': 'computer', 'setup_config': {'os_type': 'linux'}})]


async def upload_tree(session, local, remote):
    """Binary-safe upload with no dependence on the solver's working directory."""
    await session.run_command('mkdir -p ' + shlex.quote(remote), check=True)
    for path in sorted(local.rglob('*')):
        if not path.is_file():
            continue
        destination = remote + '/' + path.relative_to(local).as_posix()
        await session.run_command('mkdir -p ' + shlex.quote(str(Path(destination).parent)), check=True)
        await session.write_bytes(destination, path.read_bytes())


@cb.setup_task(split='train')
async def start(task_cfg, session: cb.DesktopSession):
    meta = task_cfg.metadata
    await upload_tree(session, PACKAGE / 'input', meta['input_dir'])
    await upload_tree(session, PACKAGE / 'software', meta['software_dir'])
    await session.run_command('mkdir -p ' + shlex.quote(meta['remote_output_dir']), check=True)


@cb.evaluate_task(split='train')
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    # References and measurement code are staged only after solver time ends.
    # A unique scratch directory prevents concurrent grading runs from colliding.
    scratch = '/tmp/race_pipeline_eval_' + uuid.uuid4().hex
    try:
        await upload_tree(session, PACKAGE / 'scripts', scratch + '/scripts')
        await upload_tree(session, PACKAGE / 'reference', scratch + '/reference')
        report = scratch + '/score.json'
        command = shlex.join([
            'timeout', '--kill-after=15', '1300', EVALUATOR_PYTHON, '-B',
            scratch + '/scripts/score_pipeline.py',
            '--submission', task_cfg.metadata['remote_output_dir'],
            '--reference', scratch + '/reference', '--report', report,
        ])
        await asyncio.wait_for(session.run_command(command, check=False), timeout=1340)
        payload = json.loads(await session.read_file(report))
        logger.info('Race pipeline evaluation: %s', json.dumps(payload))
        score = float(payload['score'])
        return [min(1.0, max(0.0, score)) if math.isfinite(score) else 0.0]
    except Exception:
        logger.exception('Race pipeline evaluator failed')
        return [0.0]
    finally:
        cleanup = 'import shutil,sys; shutil.rmtree(sys.argv[1],ignore_errors=True)'
        await session.run_command(shlex.join([EVALUATOR_PYTHON, '-c', cleanup, scratch]), check=False)
