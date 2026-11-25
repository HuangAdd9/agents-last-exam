from __future__ import annotations

import importlib.util
import json
import logging
import secrets
import shlex
import tempfile
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path

try:
    import cua_bench as cb
    from tasks.common_setup import BaseTaskSetup
    from tasks.linux_runtime import LinuxTaskConfig
    _setup = BaseTaskSetup()
except ImportError:
    class _CB:
        class Task:
            def __init__(self, **kw): self.__dict__.update(kw)
        class DesktopSession: pass
        @staticmethod
        def tasks_config(split='train'): return lambda f: f
        @staticmethod
        def setup_task(split='train'): return lambda f: f
        @staticmethod
        def evaluate_task(split='train'): return lambda f: f
    cb = _CB()

    @dataclass
    class LinuxTaskConfig:
        REMOTE_OUTPUT_DIR: str = 'output'
        REMOTE_ROOT_DIR: str = '/media/user/data/agenthle'
        DOMAIN_NAME: str = ''
        TASK_NAME: str = ''
        VARIANT_NAME: str = 'base'
        OS_TYPE: str = 'linux'
        @property
        def task_dir(self): return f'{self.REMOTE_ROOT_DIR}/{self.DOMAIN_NAME}/{self.TASK_NAME}/{self.VARIANT_NAME}'
        @property
        def input_dir(self): return f'{self.task_dir}/input'
        @property
        def software_dir(self): return f'{self.task_dir}/software'
        @property
        def remote_output_dir(self): return f'{self.task_dir}/{self.REMOTE_OUTPUT_DIR}'
        @property
        def reference_dir(self): return f'{self.task_dir}/reference'
        def to_metadata(self):
            return {'domain_name': self.DOMAIN_NAME, 'task_name': self.TASK_NAME,
                    'variant_name': self.VARIANT_NAME, 'task_dir': self.task_dir,
                    'input_dir': self.input_dir, 'software_dir': self.software_dir,
                    'remote_output_dir': self.remote_output_dir, 'reference_dir': self.reference_dir}
    class BaseTaskSetup:
        async def __call__(self, *a, **k): return None
    _setup = BaseTaskSetup()

ROOT = Path(__file__).resolve().parent
TASK_NAME = 'ale_anchor_phonon_dispersion_thermodynamics_2d_hexagonal_lattice_phonon_dispersion_and_therm'
REQUIRED = ('diatomic_1d.npz', 'dispersion_2d.npz', 'dos.npz', 'thermodynamics.npz', 'results.json', 'solver.py')
EVAL_FILES = REQUIRED[:-1]
logger = logging.getLogger(__name__)


def _load_script(name):
    package_name = f'{__name__}._task_scripts'
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT / 'scripts')]
        sys.modules[package_name] = package
    module_name = f'{package_name}.{name}'
    spec = importlib.util.spec_from_file_location(module_name, ROOT / 'scripts' / f'{name}.py')
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot load task script: {name}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Config(LinuxTaskConfig):
    """Concrete framework config; annotations are required for inherited dataclass fields."""
    DOMAIN_NAME: str = 'physical_sciences'
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = 'base'
    OS_TYPE: str = 'linux'

    @property
    def task_description(self):
        return json.loads(Path(__file__).with_name("task_card.json").read_text(encoding="utf-8"))["taskPrompt"]

    def to_metadata(self):
        d = super().to_metadata()
        d.update({'task_id': f'physical_sciences/{TASK_NAME}',
                  'input_files': ['problem_spec.md', 'model_config.json'],
                  'required_output_files': list(REQUIRED)})
        return d


config = Config()


@cb.tasks_config(split='train')
def load():
    return [cb.Task(description=config.task_description, metadata=config.to_metadata(),
                    computer={'provider': 'computer', 'setup_config': {'os_type': 'linux'}})]


@cb.setup_task(split='train')
async def start(task_cfg, session):
    await _setup(task_cfg, session)
    await session.interface.create_dir(task_cfg.metadata["remote_output_dir"])


def _hidden_config(visible_config):
    base = dict(visible_config)
    rng = secrets.SystemRandom()
    m2 = round(rng.uniform(1.20, 2.30), 6)
    k3 = round(rng.uniform(0.12, 0.34), 6)
    base.update(m2=m2, chain_m2=m2, k3=k3)
    try:
        with Path(os.environ.get('PHONON_EVALUATOR_RESULTS','/tmp/phonon_evaluator_results.jsonl')).open('a', encoding='utf-8') as log:
            log.write(json.dumps({'hidden_config': base}, sort_keys=True) + '\n')
    except OSError:
        pass
    return base


async def _pull(session, remote_dir, local_dir):
    local_dir.mkdir(parents=True, exist_ok=True)
    for name in EVAL_FILES:
        path = f'{remote_dir}/{name}'
        if not await session.file_exists(path):
            raise RuntimeError(f'missing candidate artifact: {path}')
        (local_dir / name).write_bytes(await session.read_bytes(path))


@cb.evaluate_task(split='train')
async def evaluate(task_cfg, session) -> list[float]:
    """Host scorer with a strict guest boundary.

    Only solver.py runs in the guest. Config generation, reference oracle,
    schema validation, and scoring happen after artifacts are pulled back.
    """
    oracle = _load_script('oracle')
    scorer = _load_script('score_outputs')
    score_visible, weighted_score = scorer.score_visible, scorer.weighted_score

    m = task_cfg.metadata
    try:
        visible_config = json.loads(
            (await session.read_bytes(f"{m['input_dir']}/model_config.json")).decode("utf-8")
        )
    except Exception:
        return [0.0]
    hidden_cfg = _hidden_config(visible_config)
    task_cfg.metadata.setdefault('evaluator_results', []).append({'hidden_config': hidden_cfg})
    logger.info('hidden_config=%s', json.dumps(hidden_cfg, sort_keys=True))
    hidden_config_path = f"{m['input_dir']}/.evaluator_hidden_config.json"
    hidden_output_dir = f"{m['task_dir']}/.evaluator_hidden_output"
    await session.interface.create_dir(hidden_output_dir)
    await session.write_file(hidden_config_path, json.dumps(hidden_cfg, sort_keys=True) + '\n')
    command = (f"{shlex.quote(m['software_dir'] + '/python')} "
               f"{shlex.quote(m['remote_output_dir'] + '/solver.py')} "
               f"--config {shlex.quote(hidden_config_path)} "
               f"--output-dir {shlex.quote(hidden_output_dir)}")
    run = await session.run_command(f'timeout 80s {command}', check=False)
    return_code = (run.get('return_code', 1) if isinstance(run, dict)
                   else getattr(run, 'return_code', getattr(run, 'returncode', 1)))
    if return_code != 0:
        return [0.0]
    with tempfile.TemporaryDirectory(prefix='phonon_host_eval_') as td:
        root = Path(td); visible = root / 'visible'; hidden = root / 'hidden'; expected = root / 'expected'; reference = root / 'reference'
        try:
            await _pull(session, m['remote_output_dir'], visible)
            await _pull(session, hidden_output_dir, hidden)
            await _pull(session, m['reference_dir'], reference)
        except Exception as exc:
            logger.error('artifact pull failed: %s', exc)
            return [0.0]
        expected.mkdir()
        oracle.write_outputs(oracle.solve_reference(hidden_cfg), expected)
        visible_components = score_visible(visible, reference)
        hidden_components = score_visible(hidden, expected)
        hidden_pass = all(hidden_components[k] >= 1.0 for k in ('tier1', 'tier2', 'dos', 'thermo', 'summary'))
        return [float(weighted_score(visible_components, hidden_pass))]
