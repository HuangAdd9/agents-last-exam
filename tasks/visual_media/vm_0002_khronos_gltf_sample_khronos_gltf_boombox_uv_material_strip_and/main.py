"""ALE adapter for repairing the approved Khronos Boom Box GLB."""
from __future__ import annotations

import json
import logging
import os
import shlex
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

try:
    import cua_bench as cb
except ModuleNotFoundError:  # makes author-side validation/import self-contained
    class _Task:
        def __init__(self, description, metadata, computer):
            self.description, self.metadata, self.computer = description, metadata, computer
    def _decorator(*args, **kwargs):
        return lambda fn: fn
    cb = SimpleNamespace(Task=_Task, tasks_config=_decorator, setup_task=_decorator, evaluate_task=_decorator)

try:
    from tasks.common_config import GeneralTaskConfig
    from tasks.common_setup import BaseTaskSetup
except ModuleNotFoundError:
    @dataclass
    class GeneralTaskConfig:
        REMOTE_ROOT_DIR: str = os.environ.get("REMOTE_ROOT_DIR", "/tmp/ale-task")
        REMOTE_OUTPUT_DIR: str = "output"
        DOMAIN_NAME: str = ""
        TASK_NAME: str = ""
        VARIANT_NAME: str = "base"
        @property
        def task_dir(self): return str(PurePosixPath(self.REMOTE_ROOT_DIR, self.DOMAIN_NAME, self.TASK_NAME, self.VARIANT_NAME))
        @property
        def input_dir(self): return str(PurePosixPath(self.task_dir, "input"))
        @property
        def software_dir(self): return str(PurePosixPath(self.task_dir, "software"))
        @property
        def reference_dir(self): return str(PurePosixPath(self.task_dir, "reference"))
        @property
        def remote_output_dir(self): return str(PurePosixPath(self.task_dir, self.REMOTE_OUTPUT_DIR))
        def to_metadata(self):
            return {"task_dir": self.task_dir, "input_dir": self.input_dir, "software_dir": self.software_dir,
                    "reference_dir": self.reference_dir, "remote_output_dir": self.remote_output_dir}
    class BaseTaskSetup:
        async def __call__(self, task_cfg, session): return None


TASK_NAME = "vm_0002_khronos_gltf_sample_khronos_gltf_boombox_uv_material_strip_and"
TASK_ID = "visual_media/" + TASK_NAME
LOCAL_ROOT = Path(__file__).resolve().parent
SCORER = LOCAL_ROOT / "scripts" / "score_boombox.py"
logger = logging.getLogger(__name__)
EVALUATOR_PYTHON = os.environ.get('ALE_EVALUATOR_PYTHON', '/opt/ale-run/.venv/bin/python')
_setup = BaseTaskSetup()


def _join(base: str, *parts: str) -> str:
    return str(PurePosixPath(base, *parts))


@dataclass
class BoomBoxConfig(GeneralTaskConfig):
    DOMAIN_NAME: str = "visual_media"
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = "base"

    @property
    def task_description(self) -> str:
        return (
            f"Repair the stripped Khronos Boom Box model at {_join(self.input_dir, 'boombox_stripped.glb')}. "
            f"Use {_join(self.input_dir, 'appearance_target.png')} as the visible appearance target, "
            "restore valid UVs and the embedded material/texture bundle while preserving geometry, transforms, "
            f"and scale, then export exactly {_join(self.remote_output_dir, 'boombox_repaired.glb')}."
        )

    def to_metadata(self) -> dict[str, Any]:
        data = super().to_metadata()
        data.update({"task_id": TASK_ID,
                     "output_file": _join(self.remote_output_dir, "boombox_repaired.glb"),
                     "reference_glb": _join(self.reference_dir, "boombox_intact.glb")})
        return data


@cb.tasks_config(split="train")
def load():
    cfg = BoomBoxConfig()
    return [cb.Task(description=cfg.task_description, metadata=cfg.to_metadata(),
                    computer={"provider": "computer", "setup_config": {"os_type": "linux"}})]


async def _stage_tree(session, source: Path, destination: str) -> None:
    await session.interface.create_dir(destination)
    for path in sorted(source.rglob("*")):
        remote = _join(destination, *path.relative_to(source).parts)
        if path.is_dir():
            await session.interface.create_dir(remote)
        else:
            await session.write_bytes(remote, path.read_bytes())


@cb.setup_task(split="train")
async def start(task_cfg, session):
    await _setup(task_cfg, session)
    await _stage_tree(session, LOCAL_ROOT / "input", task_cfg.metadata["input_dir"])
    await _stage_tree(session, LOCAL_ROOT / "software", task_cfg.metadata["software_dir"])


async def _read_optional(session, path: str) -> bytes | None:
    try:
        if hasattr(session, "file_exists") and not await session.file_exists(path):
            return None
        return await session.read_bytes(path)
    except Exception:
        return None


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session) -> list[float]:
    """Run the strict glTF/resource scorer plus controlled multi-view renders."""
    metadata = task_cfg.metadata
    submission = await _read_optional(session, metadata["output_file"])
    reference = await _read_optional(session, metadata["reference_glb"])
    if submission is None or reference is None:
        return [0.0]
    eval_dir = _join(metadata["task_dir"], "_boombox_score")
    scorer_remote = _join(eval_dir, "score_boombox.py")
    submission_remote = _join(eval_dir, "candidate.glb")
    reference_remote = _join(eval_dir, "reference.glb")
    result_remote = _join(eval_dir, "result.json")
    try:
        await session.interface.create_dir(eval_dir)
        await session.write_bytes(scorer_remote, SCORER.read_bytes())
        await session.write_bytes(submission_remote, submission)
        await session.write_bytes(reference_remote, reference)
        command = shlex.join([EVALUATOR_PYTHON, '-B', scorer_remote,
                                  '--candidate', submission_remote,
                                  '--reference', reference_remote,
                                  '--result', result_remote])
        execution = await session.run_command(command, check=False, timeout=180)
        result_bytes = await _read_optional(session, result_remote)
        if result_bytes is None:
            logger.error("Boom Box scorer produced no result at %s: %s", result_remote, execution)
            return [0.0]
        result = json.loads(result_bytes.decode("utf-8"))
        score = float(result.get("score", 0.0))
        logger.info("Boom Box score details: %s", json.dumps(result, sort_keys=True))
        return [max(0.0, min(1.0, score))]
    except Exception as exc:
        logger.error("Boom Box scoring failed: %s", exc)
        return [0.0]


if __name__ == "__main__":
    print(load()[0].description)
