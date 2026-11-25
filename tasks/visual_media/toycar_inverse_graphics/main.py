from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig


TASK_NAME = "toycar_inverse_graphics"
TASK_ID = f"visual_media/{TASK_NAME}"
ROOT = Path(__file__).resolve().parent
_setup = BaseTaskSetup()
logger = logging.getLogger(__name__)
EVALUATOR_PYTHON = os.environ.get('ALE_EVALUATOR_PYTHON', '/opt/ale-run/.venv/bin/python')


def _join(base: str, *parts: str) -> str:
    return str(PurePosixPath(base, *parts))


@dataclass
class ToyCarConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "visual_media"
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = "base"

    @property
    def output_model(self) -> str:
        return _join(self.remote_output_dir, "reconstructed_scene.glb")

    @property
    def output_report(self) -> str:
        return _join(self.remote_output_dir, "reconstruction_report.md")

    @property
    def task_description(self) -> str:
        return json.loads((ROOT / "task_card.json").read_text(encoding="utf-8"))["taskPrompt"]

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "output_model": self.output_model,
                "output_report": self.output_report,
                "eval_contract": _join(self.reference_dir, "evaluation_contract.json"),
                "eval_points": _join(self.reference_dir, "reference_surface.npz"),
                "eval_dir": _join(self.task_dir, "_eval"),
            }
        )
        return metadata


CONFIG = ToyCarConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=CONFIG.task_description,
            metadata=CONFIG.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    """Measure the submitted model; this is the task's only official score entry point."""
    metadata = task_cfg.metadata
    required = (metadata["output_model"], metadata["eval_contract"], metadata["eval_points"])
    try:
        for path in required:
            if not await session.file_exists(path):
                return [0.0]
        eval_dir = metadata["eval_dir"]
        await session.interface.create_dir(eval_dir)
        scorer = _join(eval_dir, "score_reconstruction.py")
        blender_helper = _join(eval_dir, "measure_reconstruction.py")
        result_path = _join(eval_dir, "score.json")
        await session.write_file(
            scorer, (ROOT / "scripts" / "score_reconstruction.py").read_text(encoding="utf-8")
        )
        await session.write_file(
            blender_helper,
            (ROOT / "scripts" / "measure_reconstruction.py").read_text(encoding="utf-8"),
        )
        command = (
            f'{EVALUATOR_PYTHON} -B "{scorer}" --candidate "{metadata["output_model"]}" '
            f'--report "{metadata["output_report"]}" '
            f'--contract "{metadata["eval_contract"]}" --reference-points "{metadata["eval_points"]}" '
            f'--reference-dir "{metadata["reference_dir"]}" --blender-helper "{blender_helper}" '
            f'--work-dir "{eval_dir}" --result "{result_path}"'
        )
        # DesktopSession.run_command has no timeout keyword in the ALE runtime.
        execution = await session.run_command(command, check=False)
        if not await session.file_exists(result_path):
            logger.error("Toy Car scorer produced no result: %s", execution)
            return [0.0]
        payload = json.loads((await session.read_bytes(result_path)).decode("utf-8"))
        score = max(0.0, min(1.0, float(payload.get("score", 0.0))))
        logger.info("Toy Car score details: %s", json.dumps(payload, sort_keys=True))
        return [score]
    except Exception as exc:
        logger.exception("Toy Car evaluation failed: %s", exc)
        return [0.0]


if __name__ == "__main__":
    print(CONFIG.task_description)
