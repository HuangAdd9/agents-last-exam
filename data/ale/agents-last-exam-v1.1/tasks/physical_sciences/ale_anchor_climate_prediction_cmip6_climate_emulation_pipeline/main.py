"""ALE hooks for the compact CMIP6-style climate emulation benchmark."""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import cua_bench as cb
    from tasks.common_setup import BaseTaskSetup
    from tasks.linux_runtime import LinuxTaskConfig
except ImportError:
    # Keep this task importable by lightweight generation and validation tools.
    class _CB:
        class Task:
            def __init__(self, **kwargs: Any) -> None:
                self.__dict__.update(kwargs)

        DesktopSession = object

        @staticmethod
        def tasks_config(split: str = "train"):
            return lambda function: function

        @staticmethod
        def setup_task(split: str = "train"):
            return lambda function: function

        @staticmethod
        def evaluate_task(split: str = "train"):
            return lambda function: function

    cb = _CB()

    class LinuxTaskConfig:
        OS_TYPE = "linux"
        task_dir = "base"
        input_dir = "base/input"
        remote_output_dir = "base/output"
        reference_dir = "base/reference"
        software_dir = "base/software"

        def to_metadata(self) -> dict[str, Any]:
            return {
                "task_dir": self.task_dir,
                "input_dir": self.input_dir,
                "remote_output_dir": self.remote_output_dir,
                "reference_dir": self.reference_dir,
                "software_dir": self.software_dir,
            }

    class BaseTaskSetup:
        async def __call__(self, *_args: Any, **_kwargs: Any) -> None:
            return None


ROOT = Path(__file__).resolve().parent
SETUP = BaseTaskSetup()


@dataclass
class Config(LinuxTaskConfig):
    DOMAIN_NAME: str = "physical_sciences"
    TASK_NAME: str = "ale_anchor_climate_prediction_cmip6_climate_emulation_pipeline"
    VARIANT_NAME: str = "base"

    @property
    def task_description(self) -> str:
        task_card = Path(__file__).with_name("task_card.json")
        return json.loads(task_card.read_text(encoding="utf-8"))["taskPrompt"]

    def to_metadata(self) -> dict[str, Any]:
        try:
            metadata = super().to_metadata()
        except (AttributeError, TypeError):
            metadata = {}

        metadata.update(
            {
                "task_id": f"{self.DOMAIN_NAME}/{self.TASK_NAME}",
                "input_files": [
                    "climate_fixture.npz",
                    "metadata.json",
                    "problem_spec.md",
                    "Starter.md",
                    "output_contract.json",
                ],
                "eval_tmp_dir": f"{self.task_dir}/_eval_tmp",
            }
        )
        return metadata


config = Config()


@cb.tasks_config(split="train")
def load() -> list[Any]:
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg: Any, session: Any) -> None:
    await SETUP(task_cfg, session)
    await session.interface.create_dir(task_cfg.metadata["remote_output_dir"])


def _result_value(result: Any, key: str, default: Any) -> Any:
    """Read command results returned as either mappings or result objects."""
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg: Any, session: Any) -> list[float]:
    metadata = task_cfg.metadata
    eval_dir = metadata.get("eval_tmp_dir", f"{metadata['task_dir']}/_eval_tmp")
    await session.interface.create_dir(eval_dir)

    verifier_path = f"{eval_dir}/score_outputs.py"
    verifier_source = (ROOT / "scripts" / "score_outputs.py").read_text(encoding="utf-8")
    # DesktopSession.write_file expects text and performs its own UTF-8 encoding.
    await session.write_file(verifier_path, verifier_source)

    command_parts = [
        f"{metadata['software_dir']}/python",
        verifier_path,
        "--input-dir",
        metadata["input_dir"],
        "--output-dir",
        metadata["remote_output_dir"],
        "--reference-dir",
        metadata["reference_dir"],
    ]
    command = " ".join(shlex.quote(str(part)) for part in command_parts)
    result = await session.run_command(command, check=False)

    return_code = _result_value(result, "return_code", None)
    if return_code is None:
        return_code = _result_value(result, "returncode", 1)
    if return_code != 0:
        return [0.0]

    stdout = str(_result_value(result, "stdout", ""))
    for line in reversed(stdout.splitlines()):
        try:
            score = json.loads(line).get("score", 0.0)
            return [float(score)]
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            continue

    return [0.0]
