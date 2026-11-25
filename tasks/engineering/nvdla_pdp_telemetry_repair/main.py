"""ALE adapter for the v8 NVDLA PDP telemetry task."""

from __future__ import annotations

import json
import logging
import math
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)
TASK_NAME = "nvdla_pdp_telemetry_repair"
TASK_ID = f"engineering/{TASK_NAME}"


@dataclass
class PdpTelemetryConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "engineering"
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = "base"

    @property
    def task_description(self) -> str:
        brief = Path(__file__).with_name("TASK.md").read_text(encoding="utf-8")
        return (
            brief
            + "\n\n## ALE workspace\n"
            + f"Work in `{self.remote_output_dir}`. Read its README.md and SPEC.md.\n"
            + f"The read-only initial project is in `{self.input_dir}`. "
            + "A writable copy is already prepared in the output directory. "
            + "All relative paths in the task brief refer to that writable copy.\n"
            + f"Deliver modified RTL under `{self.remote_output_dir}/rtl/`, "
            + "with your diagnosis and regressions in the same output directory.\n"
        )

    def to_metadata(self) -> dict:
        return super().to_metadata() | {"task_id": TASK_ID}


class PdpTelemetrySetup(BaseTaskSetup):
    async def setup(self, task_cfg, session) -> None:
        meta = task_cfg.metadata
        script = """
import pathlib, shutil, subprocess, sys
source, out, reference = (pathlib.Path(p).resolve() for p in sys.argv[1:])
if source.name != 'input' or out.parent != source.parent or reference.parent != source.parent:
    raise RuntimeError('Invalid task-local setup paths')
if out.name not in ('output', 'output_test_pos', 'output_test_neg') or reference.name != 'reference':
    raise RuntimeError('Invalid task-local output/reference directory')
for name in ('TASK.md', 'SPEC.md', 'README.md', 'tools/build.py', 'environment/runtime-image.txt'):
    if not (source / name).is_file():
        raise RuntimeError('Missing staged input: ' + name)
if not (source / 'rtl').is_dir():
    raise RuntimeError('Missing staged RTL')
image = (source / 'environment/runtime-image.txt').read_text().strip()
subprocess.run(['docker', 'image', 'inspect', image], check=True, stdout=subprocess.DEVNULL)
if reference.exists():
    shutil.rmtree(reference)
if out.exists():
    shutil.rmtree(out)
shutil.copytree(source, out)
if reference.exists():
    raise RuntimeError('Private reference is visible during setup')
print('Prepared writable ALE project:', out)
"""
        command = shlex.join([
            "python3", "-c", script, meta["input_dir"],
            meta["remote_output_dir"], meta["reference_dir"],
        ])
        result = await session.run_command(command, check=False)
        if result.get("return_code") != 0:
            raise RuntimeError(f"PDP setup failed: {result}")


_setup = PdpTelemetrySetup()


@cb.tasks_config(split="train")
def load():
    cfg = PdpTelemetryConfig(
        REMOTE_ROOT_DIR=os.environ.get("REMOTE_ROOT_DIR", "/media/user/data/agenthle"),
        REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output"),
    )
    return [cb.Task(
        description=cfg.task_description,
        metadata=cfg.to_metadata(),
        computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
    )]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    reference = meta["reference_dir"]
    project = meta["remote_output_dir"]
    for relative in ("grader/grade.py", "grader/cases.json", "grader/unit_cases.json",
                     "tools/build.py", "environment/runtime-image.txt"):
        if not await session.file_exists(f"{reference}/{relative}"):
            raise RuntimeError(f"Missing staged evaluator: {reference}/{relative}")
    if not await session.directory_exists(f"{project}/rtl"):
        return [0.0]

    report_dir = f"{project}/.ale-evaluation/{uuid4().hex}"
    result = await session.run_command(shlex.join([
        "python3", f"{reference}/grader/grade.py", project, report_dir,
    ]), check=False)
    rc = result.get("return_code")
    if rc not in (0, 1):
        raise RuntimeError(f"PDP grader infrastructure failure; logs: {report_dir}; {result}")
    try:
        report = json.loads(await session.read_file(f"{report_dir}/summary.json"))
        score = float(report["score"]) / 100.0
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid PDP grade summary at {report_dir}") from exc
    if report.get("status") != "completed" or not math.isfinite(score) or not 0 <= score <= 1:
        raise RuntimeError(f"Incomplete or invalid PDP grade: {report}")
    logger.info("PDP evaluation at %s: %s", report_dir, json.dumps(report))
    return [score]
