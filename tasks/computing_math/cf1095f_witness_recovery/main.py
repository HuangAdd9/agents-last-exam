"""ALE native Linux CLI task: recover a minimum-cost spanning tree witness."""

import asyncio
import json
import logging
import math
import shlex
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)


async def run_command(session, command, *, check=False):
    # This SDK version loses CommandResult.exit_code in RemoteDesktopSession.
    # Carry the shell status in stdout so setup/launcher failures cannot vanish.
    marker = "__WITNESS_SHELL_EXIT__="
    shell = (
        "set +e; bash -c "
        + shlex.quote(command)
        + '; witness_status=$?; printf "\\n'
        + marker
        + '%s\\n" "$witness_status"'
    )
    result = await session.run_command("bash -c " + shlex.quote(shell), check=False)
    stdout = result.get("stdout", "")
    before, separator, after = stdout.rpartition("\n" + marker)
    if not separator or not after.strip().isdigit():
        raise RuntimeError(f"Missing shell exit status: {result}")
    result = dict(result, stdout=before, return_code=int(after.strip()))
    if check and result["return_code"]:
        raise RuntimeError(f"Remote command failed: {result}")
    return result


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "computing_math"
    TASK_NAME: str = "cf1095f_witness_recovery"
    VARIANT_NAME: str = "base"

    @property
    def task_description(self):
        return f"""Finish the construction exporter for the commissioning handover.
The current program computes the cost invoice but does not export a construction.
The service handover is in {self.input_dir}.
The editable estimator is {self.remote_output_dir}/solution.cpp; submit that
single self-contained C++17 source. Internal implementation may be replaced.
Run bash run_tests.sh from {self.remote_output_dir} for public importer checks.
Native Linux CPU command-line task; no desktop interaction is required."""

    def to_metadata(self):
        m = super().to_metadata()
        m.update(
            task_id="computing_math/cf1095f_witness_recovery",
            output_file=f"{self.remote_output_dir}/solution.cpp",
            task_version=2,
            package_revision=9,
        )
        return m


class CleanSetup(BaseTaskSetup):
    async def setup(self, task_cfg, session):
        m = task_cfg.metadata
        for path in (
            f"{m['input_dir']}/README.md",
            f"{m['input_dir']}/solution.cpp",
            f"{m['input_dir']}/contracts/commissioning.txt",
            f"{m['input_dir']}/contracts/wire.txt",
            f"{m['input_dir']}/archive/v1-cost-note.txt",
            f"{m['input_dir']}/release.txt",
            f"{m['software_dir']}/run_tests.sh",
            f"{m['software_dir']}/public_check.py",
        ):
            if not await session.file_exists(path):
                raise RuntimeError(f"Public task file missing: {path}")
        for key, name in (
            ("remote_output_dir", "output"),
            ("reference_dir", "reference"),
        ):
            path = Path(m[key])
            if path.parent != Path(m["task_dir"]) or path.name != name:
                raise RuntimeError("Invalid task-local reset path")
        out = shlex.quote(m["remote_output_dir"])
        ref = shlex.quote(m["reference_dir"])
        source = shlex.quote(f"{m['input_dir']}/solution.cpp")
        helper = shlex.quote(f"{m['software_dir']}/run_tests.sh")
        result = await run_command(
            session,
            f"g++ --version && python3 --version && rm -rf -- {out} {ref} && "
            f"mkdir -p -- {out} && cp -- {source} {out}/solution.cpp && "
            f"cp -- {helper} {out}/run_tests.sh",
            check=False,
        )
        if result.get("return_code") != 0:
            raise RuntimeError(f"Workspace initialization failed: {result}")
        if await session.directory_exists(
            m["reference_dir"]
        ) or await session.file_exists(m["reference_dir"]):
            raise RuntimeError("Private reference visible at start")


config = TaskConfig()
_setup = CleanSetup()


@cb.tasks_config(split="train")
def load():
    cfg = TaskConfig()
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    m = task_cfg.metadata
    for name in ("judge.py", "checker.cpp", "manifest.json"):
        if not await session.file_exists(f"{m['reference_dir']}/{name}"):
            raise RuntimeError(f"Private grading file missing: {name}")
    wrapper = (Path(__file__).parent / "scripts/run_judge.sh").read_text()
    result = await run_command(
        session, "mktemp -d /tmp/cf1095f-witness-eval.XXXXXX", check=False
    )
    if result.get("return_code") != 0:
        raise RuntimeError(f"Cannot allocate evaluation workspace: {result}")
    work = result["stdout"].strip()
    if not work.startswith("/tmp/cf1095f-witness-eval.") or "\n" in work:
        raise RuntimeError("Invalid evaluator workspace response")
    await session.write_file(f"{work}/run.sh", wrapper)
    command = " ".join(
        shlex.quote(s)
        for s in ["bash", f"{work}/run.sh", m["output_file"], m["reference_dir"], work]
    )
    result = await run_command(
        session,
        f"nohup {command} > {shlex.quote(work)}/launcher.log 2>&1 < /dev/null &",
        check=False,
    )
    if result.get("return_code") != 0:
        raise RuntimeError(f"Cannot launch evaluator: {result}")
    for _ in range(450):
        if await session.file_exists(f"{work}/rc"):
            break
        await asyncio.sleep(2)
    else:
        raise RuntimeError("Evaluation infrastructure exceeded 900 seconds")
    rc = (await session.read_file(f"{work}/rc")).strip()
    if rc != "0":
        diagnostic = await session.read_file(f"{work}/stderr")
        raise RuntimeError(f"Evaluation infrastructure rc={rc}: {diagnostic}")
    try:
        report = json.loads(await session.read_file(f"{work}/report.json"))
        score = float(report["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid score")
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Malformed evaluator report") from exc
    await session.write_file(
        f"{m['remote_output_dir']}/evaluation_report.json", json.dumps(report, indent=2)
    )
    logger.info("Witness evaluation: %s", json.dumps(report))
    return [score]
