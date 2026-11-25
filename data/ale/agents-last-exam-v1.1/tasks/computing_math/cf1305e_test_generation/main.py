"""ALE dispatch-17 generator task."""

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


async def checked_command(session, command):
    marker = "__CF1305E_RC__="
    wrapped = (
        "bash -c "
        + shlex.quote(command)
        + '; dispatch_rc=$?; printf "\\n'
        + marker
        + '%s\\n" "$dispatch_rc"'
    )
    result = await session.run_command("bash -c " + shlex.quote(wrapped), check=False)
    before, sep, after = result.get("stdout", "").rpartition("\n" + marker)
    if not sep or not after.strip().isdigit():
        raise RuntimeError(f"Lost command status: {result}")
    if int(after.strip()) != 0:
        raise RuntimeError(
            f"Task infrastructure command failed: {before} {result.get('stderr', '')}"
        )
    return before


@dataclass
class TaskConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = "computing_math"
    TASK_NAME: str = "cf1305e_test_generation"
    VARIANT_NAME: str = "base"

    @property
    def task_description(self):
        return (
            f"接手 E 的 dispatch 17。工作台在 {self.input_dir}，从 INBOX 开始。"
            f"补好测试发送端，把自包含 C++17 生成器交到 {self.remote_output_dir}/gen.cpp。"
            "它空输入启动，负责给留在服务器的程序送测试；按暴露的故障结算。"
            "旧目录、旧脚本和收据都保留了，按现行收件条款交付。"
            f"本地重放入口是 python3 {self.software_dir}/replay.py <gen.cpp路径>。"
        )

    def to_metadata(self):
        meta = super().to_metadata()
        meta.update(
            task_id="computing_math/cf1305e_test_generation",
            output_file=f"{self.remote_output_dir}/gen.cpp",
            task_version=5,
        )
        return meta


class Setup(BaseTaskSetup):
    async def setup(self, task_cfg, session):
        meta = task_cfg.metadata
        work = (
            await checked_command(session, "mktemp -d /tmp/cf1305e-setup.XXXXXX")
        ).strip()
        if not work.startswith("/tmp/cf1305e-setup.") or "\n" in work:
            raise RuntimeError("Invalid setup directory")
        scripts = Path(__file__).parent / "scripts"
        for name in ("initialize.py", "public_sha256.json"):
            await session.write_file(f"{work}/{name}", (scripts / name).read_text())
        command = " ".join(
            shlex.quote(x)
            for x in [
                "sudo",
                "-n",
                "/usr/bin/python3",
                "-I",
                f"{work}/initialize.py",
                meta["task_dir"],
                f"{work}/public_sha256.json",
            ]
        )
        await checked_command(session, command)
        if await session.directory_exists(meta["reference_dir"]):
            raise RuntimeError("Private reference visible during agent phase")


config = TaskConfig()
_setup = Setup()


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
    meta = task_cfg.metadata
    for name in ("judge.py", "engine.py", "manifest.json", "pool.cpp", "checker.cpp"):
        if not await session.file_exists(f"{meta['reference_dir']}/{name}"):
            raise RuntimeError(f"Missing private grading asset: {name}")
    work = (
        await checked_command(session, "mktemp -d /tmp/cf1305e-eval.XXXXXX")
    ).strip()
    if not work.startswith("/tmp/cf1305e-eval.") or "\n" in work:
        raise RuntimeError("Invalid evaluation directory")
    await session.write_file(
        f"{work}/run.sh", (Path(__file__).parent / "scripts/run_judge.sh").read_text()
    )
    command = " ".join(
        shlex.quote(x)
        for x in [
            "bash",
            f"{work}/run.sh",
            meta["output_file"],
            meta["reference_dir"],
            work,
        ]
    )
    await checked_command(
        session,
        f"nohup {command} > {shlex.quote(work + '/launcher.log')} 2>&1 < /dev/null &",
    )
    for _ in range(450):
        if await session.file_exists(f"{work}/rc"):
            break
        await asyncio.sleep(2)
    else:
        raise RuntimeError("Judge watchdog exceeded 900 seconds")
    rc = (await session.read_file(f"{work}/rc")).strip()
    if rc != "0":
        stderr = await session.read_file(f"{work}/stderr")
        raise RuntimeError(f"Judge infrastructure failure rc={rc}: {stderr}")
    try:
        report = json.loads(await session.read_file(f"{work}/report.json"))
        score = float(report["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("score outside [0,1]")
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Malformed judge report") from exc
    logger.info("cf1305e dispatch receipt: %s", json.dumps(report))
    await checked_command(
        session, "mkdir -p -- " + shlex.quote(meta["remote_output_dir"])
    )
    await session.write_file(
        f"{meta['remote_output_dir']}/evaluation_report.json",
        json.dumps(report, indent=2),
    )
    return [score]
