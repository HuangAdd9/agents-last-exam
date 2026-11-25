"""Repair a cross-command source-memory switching defect in NVDLA BDMA RTL."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
from dataclasses import dataclass
from typing import Any

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)

DOMAIN_NAME = "engineering"
TASK_NAME = "nvdla_bdma_source_switch_repair"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"
VARIANT_NAME = "base"
TASK_IMAGE = "ale/task-nvdla-bdma-source-switch:8e06b1b9-v1"
GRADER_IMAGE = "ale/seed-nvdla-v1:8e06b1b9-verilator3912"
TASK_CONTAINER = "ale-nvdla-bdma-source-switch"
GRADER_TIMEOUT_S = 7200
GRADER_POLL_INTERVAL_S = 20
EVAL_ROOT = f"/dev/shm/agenthle_eval/{TASK_NAME}"
PUBLIC_CASES = (
    "bdma_chain_same_source_mc",
    "bdma_chain_switch_mc_to_cvsram",
)


@dataclass
class NvdlaBdmaConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def instructions_path(self) -> str:
        return f"{self.input_dir}/task_instructions.md"

    @property
    def public_tests_dir(self) -> str:
        return f"{self.input_dir}/public_tests"

    @property
    def setup_image_context(self) -> str:
        return f"{self.input_dir}/.setup/image_context.tar"

    @property
    def workspace_dir(self) -> str:
        return f"{self.remote_output_dir}/workspace"

    @property
    def output_bdma_dir(self) -> str:
        return f"{self.workspace_dir}/vmod/nvdla/bdma"

    @property
    def output_journal(self) -> str:
        return f"{self.remote_output_dir}/JOURNAL.md"

    @property
    def reference_grader(self) -> str:
        return f"{self.reference_dir}/grader.py"

    @property
    def reference_cases(self) -> str:
        return f"{self.reference_dir}/hidden_cases"

    @property
    def task_description(self) -> str:
        build_command = (
            f"docker exec {TASK_CONTAINER} bash -lc "
            "'cd /workspace/nvdla-hw && tools/bin/tmake -build verilator'"
        )
        test_command = (
            f"docker exec {TASK_CONTAINER} bash -lc "
            "'cd /workspace/nvdla-hw && timeout 60s make -C verif/verilator run "
            "TEST=bdma_chain_switch_mc_to_cvsram'"
        )
        return f"""\
You are repairing a command-pipeline RTL defect in the NVDLA BDMA engine.

## Task files

- Detailed task brief: `{self.instructions_path}`
- Writable BDMA source: `{self.output_bdma_dir}`
- Public traces: `{self.public_tests_dir}`
- Required journal: `{self.output_journal}`

The complete NVDLA tree and build products live in the prepared Docker container
`{TASK_CONTAINER}`. The writable BDMA directory above is mounted over the same source path in the
container, so edits made with normal file tools are immediately visible to the native NVDLA build.

Build with the repository's native command:

```bash
{build_command}
```

Run a public trace with the native target and a per-run timeout, for example:

```bash
{test_command}
```

Edit RTL only below `{self.output_bdma_dir}`. Do not modify input traces, expected data, generated
RTL, the Verilator harness, or the build system. The evaluator discards the task container, overlays
your submitted BDMA RTL on a clean seed, performs a clean rebuild, and runs hidden command chains.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update(
            {
                "task_id": TASK_ID,
                "instructions_path": self.instructions_path,
                "public_tests_dir": self.public_tests_dir,
                "setup_image_context": self.setup_image_context,
                "workspace_dir": self.workspace_dir,
                "output_bdma_dir": self.output_bdma_dir,
                "output_journal": self.output_journal,
                "reference_grader": self.reference_grader,
                "reference_cases": self.reference_cases,
                "task_image": TASK_IMAGE,
                "grader_image": GRADER_IMAGE,
                "task_container": TASK_CONTAINER,
            }
        )
        return metadata


class NvdlaBdmaSetup(BaseTaskSetup):
    async def setup(self, task_cfg: Any, session: cb.DesktopSession) -> None:
        meta = task_cfg.metadata
        required_inputs = [meta["instructions_path"], meta["setup_image_context"]]
        required_inputs.extend(
            f"{meta['public_tests_dir']}/{case_name}/input.txn"
            for case_name in PUBLIC_CASES
        )
        missing = [path for path in required_inputs if not await session.file_exists(path)]
        if missing:
            raise RuntimeError(f"staged NVDLA task input is missing: {missing}")

        workspace = shlex.quote(meta["workspace_dir"])
        bdma_dir = shlex.quote(meta["output_bdma_dir"])
        image = shlex.quote(meta["task_image"])
        container = shlex.quote(meta["task_container"])
        setup_archive = shlex.quote(meta["setup_image_context"])
        setup_dir = shlex.quote(f"/dev/shm/ale_setup/{TASK_NAME}")
        staged_setup_dir = shlex.quote(f"{meta['input_dir']}/.setup")
        public_mounts = " ".join(
            "-v "
            + shlex.quote(
                f"{meta['public_tests_dir']}/{case_name}:"
                f"/workspace/nvdla-hw/verif/traces/traceplayer/{case_name}:ro"
            )
            for case_name in PUBLIC_CASES
        )
        script = f"""set -euo pipefail
rm -rf -- {setup_dir}
mkdir -p -- {setup_dir}
tar -C {setup_dir} -xf {setup_archive}
docker build -f {setup_dir}/image/Dockerfile -t {image} {setup_dir} >/dev/null
rm -rf -- {setup_dir} {staged_setup_dir}
docker image inspect {image} >/dev/null
docker rm -f {container} >/dev/null 2>&1 || true
rm -rf -- {workspace}
mkdir -p -- {workspace}
docker run --rm --entrypoint tar {image} \
  -C /workspace/nvdla-hw -cf - vmod/nvdla/bdma | tar -C {workspace} -xf -
docker create --name {container} --network none \
  -v {bdma_dir}:/workspace/nvdla-hw/vmod/nvdla/bdma \
  {public_mounts} \
  {image} bash -lc 'exec tail -f /dev/null' >/dev/null
docker start {container} >/dev/null
docker exec {container} test -w /workspace/nvdla-hw/vmod/nvdla/bdma
"""
        result = await session.run_command(
            "bash -lc " + shlex.quote(script),
            check=False,
        )
        if result.get("return_code") != 0:
            raise RuntimeError(
                "failed to prepare NVDLA task container: "
                f"stdout={result.get('stdout', '')[:1000]} "
                f"stderr={result.get('stderr', '')[:1000]}"
            )
        logger.info(
            "[%s] task container %s ready with writable BDMA source at %s",
            meta["variant_name"],
            meta["task_container"],
            meta["output_bdma_dir"],
        )


_setup = NvdlaBdmaSetup()


@cb.tasks_config(split="train")
def load():
    cfg = NvdlaBdmaConfig(
        REMOTE_OUTPUT_DIR=os.environ.get("REMOTE_OUTPUT_DIR", "output")
    )
    return [
        cb.Task(
            description=cfg.task_description,
            metadata=cfg.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": cfg.OS_TYPE}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    container = shlex.quote(meta["task_container"])
    await session.run_command(
        f"docker rm -f {container} >/dev/null 2>&1 || true",
        check=False,
    )

    if not await session.directory_exists(meta["output_bdma_dir"]):
        logger.info("[%s] submitted BDMA directory is missing", meta["variant_name"])
        return [0.0]

    required_reference = [meta["reference_grader"], meta["reference_cases"]]
    missing_reference = [
        path
        for path in required_reference
        if not (await session.file_exists(path) or await session.directory_exists(path))
    ]
    if missing_reference:
        raise RuntimeError(f"missing NVDLA evaluator reference: {missing_reference}")

    eval_root = shlex.quote(EVAL_ROOT)
    prepare = await session.run_command(
        "bash -lc "
        + shlex.quote(
            f"rm -rf -- {eval_root}; mkdir -p -- {eval_root}/logs; "
            f"docker image inspect {shlex.quote(meta['grader_image'])} >/dev/null"
        ),
        check=False,
    )
    if prepare.get("return_code") != 0:
        raise RuntimeError(
            "failed to prepare NVDLA evaluator: "
            f"stdout={prepare.get('stdout', '')[:1000]} "
            f"stderr={prepare.get('stderr', '')[:1000]}"
        )

    grader_command = " ".join(
        [
            "python3",
            shlex.quote(meta["reference_grader"]),
            shlex.quote(meta["remote_output_dir"]),
            "--cases",
            shlex.quote(meta["reference_cases"]),
            "--image",
            shlex.quote(meta["grader_image"]),
            "--timeout",
            str(GRADER_TIMEOUT_S - 300),
            "--keep-logs",
            f"{eval_root}/logs",
        ]
    )
    launcher = f"{EVAL_ROOT}/run_grader.sh"
    await session.write_file(
        launcher,
        "#!/usr/bin/env bash\n"
        f"{grader_command} > {EVAL_ROOT}/grader.stdout 2> {EVAL_ROOT}/grader.stderr\n"
        "status=$?\n"
        f"printf '%s\\n' \"$status\" > {EVAL_ROOT}/grader.rc\n",
    )
    unit = "ale-nvdla-bdma-source-switch-grade"
    launch = await session.run_command(
        "bash -lc "
        + shlex.quote(
            f"sudo -n systemctl stop {unit}.service >/dev/null 2>&1 || true; "
            f"sudo -n systemctl reset-failed {unit}.service >/dev/null 2>&1 || true; "
            f"sudo -n systemd-run --quiet --collect --unit={unit} "
            f"--uid=user --gid=user /bin/bash {shlex.quote(launcher)}"
        ),
        check=False,
    )
    if launch.get("return_code") != 0:
        raise RuntimeError(
            "failed to launch NVDLA grader: "
            f"stdout={launch.get('stdout', '')[:1000]} "
            f"stderr={launch.get('stderr', '')[:1000]}"
        )

    waited = 0
    while waited < GRADER_TIMEOUT_S:
        await asyncio.sleep(GRADER_POLL_INTERVAL_S)
        waited += GRADER_POLL_INTERVAL_S
        poll = await session.run_command(
            f"test -f {shlex.quote(EVAL_ROOT + '/grader.rc')} && echo DONE || echo RUNNING",
            check=False,
        )
        if (poll.get("stdout") or "").strip() == "DONE":
            break
    else:
        await session.run_command(
            f"sudo -n systemctl stop {unit}.service >/dev/null 2>&1 || true",
            check=False,
        )
        raise TimeoutError(f"NVDLA grader exceeded {GRADER_TIMEOUT_S}s")

    stdout = await session.read_file(f"{EVAL_ROOT}/grader.stdout")
    stderr = await session.read_file(f"{EVAL_ROOT}/grader.stderr")
    try:
        payload = json.loads(stdout.strip())
    except json.JSONDecodeError as exc:
        logger.error(
            "[%s] invalid grader output: stdout=%s stderr=%s",
            meta["variant_name"],
            stdout[-2000:],
            stderr[-2000:],
        )
        raise RuntimeError("NVDLA grader returned invalid JSON") from exc

    score = max(0.0, min(1.0, float(payload.get("score", 0.0))))
    logger.info(
        "[%s] score=%.4f passed=%s outcomes=%s",
        meta["variant_name"],
        score,
        payload.get("passed", False),
        payload.get("outcomes", {}),
    )
    return [score]
