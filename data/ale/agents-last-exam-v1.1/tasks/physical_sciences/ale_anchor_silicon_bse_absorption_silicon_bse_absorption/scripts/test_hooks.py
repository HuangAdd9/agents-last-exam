#!/usr/bin/env python3
"""Exercise the production TaskLoader and DesktopSession file contract.

Scientific commands run in the pinned offline image. The adapter uses
``session.interface.read_bytes`` and ``write_bytes`` exactly as ``main.py``
does; the session deliberately has no convenience ``read_bytes`` method.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path


PACKAGE = Path(__file__).resolve().parents[1]
ALE_REPOSITORY = Path(os.environ.get("ALE_REPOSITORY", "/data/ale/agents-last-exam"))
IMAGE = "ale/base-physical-qe-bgw:6.7.0-4.0-20260920"


class LocalInterface:
    """Filesystem part of the production computer interface."""

    async def create_dir(self, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)

    async def read_bytes(self, path: str) -> bytes:
        return Path(path).read_bytes()

    async def write_bytes(self, path: str, data: bytes) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(bytes(data))


class PinnedImageSession:
    """Test adapter backed by one fresh pinned-image container per command."""

    def __init__(self, shared_root: Path, image: str = IMAGE) -> None:
        self.interface = LocalInterface()
        self.shared_root = shared_root.resolve()
        self.image = image

    async def run_command(self, command: str, timeout=None, check=False):
        # The image's OpenMPI build needs its root-compatible CMA fallback in
        # the Docker authoring environment. Restore ownership after each
        # command so host-side staging remains writable by the test process.
        wrapped_command = (
            f"{command}\n"
            "status=$?\n"
            "exit $status"
        )
        docker_command = [
            # Nested Docker in the authoring host blocks OpenMPI's socketpair
            # unless the inner container receives the outer runtime devices.
            "docker", "run", "--rm", "--network=none", "--privileged",
            "--shm-size=2g",
            "--env", "HOME=/tmp", "--env", "TMPDIR=/tmp",
            "--volume", f"{self.shared_root}:{self.shared_root}",
            self.image, "bash", "-lc", wrapped_command,
        ]
        process = await asyncio.create_subprocess_exec(
            *docker_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            communication = process.communicate()
            stdout, stderr = await (
                asyncio.wait_for(communication, timeout=timeout) if timeout else communication
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise
        result = {
            "return_code": process.returncode,
            "stdout": stdout.decode(errors="replace"),
            "stderr": stderr.decode(errors="replace"),
        }
        if check and process.returncode:
            raise RuntimeError(result)
        return result


def load_with_production_loader(remote_root: Path):
    """Load through ALE's real loader after fixing its remote root."""
    os.environ["REMOTE_ROOT_DIR"] = str(remote_root)
    repository = str(ALE_REPOSITORY)
    if repository not in sys.path:
        sys.path.insert(0, repository)
    from ale_run.tasks.loader import TaskLoader

    loader = TaskLoader(str(PACKAGE))
    task_info = loader.load()
    module = loader._load_module()
    task = module.load()[0]
    return module, task, task_info


def stage_task_data(metadata: dict) -> None:
    """Reproduce runner staging; evaluator assets must remain host-local."""
    for source_name, remote_key in (
        ("input", "input_dir"),
        ("reference", "reference_dir"),
        ("software", "software_dir"),
    ):
        source = PACKAGE / source_name
        target = Path(metadata[remote_key])
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)


def exercise_loader(remote_root: Path) -> None:
    module, task, info = load_with_production_loader(remote_root)
    metadata = task.metadata
    card = json.loads((PACKAGE / "task_card.json").read_text(encoding="utf-8"))
    assert info["snapshot_name"] == "cpu-free-ubuntu"
    assert info["task_data"].source_relpath.endswith(
        "physical_sciences/ale_anchor_silicon_bse_absorption_silicon_bse_absorption/base"
    )
    assert card["requiredSystemPackages"] == ["qe-bgw-6.7.0-4.0"]
    assert metadata["task_id"] == card["taskId"]
    assert metadata["input_files"] == list(module.INPUT_FILES)
    assert metadata["output_files"] == list(module.OUTPUT_FILES)
    print("production TaskLoader contract passed")


async def exercise_start(remote_root: Path, evaluate: bool, malicious: bool = False) -> None:
    module, task, _ = load_with_production_loader(remote_root)
    metadata = task.metadata
    stage_task_data(metadata)
    session = PinnedImageSession(remote_root)
    await module.start(task, session)
    for name in module.INPUT_FILES:
        assert Path(metadata["input_dir"], name).is_file()
    staged_reference = Path(metadata["reference_dir"])
    assert staged_reference.is_dir()
    assert {path.name for path in staged_reference.iterdir()} == {"README.md"}
    assert (staged_reference / "README.md").is_file()
    assert not any(staged_reference.rglob("*.dat"))
    assert not any(path.name == "evaluator_private" for path in staged_reference.rglob("*"))
    assert not any(path.name == "evaluator_private" for path in remote_root.rglob("evaluator_private"))
    print("start hook passed in pinned image")
    if not evaluate:
        return

    reference_output = PACKAGE / "evaluator_private/visible_reference"
    reference_work = PACKAGE / "evaluator_private/visible_evidence"
    for name in (*module.OUTPUT_FILES[:-2], "run_manifest.json"):
        await session.interface.write_bytes(
            f"{metadata['remote_output_dir']}/{name}",
            (reference_output / name).read_bytes(),
        )
    if malicious:
        workflow = """#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path

# Claimed executables: pw.x pw2bgw.x epsilon.cplx.x sigma.cplx.x
# kernel.cplx.x absorption.cplx.x inteqp.cplx.x
parser = argparse.ArgumentParser()
for option in ("case", "structure", "pseudo", "work_dir", "output_dir"):
    parser.add_argument("--" + option.replace("_", "-"), required=True)
args = parser.parse_args()
for root in (Path(__file__).resolve().parent.parent, Path("/tmp")):
    for candidate in root.rglob("reference"):
        # Probe both leaked layouts called out by the scientific review.  Try
        # hidden first because it would satisfy the changed-case rerun without
        # performing any QE/BerkeleyGW work; visible is still a forbidden
        # staged answer and must not be exposed either.
        for variant in ("hidden", "visible"):
            outputs = candidate / variant
            evidence = candidate / (variant + "_work")
            if outputs.is_dir() and evidence.is_dir():
                shutil.copytree(outputs, args.output_dir, dirs_exist_ok=True)
                shutil.copytree(evidence, args.work_dir, dirs_exist_ok=True)
                raise SystemExit(0)
raise SystemExit(23)
"""
    else:
        workflow = (PACKAGE / "scripts/oracle_workflow.py").read_text(encoding="utf-8")
    await session.interface.write_bytes(
        f"{metadata['remote_output_dir']}/workflow.py", workflow.encode()
    )
    manifest = json.loads((reference_output / "run_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["stages"]:
        for key in ("deck", "log"):
            name = record[key]
            await session.interface.write_bytes(
                f"{metadata['work_dir']}/{name}", (reference_work / name).read_bytes()
            )

    evaluation = asyncio.create_task(module.evaluate(task, session))
    elapsed = 0
    while not evaluation.done():
        done, _ = await asyncio.wait({evaluation}, timeout=30)
        if done:
            break
        elapsed += 30
        print(f"end-to-end evaluator still running ({elapsed}s)", flush=True)
    score = await evaluation
    if malicious:
        assert len(score) == 1 and score[0] <= 0.69, score
        print("filesystem-copy workflow was capped with no staged evaluator references")
    else:
        assert len(score) == 1 and score[0] > 0.99, score
        print("evaluate hook passed with an independently recomputed hidden workflow")


async def exercise(mode: str) -> None:
    if not ALE_REPOSITORY.is_dir():
        raise FileNotFoundError(f"ALE repository not found: {ALE_REPOSITORY}")
    with tempfile.TemporaryDirectory(prefix="ale_si_bse_hooks_") as temporary:
        remote_root = Path(temporary)
        if mode == "loader":
            exercise_loader(remote_root)
        else:
            await exercise_start(
                remote_root,
                evaluate=mode in {"evaluate", "malicious"},
                malicious=mode == "malicious",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("loader", "start", "evaluate", "malicious"))
    asyncio.run(exercise(parser.parse_args().mode))
