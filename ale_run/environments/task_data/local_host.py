"""``task_data_source: local:<host_dir>`` — stage task data from a HOST directory.

For the local Docker provider the host holds the task data (fetched once, e.g.
from Hugging Face) in the canonical layout:

    <host_dir>/<domain>/<task>/<variant>/{input, software, reference}

``stage_input`` copies ``input/`` (+ ``software/``) into the sandbox before the
agent runs. ``stage_reference`` copies ``reference/`` in AFTER the agent, just
before ``evaluate`` — so the reference (the answers) is never inside the sandbox
while the agent runs, exactly like the gs:// path. No encryption: the
answer is hidden by *timing*, not a password, so the host data can be plain
(no reference.7z). If the host has no reference directory, evaluation falls
back to a password-encrypted ``reference.7z`` already baked into the sandbox.

Docker shells out to ``docker cp``. Proxmox and static CUA environments pack
the selected directories into one archive, upload it to the guest in bounded
chunks through CUA, then reassemble and extract it in the guest.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import shlex
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ...base_interface import SandboxHandle, TaskDataSpec
from . import join, prepare_task_subdir, shell_q, task_subdir

logger = logging.getLogger(__name__)

_PREFIX = "local:"
_PROXMOX_UPLOAD_CHUNK_BYTES = 4 * 1024 * 1024
_STATIC_UPLOAD_CHUNK_BYTES = 512 * 1024


def _host_task_dir(source: str, task_data: TaskDataSpec) -> str:
    """Host-side ``<root>/<domain>/<task>/<variant>`` for ``local:<root>``."""
    root = source[len(_PREFIX) :]
    return os.path.join(
        root,
        task_data.domain_name or "",
        task_data.task_name or "",
        task_data.variant_name or "",
    )


async def _docker_cp(src: str, sandbox: SandboxHandle, dst: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "docker", "cp", src, f"{sandbox.id}:{dst}",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker cp {src} -> {sandbox.id}:{dst} failed "
            f"(rc={proc.returncode}): {err.decode(errors='replace')[:300]}"
        )
    owner = []
    for flag in ("-u", "-g"):
        result = await sandbox.run_command(f"id {flag}")
        value = (result.stdout or "").strip()
        if result.returncode or not value.isascii() or not value.isdecimal():
            raise RuntimeError("Could not determine task-data user/group in Docker sandbox")
        owner.append(value)
    proc = await asyncio.create_subprocess_exec(
        "docker", "exec", "--user", "0", sandbox.id,
        "chown", "-R", "-h", "--", ":".join(owner), dst,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker task-data ownership repair failed for {sandbox.id}:{dst} "
            f"(rc={proc.returncode}): {err.decode(errors='replace')[:300]}"
        )


async def _proxmox_copy_dirs(
    sandbox: SandboxHandle,
    *,
    host_task_dir: str,
    subdirs: list[str],
    remote_base: str,
) -> None:
    """Archive selected task directories and extract them in a Proxmox guest."""
    stage_dir = join(
        sandbox,
        remote_base,
        f".ale-local-stage-{uuid.uuid4().hex}",
    )
    remote_archive = join(sandbox, stage_dir, "payload.tar.gz")
    await sandbox.mkdir(remote_base)
    await sandbox.mkdir(stage_dir)

    try:
        with tempfile.TemporaryDirectory(prefix="ale-local-task-") as temp_dir:
            archive = Path(temp_dir) / "payload.tar.gz"
            with tarfile.open(archive, "w:gz") as payload:
                for subdir in subdirs:
                    payload.add(
                        os.path.join(host_task_dir, subdir),
                        arcname=subdir,
                    )

            part_count = 0
            with archive.open("rb") as stream:
                chunk_size = (
                    _STATIC_UPLOAD_CHUNK_BYTES
                    if sandbox.metadata.get("provider") == "static"
                    else _PROXMOX_UPLOAD_CHUNK_BYTES
                )
                while chunk := stream.read(chunk_size):
                    remote_part = join(
                        sandbox,
                        stage_dir,
                        f"part-{part_count:08d}",
                    )
                    await sandbox.write_file(remote_part, chunk)
                    part_count += 1

        if part_count == 0:
            raise RuntimeError("local task archive is unexpectedly empty")

        if sandbox.is_linux:
            stage_q = shlex.quote(stage_dir)
            command = (
                f"cat {stage_q}/part-* > {shlex.quote(remote_archive)} && "
                f"tar -xzf {shlex.quote(remote_archive)} "
                f"-C {shlex.quote(remote_base)}"
            )
        else:
            stage_ps = stage_dir.replace("'", "''")
            archive_ps = remote_archive.replace("'", "''")
            base_ps = remote_base.replace("'", "''")
            script = "\n".join(
                [
                    f"$stage = '{stage_ps}'",
                    f"$archive = '{archive_ps}'",
                    "$out = [IO.File]::Create($archive)",
                    "try {",
                    "  Get-ChildItem -LiteralPath $stage -Filter 'part-*' |",
                    "    Sort-Object Name | ForEach-Object {",
                    "      $part = [IO.File]::OpenRead($_.FullName)",
                    "      try { $part.CopyTo($out) } finally { $part.Dispose() }",
                    "    }",
                    "} finally { $out.Dispose() }",
                    f"& tar.exe -xzf $archive -C '{base_ps}'",
                    "exit $LASTEXITCODE",
                ]
            )
            encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
            command = f"powershell -NoProfile -EncodedCommand {encoded}"

        result = await sandbox.run_command(command, timeout=900)
        if result.returncode != 0:
            raise RuntimeError(
                "Proxmox local task extraction failed "
                f"(rc={result.returncode}): "
                f"{(result.stderr or result.stdout or '')[:300]}"
            )
    finally:
        await sandbox.rm([stage_dir])


async def _docker_mkdir(sandbox: SandboxHandle, path: str) -> None:
    """Create a staging path through Docker itself.

    The CUA mkdir RPC can race with container startup/cleanup at high
    concurrency.  Reference staging happens immediately before ``docker cp``;
    use the provider's container handle as the authoritative filesystem so the
    cp destination's parent is guaranteed to exist.
    """
    proc = await asyncio.create_subprocess_exec(
        "docker", "exec", "--user", "0", sandbox.id, "mkdir", "-p", path,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker mkdir {sandbox.id}:{path} failed (rc={proc.returncode}): "
            f"{err.decode(errors='replace')[:300]}"
        )


async def stage_input(
    sandbox: SandboxHandle,
    task_data: TaskDataSpec,
    *,
    source: str,
) -> dict[str, Any]:
    """Copy input/ (+ software/) from the host into the sandbox; make output/."""
    provider = sandbox.metadata.get("provider")
    if provider not in {"docker", "proxmox", "static"}:
        raise NotImplementedError(
            "task_data_source=local is currently supported by the direct "
            "Docker, Proxmox, and static CUA providers"
        )
    host = _host_task_dir(source, task_data)
    if not os.path.isdir(os.path.join(host, "input")):
        raise RuntimeError(
            f"task_data_source=local: expected input/ at {host!r} on the host, "
            f"not found. Did you fetch the task data? (see the local-docker README)"
        )
    base = await prepare_task_subdir(sandbox, task_data)
    in_dst = join(sandbox, base, "input")
    staged = ["input"]
    sw = os.path.join(host, "software")
    has_software = os.path.isdir(sw)
    if has_software:
        staged.append("software")

    if provider == "docker":
        await sandbox.mkdir(join(sandbox, base, "output"))
        # ``docker cp <src>/. <dst>`` copies directory contents, whether or not
        # the destination already exists.
        await sandbox.mkdir(in_dst)
        await _docker_cp(os.path.join(host, "input") + "/.", sandbox, in_dst)
    else:
        # A Proxmox clone starts clean, but an attached static VM can contain
        # output/reference left by an earlier rollout. Remove both before the
        # agent starts so a reused VM cannot see a previous answer or hidden
        # evaluator material. The host task package remains authoritative.
        targets = [
            in_dst,
            join(sandbox, base, "software"),
            join(sandbox, base, "output"),
            join(sandbox, base, "reference"),
        ]
        await sandbox.rm(targets)
        await sandbox.mkdir(join(sandbox, base, "output"))
        await _proxmox_copy_dirs(
            sandbox,
            host_task_dir=host,
            subdirs=staged,
            remote_base=base,
        )

    if has_software:
        sw_dst = join(sandbox, base, "software")
        if provider == "docker":
            await sandbox.mkdir(sw_dst)
            await _docker_cp(sw + "/.", sandbox, sw_dst)
        if sandbox.is_linux:
            result = await sandbox.run_command(
                f"find {shell_q(sandbox, join(sandbox, base, 'software'))} "
                f"-type f -exec chmod +x {{}} +",
                timeout=60,
            )
            if result.returncode:
                raise RuntimeError(f"Could not make staged software executable: {result.stderr}")
    logger.info("local: staged %s for %s from %s", staged, task_data.task_name, host)
    return {"staged": staged, "source": "local"}


async def stage_reference(
    sandbox: SandboxHandle,
    task_data: TaskDataSpec,
    *,
    source: str,
) -> dict[str, Any]:
    """Copy reference/ from the host into the container — AFTER the agent, just
    before evaluate, so the agent never sees the answers during its run."""
    provider = sandbox.metadata.get("provider")
    if provider not in {"docker", "proxmox", "static"}:
        raise NotImplementedError(
            "task_data_source=local is currently supported by the direct "
            "Docker, Proxmox, and static CUA providers"
        )
    host = _host_task_dir(source, task_data)
    ref = os.path.join(host, "reference")
    if not os.path.isdir(ref):
        from . import baked_in_sandbox

        report = await baked_in_sandbox.stage_reference(
            sandbox,
            task_data,
            source="baked_in_sandbox",
        )
        if not report.get("skipped"):
            logger.info(
                "local: no host reference for %s; used baked reference.7z",
                task_data.task_name,
            )
        return report
    base = task_subdir(sandbox, task_data)
    target = join(sandbox, base, "reference")
    # ``docker cp SRC CONTAINER:DEST`` requires DEST's parent to exist when
    # the source directory is copied into a fresh task workspace.  Normally
    # stage_input has already created it, but reference-only tasks and retries
    # can reach this phase with no input staging (or after cleanup).  Create
    # both levels explicitly so staging is idempotent and never depends on a
    # stale container directory.
    if provider == "docker":
        await _docker_mkdir(sandbox, base)
    else:
        await sandbox.mkdir(base)
    await sandbox.rm([target])  # defend against stale reference from a prior run
    if provider == "docker":
        await _docker_cp(ref, sandbox, target)
    else:
        await _proxmox_copy_dirs(
            sandbox,
            host_task_dir=host,
            subdirs=["reference"],
            remote_base=base,
        )
    logger.info("local: staged reference %s -> %s:%s", ref, sandbox.id, target)
    return {"staged": ["reference"], "source": "local"}
