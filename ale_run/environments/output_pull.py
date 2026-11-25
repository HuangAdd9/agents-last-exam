"""Pull the agent's output off the sandbox after the run.

Dispatched by the lifecycle on ``artifacts_path.output_path``:

  None         → skip; output stays on the sandbox and is lost on teardown
  ``"local"``  → :func:`pull_to_host` (provider-local fast path with cua HTTP
                 fallback)
  ``"gs://X"`` → :func:`push_to_gcs` (VM-side gsutil; nothing on host)
  ``"s3://X"`` → :func:`push_to_s3` (in-box aws CLI; nothing on host)
  ``"oss://X"`` → :func:`push_to_oss` (in-box ossutil; nothing on host)
"""

from __future__ import annotations

import asyncio
import errno
import logging
import math
import os
import shlex
import shutil
from pathlib import Path
from typing import Any

from ..base_interface import SandboxHandle, TaskDataSpec

logger = logging.getLogger(__name__)

# Max concurrent per-file downloads in pull_to_host. Each download is itself
# chunked (see download_to_local), so this bounds the number of in-flight cua
# RPCs, not the payload size.
_PULL_CONCURRENCY = 8
_OUTPUT_PULL_TIMEOUT_S = 180
_DOCKER_COPY_TIMEOUT_S = 3600
_DEPENDENCY_DIRS = frozenset({'.git', '.venv', 'venv', 'node_modules', '__pycache__',
                              '.cache', '.conda', 'site-packages', '.flash_runtime',
                              '.pytest_cache', '.mypy_cache'})
_QEMU_SHARE_COPY_TIMEOUT_S = 3600
_GCS_PUSH_TIMEOUT_S = 3600
_S3_PUSH_TIMEOUT_S = 3600
_OSS_PUSH_TIMEOUT_S = 3600


def _output_dir(sandbox: SandboxHandle, task_data: TaskDataSpec) -> str:
    if task_data.sandbox_task_name:
        from .task_data import join, task_subdir

        return task_data.remote_output_dir or join(sandbox, task_subdir(sandbox, task_data), "output")
    sep = "/" if sandbox.is_linux else "\\"
    return sep.join(
        [
            sandbox.task_data_root.rstrip("/\\"),
            task_data.domain_name,
            task_data.task_name,
            task_data.variant_name,
            "output",
        ]
    )


def _docker_container(sandbox: SandboxHandle) -> str | None:
    """Container name iff this is a local Docker sandbox, else None.

    The direct Docker provider stamps both ``provider=docker`` and
    ``container_name`` into metadata. The QEMU provider also has an outer
    container, but that container does not expose the guest filesystem to
    ``docker cp``."""
    if sandbox.metadata.get("provider") != "docker":
        return None
    return sandbox.metadata.get("container_name")


def _qemu_exchange(sandbox: SandboxHandle) -> tuple[Path, str, str] | None:
    if sandbox.metadata.get("provider") != "qemu":
        return None
    host_dir = sandbox.metadata.get("exchange_host_dir")
    guest_share = sandbox.metadata.get("exchange_guest_share")
    transport = sandbox.metadata.get("exchange_guest_transport")
    if not host_dir or not guest_share or transport not in {"9p", "smb"}:
        return None
    return Path(host_dir), str(guest_share), str(transport)


def _count_output_files(dest_dir: Path) -> tuple[int, int]:
    files = 0
    total_bytes = 0
    for path in dest_dir.rglob("*"):
        if not path.is_file():
            continue
        files += 1
        try:
            total_bytes += path.stat().st_size
        except OSError:
            pass
    return files, total_bytes


async def _pull_via_docker_cp(
    container: str,
    src: str,
    dest_dir: Path,
) -> dict[str, Any]:
    """``output_path == 'local'`` fast path for the docker provider: copy the
    whole output tree off the container in one host-side ``docker cp`` — no
    per-file base64 round-trip over cua-server :5000. Symmetric with the
    ``local:`` task-data source's docker-cp input staging.

    ``docker cp <c>:<src>/. <dest>`` copies the *contents* of ``src`` into
    ``dest`` (so the run dir holds the files directly, not a nested ``output/``).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "cp",
        f"{container}:{src.rstrip('/')}/.",
        str(dest_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, err = await asyncio.wait_for(proc.communicate(), timeout=_DOCKER_COPY_TIMEOUT_S)
    except (TimeoutError, asyncio.CancelledError):
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        await proc.communicate()
        raise
    if proc.returncode != 0:
        raise RuntimeError(
            f"docker cp {container}:{src} -> {dest_dir} failed "
            f"(rc={proc.returncode}): {err.decode(errors='replace')[:300]}"
        )

    files, total_bytes = _count_output_files(dest_dir)
    logger.info(
        "pull_to_host(docker cp): %s → %s (files=%d bytes=%d)",
        src,
        dest_dir,
        files,
        total_bytes,
    )
    return {
        "transport": "docker-cp",
        "vm_path": src,
        "files": files,
        "bytes": total_bytes,
        "errors": [],
    }


def _promote_qemu_output(staged_dir: Path, dest_dir: Path) -> None:
    if not staged_dir.is_dir():
        raise RuntimeError(f"QEMU shared output staging directory is missing: {staged_dir}")
    dest_dir.parent.mkdir(parents=True, exist_ok=True)
    if not dest_dir.exists():
        try:
            staged_dir.replace(dest_dir)
            return
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
    shutil.copytree(staged_dir, dest_dir, dirs_exist_ok=True)
    shutil.rmtree(staged_dir)


async def _pull_via_qemu_share(
    sandbox: SandboxHandle,
    src: str,
    exchange_dir: Path,
    guest_share: str,
    transport: str,
    dest_dir: Path,
) -> dict[str, Any]:
    staged_dir = exchange_dir / "output"
    if staged_dir.exists():
        await asyncio.to_thread(shutil.rmtree, staged_dir)

    if transport == "9p":
        if not sandbox.is_linux:
            raise RuntimeError("QEMU 9p output transport requires a Linux guest")
        source = shlex.quote(src)
        share = shlex.quote(guest_share)
        command = (
            "set -e; "
            f"test -d {source} || exit 44; "
            "mount_dir=$(mktemp -d); "
            'cleanup() { sudo -n umount "$mount_dir" 2>/dev/null || true; '
            'rmdir "$mount_dir" 2>/dev/null || true; }; '
            "trap cleanup EXIT; "
            f'sudo -n mount -t 9p -o trans=virtio,version=9p2000.L {share} "$mount_dir"; '
            'sudo -n mkdir -p "$mount_dir/output"; '
            f'sudo -n cp -rL {source}/. "$mount_dir/output/"; '
            f'sudo -n chown -R {os.getuid()}:{os.getgid()} "$mount_dir/output"'
        )
    elif transport == "smb":
        if sandbox.is_linux:
            raise RuntimeError("QEMU SMB output transport requires a Windows guest")
        source = src.replace("'", "''")
        destination = f"{guest_share.rstrip('/\\')}\\output".replace("'", "''")
        command = (
            'powershell -NoProfile -Command "'
            f"$src='{source}'; $dst='{destination}'; "
            "if (-not (Test-Path -LiteralPath $src -PathType Container)) { exit 44 }; "
            "New-Item -ItemType Directory -Force -Path $dst | Out-Null; "
            "& robocopy $src $dst /E /COPY:DAT /DCOPY:DAT /R:2 /W:1 "
            "/NFL /NDL /NJH /NJS /NP /XJ; "
            "if ($LASTEXITCODE -ge 8) { exit $LASTEXITCODE } else { exit 0 }"
            '"'
        )
    else:
        raise ValueError(f"unknown QEMU output transport: {transport!r}")

    try:
        result = await sandbox.run_command(command, timeout=_QEMU_SHARE_COPY_TIMEOUT_S)
        if result.returncode == 44:
            return {"skipped": True, "reason": "empty_or_missing", "vm_path": src}
        if result.returncode != 0:
            raise RuntimeError(
                f"QEMU shared output copy failed (rc={result.returncode}): "
                f"{(result.stderr or result.stdout or '')[:300]}"
            )

        await asyncio.to_thread(_promote_qemu_output, staged_dir, dest_dir)
        files, total_bytes = await asyncio.to_thread(_count_output_files, dest_dir)
        logger.info(
            "pull_to_host(qemu share): %s → %s (files=%d bytes=%d)",
            src, dest_dir, files, total_bytes,
        )
        return {
            "transport": "qemu-share", "vm_path": src,
            "files": files, "bytes": total_bytes, "errors": [],
        }
    finally:
        if staged_dir.exists():
            await asyncio.to_thread(shutil.rmtree, staged_dir)


async def pull_to_host(
    sandbox: SandboxHandle,
    task_data: TaskDataSpec,
    *,
    dest_dir: Path,
) -> dict[str, Any]:
    """Try local bulk copies first, then collect over CUA with a total budget.

    ``ALE_OUTPUT_PULL_TIMEOUT_S`` bounds CUA listing and all file downloads,
    including fallback from Docker/QEMU. Dependency filtering is opt-in via
    ``ALE_OUTPUT_PULL_EXCLUDE_DEPENDENCIES=1`` and applies to CUA only.
    """
    src = _output_dir(sandbox, task_data)
    if os.environ.get("ALE_OUTPUT_PULL_EXTERNAL") == "developer-stage":
        return {"skipped": True, "reason": "provider_developer_artifact_staging",
                "transport": "developer-stage", "vm_path": src}
    budget = float(os.environ.get("ALE_OUTPUT_PULL_TIMEOUT_S", _OUTPUT_PULL_TIMEOUT_S))
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError("ALE_OUTPUT_PULL_TIMEOUT_S must be positive and finite")

    exchange = _qemu_exchange(sandbox)
    if exchange:
        try:
            return await _pull_via_qemu_share(
                sandbox, src, exchange[0], exchange[1], exchange[2], dest_dir,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("QEMU shared output pull failed, falling back to cua: %s", exc)

    container = _docker_container(sandbox)
    if container:
        try:
            return await _pull_via_docker_cp(container, src, dest_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("docker cp output pull failed, falling back to cua: %s", exc)

    report = {"transport": "cua", "vm_path": src, "files": 0, "bytes": 0,
              "errors": [], "excluded_dependency_entries": 0}
    loop = asyncio.get_running_loop()
    deadline = loop.time() + budget
    try:
        async with asyncio.timeout_at(deadline):
            await _pull_to_host(sandbox, src, dest_dir, deadline, report)
    except TimeoutError:
        report["transport"] = "partial"
        report["errors"].append({
            "error": f"output collection exceeded {budget:g}s; completed files retained",
        })
    return report


async def _pull_to_host(
    sandbox: SandboxHandle,
    src: str,
    dest_dir: Path,
    deadline: float,
    report: dict[str, Any],
) -> None:
    """Collect CUA outputs and update counts only after each file commits."""
    entries = await sandbox.list_dir(src)
    if not entries:
        report.update(skipped=True, reason="empty_or_missing")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    sep = "/" if sandbox.is_linux else "\\"
    exclude_dependencies = os.environ.get("ALE_OUTPUT_PULL_EXCLUDE_DEPENDENCIES") == "1"
    jobs: list[tuple[str, Path]] = []
    for entry in entries:
        rel = entry["relpath"]
        relative = Path(rel.replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Output listing contains a nonrelative path: {rel!r}")
        directories = relative.parts if entry.get("is_dir") else relative.parts[:-1]
        if exclude_dependencies and any(
            part in _DEPENDENCY_DIRS or part.startswith(".venv") for part in directories
        ):
            report["excluded_dependency_entries"] += 1
            continue
        if entry.get("is_dir"):
            (dest_dir / relative).mkdir(parents=True, exist_ok=True)
            continue
        remote_path = f"{src.rstrip(sep)}{sep}{rel}"
        local_path = dest_dir / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        jobs.append((remote_path, local_path))

    sem = asyncio.Semaphore(_PULL_CONCURRENCY)

    async def fetch(remote_path: str, local_path: Path) -> None:
        async with sem:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError("output collection budget exhausted")
            try:
                ok = await sandbox.download_to_local(
                    remote_path, str(local_path), timeout=min(120, remaining),
                )
                if not ok:
                    raise OSError("download_failed")
                size = local_path.stat().st_size
            except Exception as exc:  # noqa: BLE001
                report["errors"].append({"vm_path": remote_path, "error": str(exc)})
                marker = local_path.with_suffix(local_path.suffix + ".unreadable")
                marker.write_text(f"vm_path={remote_path}\nreason=download_failed\n")
                return
            report["files"] += 1
            report["bytes"] += size

    tasks = [asyncio.create_task(fetch(remote, local)) for remote, local in jobs]
    try:
        await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    logger.info("pull_to_host: %s → %s (files=%d bytes=%d errors=%d)",
                src, dest_dir, report["files"], report["bytes"], len(report["errors"]))


async def push_to_gcs(
    sandbox: SandboxHandle,
    task_data: TaskDataSpec,
    *,
    run_id: str,
    bucket: str,
) -> dict[str, Any]:
    """``output_path == 'gs://...'`` — VM-side gsutil push.

    cp -r preserves the trailing src dir name (``output``) under dst, so
    dst is the run prefix; final landing is ``<bucket>/<run_id>/output/``.
    Aliased workspaces sync contents to that same destination, including
    when the task selects an alternate output directory.

    Uses ``gsutil`` with the injected SA key (via ``_gsutil`` — same path the
    read/staging side uses) rather than ``gcloud storage cp``. The VMs carry NO
    baked credential: ``gcloud``'s ambient auth falls back to the GCE metadata
    SA, which isn't provisioned on these images, so ``gcloud storage cp`` fails
    with a metadata-server token error. The injected key authenticates writes
    consistently with reads.
    """
    from .task_data.gsbucket import _gsutil

    src = _output_dir(sandbox, task_data)
    run_prefix = f"{bucket.rstrip('/')}/{run_id}/"
    gcs_dst = f"{run_prefix}output/"
    gsutil = _gsutil(sandbox)

    if sandbox.is_linux:
        if task_data.sandbox_task_name:
            cmd = f"{gsutil} -m rsync -r {shlex.quote(src)} {shlex.quote(gcs_dst)}"
        else:
            cmd = f"{gsutil} -m cp -r {shlex.quote(src)} {shlex.quote(run_prefix)}"
    else:
        cmd = f"powershell -NoProfile -Command \"{gsutil} -m cp -r '{src}' '{run_prefix}'\""
    logger.info("push_to_gcs: %s → %s", src, gcs_dst)
    r = await sandbox.run_command(cmd, timeout=_GCS_PUSH_TIMEOUT_S)
    if r.returncode != 0:
        raise RuntimeError(f"gsutil output upload failed (rc={r.returncode}): {(r.stderr or '')[:300]}")
    return {"transport": "gcs", "gcs_path": gcs_dst}


async def push_to_s3(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *,
    run_id: str, bucket: str,
) -> dict[str, Any]:
    """``output_path == 's3://...'`` — in-box ``aws s3`` push.

    Lands the env's output dir at ``<bucket>/<run_id>/output/``. Auth is the
    instance's IAM role (AwsProvider attaches an instance profile), so there is
    no key to inject — mirror of :func:`push_to_gcs` but credential-free.
    """
    src = _output_dir(sandbox, task_data)
    s3_dst = f"{bucket.rstrip('/')}/{run_id}/output/"

    if sandbox.is_linux:
        cmd = f"aws s3 cp --recursive {shlex.quote(src)} {shlex.quote(s3_dst)}"
    else:
        cmd = (
            'powershell -NoProfile -Command "'
            f"aws s3 cp --recursive '{src}' '{s3_dst}'"
            '"'
        )
    logger.info("push_to_s3: %s → %s", src, s3_dst)
    r = await sandbox.run_command(cmd, timeout=_S3_PUSH_TIMEOUT_S)
    if r.returncode != 0:
        raise RuntimeError(
            f"aws s3 cp failed (rc={r.returncode}): {(r.stderr or '')[:300]}"
        )
    return {"transport": "s3", "s3_path": s3_dst}
async def push_to_oss(
    sandbox: SandboxHandle, task_data: TaskDataSpec, *,
    run_id: str, bucket: str,
) -> dict[str, Any]:
    """``output_path == 'oss://...'`` — in-box ``ossutil`` push.

    Lands the env's output dir at ``<bucket>/<run_id>/output/``. Auth is the
    instance's RAM role (AliyunProvider attaches one via ``ram_role_name``), so
    there is no key to inject — mirror of :func:`push_to_gcs` /
    :func:`push_to_s3` but credential-free.
    """
    # Trailing slash on the source is required: `ossutil cp -r <dir> <dst>/`
    # (no slash) copies the directory ITSELF under dst, landing files at
    # ``<dst>/output/output/...``; ``<dir>/`` copies its CONTENTS to ``<dst>/``.
    # (Same rule ossbucket._sync_cmd applies.)
    src = _output_dir(sandbox, task_data).rstrip("/") + "/"
    oss_dst = f"{bucket.rstrip('/')}/{run_id}/output/"

    if sandbox.is_linux:
        cmd = f"ossutil cp -r -f {shlex.quote(src)} {shlex.quote(oss_dst)}"
    else:
        cmd = (
            'powershell -NoProfile -Command "'
            f"ossutil cp -r -f '{src}' '{oss_dst}'"
            '"'
        )
    logger.info("push_to_oss: %s → %s", src, oss_dst)
    r = await sandbox.run_command(cmd, timeout=_OSS_PUSH_TIMEOUT_S)
    if r.returncode != 0:
        raise RuntimeError(
            f"ossutil cp failed (rc={r.returncode}): {(r.stderr or '')[:300]}"
        )
    return {"transport": "oss", "oss_path": oss_dst}
