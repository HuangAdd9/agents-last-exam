"""DockerProvider — ephemeral Docker containers via ``docker run``.

Each ``acquire`` launches a fresh container from the ale-kasm image,
maps a random host port to the container's cua-server (port 8000),
waits for the server to become healthy, and returns a SandboxHandle
whose ``endpoint`` is ``http://127.0.0.1:<host_port>``.

Concurrent tasks each get their own container + port — no proxy layer
required. Docker's native port mapping IS the routing mechanism.

Container naming follows the same ``<prefix>-<task-slug>-<hash8>``
scheme as GCE VMs for log greppability.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import math
import os
import random
import re
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ...base_interface import Provider, ReleaseMode, SandboxHandle, SandboxSpec

logger = logging.getLogger(__name__)

_CUA_INTERNAL_PORT = 8000
_VNC_INTERNAL_PORT = 6901
_CUA_READY_TIMEOUT = 120
_CUA_READY_POLL_INTERVAL = 3

_DEFAULT_CONTAINER_REF = "agentslastexam/ale-kasm:latest"  # fallback if an Image has no docker_image
_CONTAINER_PREFIX = "ale"                                  # container name prefix (not configurable)
_CPUSET_LOCK_PATH = Path(tempfile.gettempdir()) / f"ale-docker-cpuset-{os.getuid()}.lock"
_CPUSET_POLL_INTERVAL = 1.0
_DOCKER_QUERY_TIMEOUT = 30.0


_GCS_KEY_CONTAINER_PATH = "/etc/agenthle/gcs-reader.json"
# System-wide boto config so gsutil authenticates regardless of which user the
# container runs as (kasm-user on ale-kasm, user on ale-ubuntu22-docker, ...).
# boto reads /etc/boto.cfg before any per-user ~/.boto, so this is user-agnostic.
_BOTO_CONFIG_CONTAINER_PATH = "/etc/boto.cfg"


@dataclass(frozen=True)
class DockerProviderConfig:
    """Docker provider config (the per-snapshot ``docker:`` block).

    image       Image NAME (registry key, e.g. "ale-kasm"). The container ref to
                boot + the cua-server port + sandbox paths are read from that
                Image's registry entry — nothing image-specific is configured here.
    shm_size    Shared memory size (default: 512m).
    cpus        CPU limit per container. 0 (default) = derive from the task's GCE
                machine type (task_card ``vm.machineType``, e.g. "c4-standard-4")
                so the container matches what gcloud would allocate. A non-zero
                value overrides the derived value (a host-protection cap).
                Also reserves ceil(cpus) logical CPUs using --cpuset-cpus;
                acquisition waits until that many CPUs are available.
    memory      Memory limit per container (e.g. "8g"). "" (default) = derive from
                the task's GCE machine type, same as ``cpus``. A non-empty value
                overrides (a host-protection cap).
    allow_swap  When false, set --memory-swap equal to --memory to disable swap.
    lxcfs_root  Optional host LXCFS mount point (e.g. /var/lib/lxcfs). Bind its
                resource views read-only into /proc and the CPU online file.
    resolution  Virtual display size (w, h) for the in-container Xvfb, passed via
                ``ALE_SCREEN_RESOLUTION``. Default (1024, 768). Accepts [w, h] or
                "WxH".
    gcs_sa_key  Host path to a GCS service-account JSON key. NOT a provider knob —
                the loader injects it from the env-yaml top level (it travels with
                ``task_data_source: gs://…``). When set, it is copied into each
                container and gsutil/boto configured so gs:// staging works; the
                key's ``project_id`` also bills requester-pays buckets.
    privileged  Run containers with ``--privileged``. Needed for tasks whose eval
                runs nested Docker (openroad, minikube/k8s, compose) — the baked
                image starts a fresh inner dockerd. Under ROOTLESS docker this is
                user-namespace-bounded (container root maps to the unprivileged
                host user), so it does not expose the host kernel.
    image_ref   Override the container ref to boot (default: the Image entry's
                ``docker_image``). Lets one env config pin a specific tag, e.g.
                a DinD-capable build, without editing the Image registry.
    image_revision  Optional expected SHA-256 image ID or repository digest.
    """

    image: str = "ale-kasm"
    shm_size: str = "512m"
    cpus: float = 0
    memory: str = ""
    resolution: tuple[int, int] | None = None
    gcs_sa_key: str = ""
    privileged: bool = False
    image_ref: str = ""
    network_mode: Literal["bridge", "host"] = "bridge"
    # Start the in-container nested dockerd (DinD) for custom development.
    # The supported rootless-Docker profile excludes nested-runtime tasks and
    # routes them to the QEMU/KVM Ubuntu guest instead. When true the container
    # gets ALE_ENABLE_DIND=1; enabling it broadly can add substantial
    # fuse-overlayfs I/O at high concurrency.
    enable_dind: bool = False
    image_revision: str | None = None
    allow_swap: bool = True
    lxcfs_root: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.allow_swap, bool):
            raise TypeError("docker allow_swap must be a boolean")
        if self.network_mode not in {"bridge", "host"}:
            raise ValueError("docker.network_mode must be 'bridge' or 'host', got " f"{self.network_mode!r}")


def _build_provider_config(raw: dict[str, Any]) -> DockerProviderConfig:
    image_revision = raw.get("image_revision")
    if image_revision is not None:
        image_revision = str(image_revision).strip() or None
    if image_revision is not None and not re.fullmatch(r"sha256:[0-9a-f]{64}", image_revision):
        raise ValueError("docker image_revision must be a sha256:<64 lowercase hex digits> digest")
    gcs_sa = raw.get("gcs_sa_key") or ""
    if gcs_sa:
        from pathlib import Path as _P
        gcs_sa = str(_P(gcs_sa).expanduser().resolve())
    from .gcloud import _parse_resolution
    return DockerProviderConfig(
        image=str(raw.get("image") or "ale-kasm"),
        shm_size=str(raw.get("shm_size") or "512m"),
        cpus=float(raw.get("cpus") or 0),
        memory=str(raw.get("memory") or ""),
        resolution=_parse_resolution(
            raw.get("resolution"), str(raw.get("image") or "ale-kasm")
        ),
        gcs_sa_key=gcs_sa,
        privileged=bool(raw.get("privileged") or False),
        image_ref=str(raw.get("image_ref") or ""),
        network_mode=str(raw.get("network_mode") or "bridge").strip().lower(),
        enable_dind=bool(raw.get("enable_dind") or False),
        image_revision=image_revision,
        allow_swap=raw.get("allow_swap", True),
        lxcfs_root=str(raw.get("lxcfs_root") or ""),
    )


def _lxcfs_mount_args(root: str) -> list[str]:
    """Build read-only resource-view mounts; fail if the configured source is missing."""
    source_root = Path(root)
    if not source_root.is_absolute() or "," in root:
        raise ValueError("docker lxcfs_root must be an absolute path without commas")
    paths = [f"proc/{name}" for name in (
        "cpuinfo", "meminfo", "stat", "swaps", "uptime", "diskstats", "slabinfo",
    )]
    paths.append("sys/devices/system/cpu/online")
    args = []
    for relative in paths:
        source = source_root / relative
        if not source.is_file():
            raise ValueError(f"LXCFS resource view is unavailable: {source}")
        # 只替换资源视图；CPU/内存实际限额仍由 cgroup 执行。
        args.extend(["--mount", f"type=bind,src={source},dst=/{relative},readonly"])
    return args


def _generate_container_name(
    prefix: str,
    *,
    snapshot: str,
    task_id: str = "",
    harness: str = "",
    model_tag: str = "",
) -> str:
    if task_id:
        body = re.sub(r"[^a-z0-9]", "-", task_id.lower()).strip("-")[:40]
    else:
        body = re.sub(r"[^a-z0-9]", "-", snapshot.lower()).strip("-")[:30]
    seed = f"{prefix}:{task_id}:{harness}:{model_tag}:{snapshot}:{time.time()}:{random.random()}"
    h = hashlib.sha256(seed.encode()).hexdigest()[:8]
    return f"{prefix}-{body}-{h}"


async def _run_docker(*args: str) -> tuple[int, str, str]:
    cmd = ["docker", *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    query = bool(args) and (args[0] in {"ps", "inspect"} or args[:2] == ("image", "inspect"))
    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=_DOCKER_QUERY_TIMEOUT if query else None,
        )
    except (TimeoutError, asyncio.CancelledError) as exc:
        # A stuck query must not keep the cross-runner CPU allocation lock.
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
        await proc.communicate()
        if isinstance(exc, TimeoutError):
            raise RuntimeError(  # noqa: TRY004
                f"docker {' '.join(args[:2])} exceeded {_DOCKER_QUERY_TIMEOUT:g}s"
            ) from exc
        raise
    return (
        proc.returncode or 0,
        stdout_b.decode(errors="replace").strip(),
        stderr_b.decode(errors="replace").strip(),
    )


@asynccontextmanager
async def _cpu_allocation_lock():
    # Same-user runners share a lock; Docker is the reservation ledger, including
    # containers kept alive after a runner exits. Never unlink the lock file.
    with _CPUSET_LOCK_PATH.open("a") as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.1)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


async def _occupied_cpus() -> set[int] | None:
    rc, stdout, stderr = await _run_docker("ps", "--quiet")
    if rc != 0:
        raise RuntimeError(f"Failed to list containers for CPU allocation: {stderr}")
    container_ids = stdout.split()
    if not container_ids:
        return set()
    rc, stdout, stderr = await _run_docker(
        "inspect", "--format",
        '{{if .State.Running}}{{json .HostConfig.CpusetCpus}}{{end}}',
        *container_ids,
    )
    if rc != 0:
        if "No such object" in stderr or "No such container" in stderr:
            return None  # A concurrent removal invalidated the snapshot; rescan.
        raise RuntimeError(f"Failed to inspect containers for CPU allocation: {stderr}")
    occupied = set()
    for line in stdout.splitlines():
        if not line.strip():
            continue
        cpu_set = json.loads(line)
        for part in cpu_set.split(",") if cpu_set else []:
            bounds = part.split("-")
            first, last = int(bounds[0]), int(bounds[-1])
            occupied.update(range(first, last + 1))
    return occupied


async def _start_container(name: str, run_args: list[str], cpus: float) -> tuple[str, str]:
    """Reserve disjoint logical CPUs on the local Docker host and start a container."""
    if not math.isfinite(cpus) or cpus <= 0:
        raise ValueError(f"Docker CPU allocation requires a positive finite quota, got {cpus}")
    allowed = set(os.sched_getaffinity(0))
    count = math.ceil(cpus)
    if count > len(allowed):
        raise ValueError(f"Container {name} needs {count} CPUs; only {len(allowed)} are allowed")
    waiting = False
    while True:
        async with _cpu_allocation_lock():
            occupied = await _occupied_cpus()
            if occupied is not None:
                free = sorted(allowed - occupied)
                if len(free) >= count:
                    cpu_set = ",".join(str(cpu) for cpu in free[:count])
                    args = [run_args[0], "--cpuset-cpus", cpu_set, *run_args[1:]]
                    logger.info("Container %s assigned cpuset=%s quota=%s", name, cpu_set, cpus)
                    start = asyncio.create_task(_run_docker(*args))
                    try:
                        rc, stdout, stderr = await asyncio.shield(start)
                        if rc != 0:
                            raise RuntimeError(f"docker run failed for {name}: {stderr}")
                        return stdout, cpu_set
                    except BaseException:
                        # A cancelled CLI request can still create a container.
                        # Finish it before cleanup and before releasing the lock.
                        try:
                            await asyncio.gather(start, return_exceptions=True)
                            rc, _, stderr = await _run_docker("rm", "-f", name)
                            if rc != 0:
                                logger.error("Failed to remove container %s: %s", name, stderr)
                        except Exception:
                            logger.exception("Failed to clean up container %s during start", name)
                        raise
        if not waiting:
            logger.info("Container %s waiting for %s free logical CPUs", name, count)
            waiting = True
        await asyncio.sleep(_CPUSET_POLL_INTERVAL)


async def _get_host_port(container_name: str, internal_port: int) -> int:
    """Inspect the container to find the host port mapped to internal_port."""
    rc, stdout, stderr = await _run_docker(
        "inspect",
        "--format",
        f'{{{{(index (index .NetworkSettings.Ports "{internal_port}/tcp") 0).HostPort}}}}',
        container_name,
    )
    if rc != 0:
        raise RuntimeError(
            f"Failed to inspect container {container_name}: {stderr}"
        )
    return int(stdout.strip())


async def _get_resource_limits(container_name: str) -> dict[str, int | float]:
    rc, stdout, stderr = await _run_docker(
        "inspect", "--format", "{{json .HostConfig}}", container_name,
    )
    if rc != 0:
        raise RuntimeError(
            f"Failed to inspect container {container_name} limits: {stderr}"
        )
    host_config = json.loads(stdout)
    return {
        "cpu_quota": host_config["NanoCpus"] / 1_000_000_000,
        "memory_limit_bytes": host_config["Memory"],
        # Docker 的 MemorySwap 是内存与 swap 合计上限，不是单独的 swap 容量。
        "memory_and_swap_limit_bytes": host_config.get("MemorySwap", 0),
    }


async def _ensure_task_numpy(container_name: str, container_ref: str) -> None:
    """Make ``python3`` in a task container able to import NumPy.

    The exported Ubuntu Docker image has a Python 3.12 CUA virtualenv, while
    some task images only carry NumPy for the system Python 3.10.  The task
    graders are invoked through the CUA virtualenv's ``python3`` and therefore
    see ``ModuleNotFoundError: numpy`` even though ``/usr/bin/python3`` has it.
    Install a binary wheel into that virtualenv once per container.  This is
    intentionally runtime-local: it does not mutate the pinned source image
    or weaken image provenance checks, and it is a no-op for images that
    already contain NumPy.

    Images without the standard CUA virtualenv are left unchanged; they use a
    different runtime and must provide their own evaluator dependencies.
    """
    # scIB ships its own Python 3.10 environment under /opt/scib-pipeline-R4.0;
    # installing a Python 3.12 wheel into the generic CUA path can shadow its
    # pinned NumPy ABI and break the evaluator.  Its task-specific evaluator
    # is responsible for its own dependency set.
    if container_ref.startswith("ale/base-life-scib-atlas-integration:"):
        return
    if not container_ref.startswith(("ale/base-", "ghcr.io/")):
        return
    python = "/opt/cua-server/.venv/bin/python"
    rc, _, _ = await _run_docker(
        "exec", container_name, python, "-c", "import numpy",
    )
    if rc == 0:
        return
    logger.info("Installing NumPy in task container %s (image=%s)", container_name, container_ref)
    try:
        rc, _, stderr = await asyncio.wait_for(
            _run_docker(
                "exec", container_name, python, "-m", "pip", "install",
                "--disable-pip-version-check", "--no-input", "--quiet",
                "--only-binary=:all:", "numpy>=2.0,<3",
            ),
            timeout=120,
        )
    except TimeoutError as exc:
        raise RuntimeError(
            f"Timed out installing NumPy in task container {container_name}"
        ) from exc
    if rc != 0:
        raise RuntimeError(
            f"Failed to install NumPy in task container {container_name}: {stderr}"
        )
    rc, _, stderr = await _run_docker("exec", container_name, python, "-c", "import numpy")
    if rc != 0:
        raise RuntimeError(
            f"NumPy install verification failed in task container {container_name}: {stderr}"
        )


async def _get_image_provenance(
    container_name: str, config: DockerProviderConfig, container_ref: str,
) -> dict[str, Any]:
    rc, image_id, stderr = await _run_docker("inspect", "--format", "{{.Image}}", container_name)
    if rc != 0:
        raise RuntimeError(f"Failed to inspect container {container_name} image: {stderr}")
    image_id = image_id.strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise RuntimeError(f"Container {container_name} has invalid image ID {image_id!r}")
    rc, stdout, stderr = await _run_docker(
        "image", "inspect", "--format", "{{json .RepoDigests}}", image_id,
    )
    if rc != 0:
        raise RuntimeError(f"Failed to inspect Docker image {image_id} digests: {stderr}")
    repo_digests = json.loads(stdout) or []
    if not isinstance(repo_digests, list) or not all(isinstance(ref, str) for ref in repo_digests):
        raise RuntimeError(f"Docker image {image_id} has invalid RepoDigests")
    digests = {ref.rsplit("@", 1)[-1] for ref in repo_digests}
    selected_digest = container_ref.rsplit("@", 1)[-1] if "@" in container_ref else None
    if selected_digest is not None and selected_digest not in digests:
        raise RuntimeError(
            f"Docker selected image digest mismatch: expected {selected_digest}, "
            f"acquired {image_id} with RepoDigests {repo_digests}"
        )
    # A node may receive a complete rootfs reconstructed from a lazy image
    # export.  Such an image necessarily has a new local image ID, but the
    # importer stamps the original registry digest in this label so the
    # binding remains auditable without accepting an arbitrary mismatch.
    reconstructed_digest = None
    accepted_revision = config.image_revision in {image_id, *digests}
    if config.image_revision is not None and not accepted_revision:
        rc, labels_json, _ = await _run_docker(
            "image", "inspect", "--format", "{{json .Config.Labels}}", image_id,
        )
        if rc == 0:
            labels = json.loads(labels_json) or {}
            if isinstance(labels, dict):
                reconstructed_digest = labels.get("io.ale.reconstructed.source-digest")
        accepted_revision = reconstructed_digest == config.image_revision
    if config.image_revision is not None and not accepted_revision:
        raise RuntimeError(
            f"Docker image_revision mismatch: expected {config.image_revision}, "
            f"acquired {image_id} with RepoDigests {repo_digests}"
        )
    return {
        "provider": "docker",
        "image": config.image,
        "image_revision": selected_digest or config.image_revision or image_id,
        "image_ref": container_ref,
        "image_id": image_id,
        **({"reconstructed_from": reconstructed_digest} if reconstructed_digest else {}),
    }


async def _wait_cua_ready(
    cua_url: str,
    timeout: float = _CUA_READY_TIMEOUT,
    poll_interval: float = _CUA_READY_POLL_INTERVAL,
) -> bool:
    """Poll cua-server until it responds healthily."""
    from .gcloud import wait_cua_ready
    return await wait_cua_ready(
        cua_url, os_type="linux",
        timeout=timeout, poll_interval=poll_interval,
    )


def _sa_key_project_id(host_key_path: str) -> str:
    """project_id field of a GCS service-account JSON key (``""`` on error).

    Used as the billing/user project for requester-pays buckets: gsutil needs
    an explicit ``-u <project>`` (no boto-config equivalent works), and the
    injected SA key already names the project it can bill. Surfaced to
    data-staging via ``SandboxHandle.metadata["gcs_user_project"]``.
    """
    import json as _json
    from pathlib import Path as _P
    try:
        return _json.loads(_P(host_key_path).read_text()).get("project_id", "") or ""
    except Exception:  # noqa: BLE001 - missing billing metadata must not prevent startup
        return ""


class DockerProvider(Provider):
    """Provider backed by ``docker run`` / ``docker rm``."""

    def __init__(self, config: DockerProviderConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = _build_provider_config(config)
        self._cfg = config

    @property
    def config(self) -> DockerProviderConfig:
        return self._cfg

    async def acquire(self, spec: SandboxSpec) -> SandboxHandle:
        name = _generate_container_name(
            _CONTAINER_PREFIX,
            snapshot=spec.snapshot,
            task_id=spec.task_id,
            harness=spec.harness,
            model_tag=spec.model_tag,
        )

        # Everything image-specific comes from the Image entry named by `image`:
        # the container ref to boot, the in-container cua-server port (8000 on
        # ale-kasm, 5000 on the ubuntu22 export), and the sandbox paths.
        from ..images import get as get_image
        image = get_image(self._cfg.image)
        container_ref = self._cfg.image_ref or image.docker_image or _DEFAULT_CONTAINER_REF
        cua_internal_port = image.cua_server_port

        res = self._cfg.resolution or (1024, 768)
        run_args = [
            "run", "-d",
            "--name", name,
            f"--shm-size={self._cfg.shm_size}",
            "--init",
            "-e", "LANG=C.UTF-8",
            "-e", "LC_ALL=C.UTF-8",
            # Virtual display size for the in-container Xvfb (the entrypoint reads
            # this; falls back to its own default if unset).
            "-e", f"ALE_SCREEN_RESOLUTION={res[0]}x{res[1]}",
            # DinD is gated off in the entrypoint unless this is set (default off).
            "-e", f"ALE_ENABLE_DIND={'1' if self._cfg.enable_dind else '0'}",
            # The image entrypoint starts X and CUA; --init only reaps processes.
            "--entrypoint", image.docker_entrypoint,
        ]
        if self._cfg.network_mode == "host":
            run_args.extend(["--network", "host"])
        else:
            run_args.extend([
                "-p", f"127.0.0.1:0:{cua_internal_port}",
                "-p", f"127.0.0.1:0:{_VNC_INTERNAL_PORT}",
            ])
        if self._cfg.privileged:
            # Needed for tasks whose eval runs nested Docker; under rootless
            # docker the privileged container's root is still user-ns-bounded.
            run_args.append("--privileged")
        # Resource sizing: translate the task-card GCE machine type
        # (spec.machine_type, e.g. "c4-standard-4") into container --cpus/--memory
        # so a Docker sandbox gets the same shape gcloud would give the task. A
        # non-zero cpus / non-empty memory in the provider config overrides the
        # derived value (a host-protection cap under single-host concurrency).
        from .gcloud import _DEFAULT_CPU_MACHINE, _parse_gce_machine_type
        shape = _parse_gce_machine_type(spec.machine_type or _DEFAULT_CPU_MACHINE)
        cpus = (
            self._cfg.cpus
            or (float(spec.vcpus) if spec.vcpus else 0)
            or (float(shape.vcpus) if shape else 0)
        )
        memory = (
            self._cfg.memory
            or (f"{spec.memory_gb}g" if spec.memory_gb else "")
            or (f"{shape.memory_gb}g" if shape else "")
        )
        if cpus and cpus > 0:
            run_args.extend(["--cpus", str(cpus)])
        if memory:
            run_args.extend(["--memory", memory])
        if not self._cfg.allow_swap:
            if not memory:
                raise ValueError("docker allow_swap=false requires a finite memory limit")
            # 合计上限等于内存上限，容器就不能额外占用宿主机 swap。
            run_args.extend(["--memory-swap", memory])
        if self._cfg.lxcfs_root:
            run_args.extend(_lxcfs_mount_args(self._cfg.lxcfs_root))
        logger.info(
            "Container %s sizing: machine_type=%s -> cpus=%s memory=%s (res=%dx%d)",
            name, spec.machine_type or f"{_DEFAULT_CPU_MACHINE}(default)",
            cpus or "unset", memory or "unset", res[0], res[1],
        )
        run_args.append(container_ref)
        run_args.append("--wait")

        logger.info("Creating Docker container %s from %s", name, container_ref)
        stdout, cpu_set = await _start_container(name, run_args, cpus)
        logger.info("Container %s started (id=%s)", name, stdout[:12])

        try:
            await _ensure_task_numpy(name, container_ref)
            image_provenance = await _get_image_provenance(name, self._cfg, container_ref)
            if self._cfg.network_mode == "host":
                cua_port = cua_internal_port
                vnc_port = _VNC_INTERNAL_PORT
                cua_url = f"http://127.0.0.1:{cua_port}"
            else:
                cua_port = await _get_host_port(name, cua_internal_port)
                vnc_port = await _get_host_port(name, _VNC_INTERNAL_PORT)
                cua_url = f"http://127.0.0.1:{cua_port}"

            logger.info(
                "Container %s ports: cua=%d vnc=%d",
                name, cua_port, vnc_port,
            )

            ready = await _wait_cua_ready(cua_url)
            if not ready:
                raise RuntimeError(
                    f"CUA server at {cua_url} (container {name}) did not become ready"
                )

            resource_limits = await _get_resource_limits(name)
            if not self._cfg.allow_swap and (
                resource_limits["memory_and_swap_limit_bytes"]
                != resource_limits["memory_limit_bytes"]
            ):
                raise RuntimeError(f"Container {name} did not retain the requested swap limit")
            gcs_user_project = ""
            if self._cfg.gcs_sa_key:
                await self._inject_gcs_credentials(name, self._cfg.gcs_sa_key)
                gcs_user_project = _sa_key_project_id(self._cfg.gcs_sa_key)

            return SandboxHandle(
                id=name,
                endpoint=cua_url,
                os=image.os,
                **image.sandbox_paths(),
                metadata={
                    "provider": "docker",
                    "container_name": name,
                    "container_id": stdout[:12],
                    "cua_port": cua_port,
                    "vnc_port": vnc_port,
                    "image": self._cfg.image,
                    "network_mode": self._cfg.network_mode,
                    "image_provenance": image_provenance,
                    "snapshot": spec.snapshot,
                    "gcs_user_project": gcs_user_project,
                    "resource_limits": resource_limits,
                    "cpuset_cpus": cpu_set,
                    "lxcfs_root": self._cfg.lxcfs_root,
                },
            )
        except BaseException:
            try:
                cleanup_rc, _, cleanup_error = await _run_docker("rm", "-f", name)
                if cleanup_rc != 0:
                    logger.error("Failed to remove container %s: %s", name, cleanup_error)
            except Exception:
                logger.exception("Failed to remove container %s after acquisition failure", name)
            raise

    async def release(
        self, sandbox: SandboxHandle, *, mode: ReleaseMode = "delete",
    ) -> None:
        name = sandbox.id
        if mode == "delete":
            logger.info("Removing container %s", name)
            rc, _, stderr = await _run_docker("rm", "-f", name)
            if rc != 0:
                logger.error("Failed to remove container %s: %s", name, stderr)
        elif mode == "stop":
            logger.info("Stopping container %s", name)
            rc, _, stderr = await _run_docker("stop", name)
            if rc != 0:
                logger.error("Failed to stop container %s: %s", name, stderr)
        elif mode == "keep":
            logger.info("Container %s kept alive (mode=keep)", name)
        else:
            raise ValueError(f"unknown release mode: {mode!r}")

    @staticmethod
    async def _inject_gcs_credentials(container_name: str, host_key_path: str) -> None:
        """Copy SA key into container and write a boto config for gsutil."""
        from pathlib import Path as _P

        key_path = _P(host_key_path)
        if not key_path.exists():
            raise FileNotFoundError(f"gcs_sa_key not found: {host_key_path}")

        project_id = _sa_key_project_id(host_key_path)

        rc, _, stderr = await _run_docker(
            "exec", "-u", "root", container_name,
            "mkdir", "-p", str(_P(_GCS_KEY_CONTAINER_PATH).parent),
        )
        rc, _, stderr = await _run_docker(
            "cp", host_key_path, f"{container_name}:{_GCS_KEY_CONTAINER_PATH}",
        )
        if rc != 0:
            raise RuntimeError(f"Failed to copy SA key into {container_name}: {stderr}")
        await _run_docker(
            "exec", "-u", "root", container_name,
            "chmod", "644", _GCS_KEY_CONTAINER_PATH,
        )

        boto_content = (
            "[Credentials]\n"
            f"gs_service_key_file = {_GCS_KEY_CONTAINER_PATH}\n\n"
            "[Boto]\n"
            "https_validate_certificates = True\n\n"
            "[GSUtil]\n"
            f"default_project_id = {project_id}\n"
        )
        # Write a system-wide /etc/boto.cfg (as root) rather than a per-user
        # ~/.boto: the container user is image-specific (kasm-user, user, ...),
        # and hardcoding one user's home silently broke gsutil auth on every
        # other image family. /etc/boto.cfg is read regardless of $HOME.
        rc, _, stderr = await _run_docker(
            "exec", "-u", "root", container_name,
            "bash", "-c",
            f"cat > {_BOTO_CONFIG_CONTAINER_PATH} << 'BOTOEOF'\n{boto_content}BOTOEOF",
        )
        if rc != 0:
            raise RuntimeError(
                f"Failed to write {_BOTO_CONFIG_CONTAINER_PATH} into "
                f"{container_name}: {stderr}"
            )
        logger.info("Injected GCS credentials into %s", container_name)

    def open_session(self, sandbox: SandboxHandle) -> Any:
        from cua_bench.computers.remote import RemoteDesktopSession

        from .gcloud import _init_computer_skip_wait

        session = RemoteDesktopSession(
            api_url=sandbox.endpoint,
            os_type=sandbox.os,
        )
        _init_computer_skip_wait(session)
        return session
