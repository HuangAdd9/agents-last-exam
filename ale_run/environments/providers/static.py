"""StaticProvider — attaches to a fixed VM by its cua-server endpoint.

By default this retains the historical attach-only behaviour.  A static VM
hosted by Proxmox can additionally opt into a fixed-snapshot lifecycle:
``acquire`` can restore an explicitly named offline baseline, starts the VM,
then waits for its fixed CUA URL.  ``release`` keeps the VM running by default.

Used for:

- **Image baking**: bring up a VM by hand, iterate on deployer code without
  paying the 3-5 min boot cost on every run.
- **Dev debug**: poke at a long-running staging VM, reproduce stuck cases.
- **Tests against a fixed scratch VM**: stable IP, deterministic state.
"""

from __future__ import annotations

import asyncio
import dataclasses
import fcntl
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import quote

from ...base_interface import Provider, ReleaseMode, SandboxHandle, SandboxSpec

logger = logging.getLogger(__name__)


# ======================================================================
# Config
# ======================================================================


@dataclasses.dataclass(frozen=True)
class StaticProxmoxPowerConfig:
    """Proxmox lifecycle configuration for one fixed VM."""

    api_url: str
    token_id: str
    token_secret: str
    node: str
    vmid: int
    verify_ssl: bool | str = True
    request_timeout_s: float = 30.0
    task_timeout_s: float = 300.0
    poll_interval_s: float = 2.0
    baseline_snapshot: str = ""
    rollback_before_acquire: bool = False
    release_action: str = "keep"
    lock_timeout_s: float = 0.0


@dataclasses.dataclass(frozen=True)
class StaticProviderConfig:
    """Pin to a pre-existing VM."""

    endpoint: str
    """Full cua-server URL, e.g. ``http://1.2.3.4:5000``."""

    image: str = "ale-ubuntu22"
    """Image family the static VM was built from. Determines the
    SandboxHandle's baked paths. Must be registered in
    :mod:`ale_run.environments.images`."""

    vm_id: str = "static"
    """Informational id — shows up in logs / run.json."""

    cleanup_on_release: bool = False
    """If True, run :attr:`cleanup_script` on the VM during ``release``.
    The VM itself is never destroyed."""

    cleanup_script: str | None = None
    """Shell snippet executed via ``sandbox.run_command`` on release.
    Only runs when ``cleanup_on_release`` is True."""

    ready_timeout_s: float = 15.0
    """How long acquire waits for the fixed VM's CUA endpoint."""

    poll_interval_s: float = 2.0
    """Delay between CUA health probes during acquire."""

    proxmox: StaticProxmoxPowerConfig | None = None
    """Optional Proxmox power/snapshot configuration for this fixed VM."""


def _build_proxmox_power_config(raw: Any) -> StaticProxmoxPowerConfig | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError("static proxmox configuration must be a mapping")

    from .proxmox import _normalize_api_url

    api_url = _normalize_api_url(str(raw.get("api_url") or ""))
    token_id = str(raw.get("token_id") or "").strip()
    token_secret = str(raw.get("token_secret") or "").strip()
    node = str(raw.get("node") or "").strip()
    vmid = int(raw.get("vmid") or 0)
    if not token_id or "!" not in token_id:
        raise ValueError("static proxmox token_id must use user@realm!token-name")
    if not token_secret:
        raise KeyError("static proxmox configuration missing token_secret")
    if not node:
        raise KeyError("static proxmox configuration missing node")
    if vmid <= 0:
        raise ValueError("static proxmox vmid must be a positive integer")

    verify_ssl = raw.get("verify_ssl", True)
    if not isinstance(verify_ssl, (bool, str)):
        raise TypeError("static proxmox verify_ssl must be a boolean or CA bundle path")
    request_timeout_s = float(raw.get("request_timeout_s") or 30.0)
    task_timeout_s = float(raw.get("task_timeout_s") or 300.0)
    poll_interval_s = float(raw.get("poll_interval_s") or 2.0)
    lock_timeout_s = float(raw.get("lock_timeout_s", 0.0))
    if request_timeout_s <= 0 or task_timeout_s <= 0 or poll_interval_s <= 0:
        raise ValueError("static proxmox timeouts and poll interval must be positive")
    if lock_timeout_s < 0:
        raise ValueError("static proxmox lock_timeout_s must be non-negative")

    baseline_snapshot = str(raw.get("baseline_snapshot") or "").strip()
    rollback_before_acquire = bool(raw.get("rollback_before_acquire", False))
    release_action = str(raw.get("release_action") or "keep").strip().lower()
    if release_action not in {"keep", "shutdown"}:
        raise ValueError(
            "static proxmox release_action must be keep or shutdown"
        )
    if rollback_before_acquire and not baseline_snapshot:
        raise ValueError(
            "static proxmox baseline_snapshot is required when rollback is enabled"
        )
    if rollback_before_acquire and baseline_snapshot == "current":
        raise ValueError("static proxmox baseline_snapshot cannot be 'current'")

    return StaticProxmoxPowerConfig(
        api_url=api_url,
        token_id=token_id,
        token_secret=token_secret,
        node=node,
        vmid=vmid,
        verify_ssl=verify_ssl,
        request_timeout_s=request_timeout_s,
        task_timeout_s=task_timeout_s,
        poll_interval_s=poll_interval_s,
        baseline_snapshot=baseline_snapshot,
        rollback_before_acquire=rollback_before_acquire,
        release_action=release_action,
        lock_timeout_s=lock_timeout_s,
    )


def _build_provider_config(raw: dict[str, Any]) -> StaticProviderConfig:
    # back-compat: yaml ``os: linux`` (old) maps to the linux default image.
    image = raw.get("image")
    if image is None:
        os_legacy = raw.get("os") or "linux"
        image = "ale-ubuntu22" if os_legacy == "linux" else "ale-win10"
    ready_timeout_s = float(raw.get("ready_timeout_s") or 15.0)
    poll_interval_s = float(raw.get("poll_interval_s") or 2.0)
    if ready_timeout_s <= 0 or poll_interval_s <= 0:
        raise ValueError("static ready_timeout_s and poll_interval_s must be positive")
    return StaticProviderConfig(
        endpoint=str(raw["endpoint"]),
        image=str(image),
        vm_id=str(raw.get("vm_id") or "static"),
        cleanup_on_release=bool(raw.get("cleanup_on_release", False)),
        cleanup_script=raw.get("cleanup_script"),
        ready_timeout_s=ready_timeout_s,
        poll_interval_s=poll_interval_s,
        proxmox=_build_proxmox_power_config(raw.get("proxmox")),
    )


# ======================================================================
# Provider
# ======================================================================


class StaticProvider(Provider):
    """Provider impl that skips ``gcloud create / delete``."""

    def __init__(self, config: StaticProviderConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = _build_provider_config(config)
        self._cfg = config
        self._proxmox_api: Any | None = None
        self._vm_lock_file: TextIO | None = None
        if config.proxmox is not None:
            from .proxmox import _ProxmoxApi

            power = config.proxmox
            self._proxmox_api = _ProxmoxApi(
                api_url=power.api_url,
                token_id=power.token_id,
                token_secret=power.token_secret,
                verify_ssl=power.verify_ssl,
                timeout_s=power.request_timeout_s,
            )

    def _vm_lock_path(self) -> Path:
        power = self._cfg.proxmox
        if power is None:
            raise RuntimeError("static Proxmox lifecycle is not configured")
        safe_node = re.sub(r"[^A-Za-z0-9_.-]", "_", power.node)
        return Path(tempfile.gettempdir()) / (
            f"ale-static-proxmox-{safe_node}-{power.vmid}.lock"
        )

    async def _acquire_vm_lock(self) -> None:
        """Hold an OS-released cross-process lock for this fixed VM."""
        power = self._cfg.proxmox
        if power is None:
            return
        if self._vm_lock_file is not None:
            raise RuntimeError(f"Static Proxmox VM {power.vmid} is already acquired")

        path = self._vm_lock_path()
        lock_file = path.open("a+", encoding="utf-8")
        deadline = time.monotonic() + power.lock_timeout_s
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                lock_file.seek(0)
                lock_file.truncate()
                lock_file.write(f"pid={os.getpid()}\n")
                lock_file.flush()
                self._vm_lock_file = lock_file
                logger.info(
                    "Acquired fixed-VM lock for Proxmox VM %s at %s",
                    power.vmid,
                    path,
                )
                return
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    lock_file.close()
                    raise RuntimeError(
                        f"Static Proxmox VM {power.vmid} is already in use; "
                        f"could not acquire {path} within {power.lock_timeout_s:g}s"
                    )
                await asyncio.sleep(min(power.poll_interval_s, 0.5))

    def _release_vm_lock(self) -> None:
        lock_file = self._vm_lock_file
        self._vm_lock_file = None
        if lock_file is None:
            return
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    async def _proxmox_request(
        self,
        method: str,
        path: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> Any:
        if self._proxmox_api is None:
            raise RuntimeError("static Proxmox power management is not configured")
        return await asyncio.to_thread(
            self._proxmox_api.request,
            method,
            path,
            data=data,
        )

    async def _wait_proxmox_task(self, upid: str, *, action: str) -> None:
        power = self._cfg.proxmox
        if power is None:
            return
        deadline = time.monotonic() + power.task_timeout_s
        path = (
            f"/nodes/{quote(power.node, safe='')}/tasks/"
            f"{quote(upid, safe='')}/status"
        )
        while True:
            status = await self._proxmox_request("GET", path)
            if isinstance(status, dict) and status.get("status") == "stopped":
                exit_status = str(status.get("exitstatus") or "")
                if exit_status != "OK":
                    raise RuntimeError(
                        f"Proxmox {action} task for VM {power.vmid} failed: "
                        f"{exit_status or status}"
                    )
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Proxmox {action} task for VM {power.vmid} did not finish "
                    f"within {power.task_timeout_s:g}s"
                )
            await asyncio.sleep(power.poll_interval_s)

    async def _vm_state(self) -> str:
        power = self._cfg.proxmox
        if power is None:
            raise RuntimeError("static Proxmox lifecycle is not configured")
        node = quote(power.node, safe="")
        status = await self._proxmox_request(
            "GET", f"/nodes/{node}/qemu/{power.vmid}/status/current"
        )
        state = str(status.get("status") or "") if isinstance(status, dict) else ""
        if state not in {"running", "stopped"}:
            raise RuntimeError(
                f"Static Proxmox VM {power.vmid} has unexpected state {state!r}"
            )
        return state

    async def _ensure_powered_off(self) -> bool:
        """Gracefully shut down the fixed VM; return whether we submitted it."""
        power = self._cfg.proxmox
        if power is None:
            return False
        if await self._vm_state() == "stopped":
            return False
        node = quote(power.node, safe="")
        logger.info("Shutting down static Proxmox VM %s", power.vmid)
        upid = await self._proxmox_request(
            "POST",
            f"/nodes/{node}/qemu/{power.vmid}/status/shutdown",
            data={"timeout": max(1, int(power.task_timeout_s))},
        )
        if not isinstance(upid, str) or not upid:
            raise RuntimeError(
                f"Proxmox returned no task id while shutting down static VM {power.vmid}"
            )
        await self._wait_proxmox_task(upid, action="shutdown")
        if await self._vm_state() != "stopped":
            raise RuntimeError(f"Static Proxmox VM {power.vmid} did not stop")
        return True

    async def _rollback_baseline(self) -> None:
        """Restore the configured baseline. Caller must have stopped the VM."""
        power = self._cfg.proxmox
        if power is None or not power.baseline_snapshot:
            raise RuntimeError("static Proxmox baseline snapshot is not configured")
        if await self._vm_state() != "stopped":
            raise RuntimeError(
                f"Static Proxmox VM {power.vmid} must be stopped before rollback"
            )
        node = quote(power.node, safe="")
        snap = quote(power.baseline_snapshot, safe="")
        logger.info(
            "Rolling static Proxmox VM %s back to snapshot %s",
            power.vmid,
            power.baseline_snapshot,
        )
        try:
            upid = await self._proxmox_request(
                "POST",
                f"/nodes/{node}/qemu/{power.vmid}/snapshot/{snap}/rollback",
            )
        except Exception as exc:
            if getattr(exc, "status_code", None) == 403:
                raise RuntimeError(
                    f"Proxmox denied rollback of VM {power.vmid}; the token needs "
                    "VM.Snapshot.Rollback and VM.PowerMgmt on this VM"
                ) from exc
            raise RuntimeError(
                f"Proxmox rollback request failed for VM {power.vmid} snapshot "
                f"{power.baseline_snapshot!r}: {exc}"
            ) from exc
        if not isinstance(upid, str) or not upid:
            raise RuntimeError(
                f"Proxmox returned no task id while rolling back static VM {power.vmid}"
            )
        await self._wait_proxmox_task(upid, action="snapshot rollback")

    async def _validate_baseline_snapshot(self) -> None:
        """Fail before shutdown unless the exact offline baseline exists."""
        power = self._cfg.proxmox
        if power is None or not power.baseline_snapshot:
            raise RuntimeError("static Proxmox baseline snapshot is not configured")
        node = quote(power.node, safe="")
        try:
            snapshots = await self._proxmox_request(
                "GET", f"/nodes/{node}/qemu/{power.vmid}/snapshot"
            )
        except Exception as exc:
            if getattr(exc, "status_code", None) == 403:
                raise RuntimeError(
                    f"Proxmox denied snapshot inspection for VM {power.vmid}; "
                    "the token needs VM.Audit on this VM"
                ) from exc
            raise
        if not isinstance(snapshots, list):
            raise TypeError(
                f"Proxmox returned an invalid snapshot list for VM {power.vmid}"
            )
        matches = [
            item
            for item in snapshots
            if isinstance(item, dict) and item.get("name") == power.baseline_snapshot
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Baseline snapshot {power.baseline_snapshot!r} was not found "
                f"exactly once on Proxmox VM {power.vmid}"
            )
        vmstate = matches[0].get("vmstate")
        if vmstate not in {None, "", 0, "0"}:
            raise RuntimeError(
                f"Baseline snapshot {power.baseline_snapshot!r} for VM "
                f"{power.vmid} contains RAM state; an offline baseline is required"
            )

    async def _restore_baseline_before_acquire(self) -> bool:
        power = self._cfg.proxmox
        if power is None or not power.rollback_before_acquire:
            return False
        await self._validate_baseline_snapshot()
        await self._ensure_powered_off()
        await self._rollback_baseline()
        return True

    async def _ensure_powered_on(self) -> bool:
        """Start the configured fixed VM if needed; return whether we started it."""
        power = self._cfg.proxmox
        if power is None:
            return False
        node = quote(power.node, safe="")
        state = await self._vm_state()
        if state == "running":
            logger.info("Static Proxmox VM %s is already running", power.vmid)
            return False
        logger.info("Starting static Proxmox VM %s on node %s", power.vmid, power.node)
        upid = await self._proxmox_request(
            "POST",
            f"/nodes/{node}/qemu/{power.vmid}/status/start",
        )
        if not isinstance(upid, str) or not upid:
            raise RuntimeError(
                f"Proxmox returned no task id while starting static VM {power.vmid}"
            )
        await self._wait_proxmox_task(upid, action="start")
        return True

    async def acquire(self, spec: SandboxSpec) -> SandboxHandle:
        from ..images import get as get_image

        await self._acquire_vm_lock()
        try:
            baseline_restored = await self._restore_baseline_before_acquire()
            started_by_provider = await self._ensure_powered_on()
            image = get_image(self._cfg.image)
            sandbox = SandboxHandle(
                id=self._cfg.vm_id,
                endpoint=self._cfg.endpoint,
                os=image.os,
                **image.sandbox_paths(),
                metadata={
                    "provider": "static",
                    "static": True,
                    "snapshot": spec.snapshot,
                    "image": image.name,
                    "proxmox_auto_start": self._cfg.proxmox is not None,
                    "started_by_provider": started_by_provider,
                    "baseline_restored": baseline_restored,
                },
            )
            deadline = time.monotonic() + self._cfg.ready_timeout_s
            last_error: Exception | None = None
            while time.monotonic() < deadline:
                try:
                    await sandbox.check_reachable(label=f"static VM {self._cfg.vm_id}")
                    return sandbox
                except Exception as exc:  # noqa: BLE001 - retry reachability failures
                    last_error = exc
                    await asyncio.sleep(self._cfg.poll_interval_s)
            raise RuntimeError(
                f"static VM {self._cfg.vm_id} CUA is not reachable at "
                f"{self._cfg.endpoint} after {self._cfg.ready_timeout_s:g}s; "
                "check VM boot and cua-server startup logs"
            ) from last_error
        except BaseException:
            self._release_vm_lock()
            raise

    async def release(self, sandbox: SandboxHandle, *, mode: ReleaseMode = "keep") -> None:
        try:
            if self._cfg.cleanup_on_release and self._cfg.cleanup_script:
                try:
                    await sandbox.run_command(self._cfg.cleanup_script, timeout=120)
                except Exception as e:  # noqa: BLE001 - release cleanup is best effort
                    logger.warning("StaticProvider cleanup failed: %s", e)

            power = self._cfg.proxmox
            if power is None or power.release_action == "keep":
                return
            if power.release_action == "shutdown":
                await self._ensure_powered_off()
        finally:
            self._release_vm_lock()


    def open_session(self, vm: SandboxHandle) -> Any:
        from cua_bench.computers.remote import RemoteDesktopSession

        from .gcloud import _init_computer_skip_wait

        session = RemoteDesktopSession(api_url=vm.endpoint, os_type=vm.os)
        _init_computer_skip_wait(session)
        return session
