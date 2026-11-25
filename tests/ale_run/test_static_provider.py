from __future__ import annotations

from typing import Any

import pytest

from ale_run.base_interface import SandboxHandle, SandboxSpec
from ale_run.environments.providers.static import StaticProvider


def _config() -> dict[str, Any]:
    return {
        "endpoint": "http://192.168.0.35:5000",
        "image": "ale-win10",
        "vm_id": "proxmox-vm-101",
        "ready_timeout_s": 1,
        "poll_interval_s": 0.001,
        "proxmox": {
            "api_url": "https://pve.example:8006",
            "token_id": "ale@pve!rollout",
            "token_secret": "test-secret",
            "verify_ssl": False,
            "node": "seed3",
            "vmid": 101,
            "task_timeout_s": 1,
            "poll_interval_s": 0.001,
        },
    }


@pytest.mark.asyncio
async def test_static_provider_starts_stopped_proxmox_vm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = StaticProvider(_config())
    calls: list[tuple[str, str]] = []

    async def request(
        method: str,
        path: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> Any:
        _ = data
        calls.append((method, path))
        if path.endswith("/status/current"):
            return {"status": "stopped"}
        if path.endswith("/status/start"):
            return "UPID:seed3:start"
        if "/tasks/" in path:
            return {"status": "stopped", "exitstatus": "OK"}
        raise AssertionError(f"unexpected request: {method} {path}")

    async def reachable(self: SandboxHandle, *, label: str = "sandbox") -> None:
        _ = (self, label)

    monkeypatch.setattr(provider, "_proxmox_request", request)
    monkeypatch.setattr(SandboxHandle, "check_reachable", reachable)

    sandbox = await provider.acquire(SandboxSpec(snapshot="cpu-free"))

    assert sandbox.metadata["started_by_provider"] is True
    assert calls == [
        ("GET", "/nodes/seed3/qemu/101/status/current"),
        ("POST", "/nodes/seed3/qemu/101/status/start"),
        ("GET", "/nodes/seed3/tasks/UPID%3Aseed3%3Astart/status"),
    ]
    await provider.release(sandbox)


@pytest.mark.asyncio
async def test_static_provider_does_not_restart_running_vm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = StaticProvider(_config())
    calls: list[tuple[str, str]] = []

    async def request(
        method: str,
        path: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> Any:
        _ = data
        calls.append((method, path))
        return {"status": "running"}

    async def reachable(self: SandboxHandle, *, label: str = "sandbox") -> None:
        _ = (self, label)

    monkeypatch.setattr(provider, "_proxmox_request", request)
    monkeypatch.setattr(SandboxHandle, "check_reachable", reachable)

    sandbox = await provider.acquire(SandboxSpec(snapshot="cpu-free"))

    assert sandbox.metadata["started_by_provider"] is False
    assert calls == [("GET", "/nodes/seed3/qemu/101/status/current")]
    await provider.release(sandbox)


@pytest.mark.asyncio
async def test_static_provider_rolls_back_before_starting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config["proxmox"].update(
        baseline_snapshot="clean-licensed-baseline",
        rollback_before_acquire=True,
    )
    provider = StaticProvider(config)
    calls: list[tuple[str, str]] = []

    async def request(
        method: str,
        path: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> Any:
        _ = data
        calls.append((method, path))
        if path.endswith("/snapshot"):
            return [{"name": "clean-licensed-baseline", "vmstate": 0}]
        if path.endswith("/status/current"):
            return {"status": "stopped"}
        if path.endswith("/rollback"):
            return "UPID:seed3:rollback"
        if path.endswith("/status/start"):
            return "UPID:seed3:start"
        if "/tasks/" in path:
            return {"status": "stopped", "exitstatus": "OK"}
        raise AssertionError(f"unexpected request: {method} {path}")

    async def reachable(self: SandboxHandle, *, label: str = "sandbox") -> None:
        _ = (self, label)

    monkeypatch.setattr(provider, "_proxmox_request", request)
    monkeypatch.setattr(SandboxHandle, "check_reachable", reachable)

    sandbox = await provider.acquire(SandboxSpec(snapshot="gpu-free"))

    assert sandbox.metadata["baseline_restored"] is True
    assert calls == [
        ("GET", "/nodes/seed3/qemu/101/snapshot"),
        ("GET", "/nodes/seed3/qemu/101/status/current"),
        ("GET", "/nodes/seed3/qemu/101/status/current"),
        (
            "POST",
            "/nodes/seed3/qemu/101/snapshot/clean-licensed-baseline/rollback",
        ),
        ("GET", "/nodes/seed3/tasks/UPID%3Aseed3%3Arollback/status"),
        ("GET", "/nodes/seed3/qemu/101/status/current"),
        ("POST", "/nodes/seed3/qemu/101/status/start"),
        ("GET", "/nodes/seed3/tasks/UPID%3Aseed3%3Astart/status"),
    ]
    await provider.release(sandbox)


@pytest.mark.asyncio
async def test_static_provider_stops_running_vm_before_acquire_rollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config["proxmox"].update(
        baseline_snapshot="clean baseline",
        rollback_before_acquire=True,
    )
    provider = StaticProvider(config)
    calls: list[tuple[str, str]] = []
    states = iter(["running", "stopped", "stopped", "stopped"])

    async def request(
        method: str,
        path: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> Any:
        _ = data
        calls.append((method, path))
        if path.endswith("/snapshot"):
            return [{"name": "clean baseline", "vmstate": 0}]
        if path.endswith("/status/current"):
            return {"status": next(states)}
        if path.endswith("/status/shutdown"):
            return "UPID:seed3:shutdown"
        if path.endswith("/rollback"):
            return "UPID:seed3:rollback"
        if path.endswith("/status/start"):
            return "UPID:seed3:start"
        if "/tasks/" in path:
            return {"status": "stopped", "exitstatus": "OK"}
        raise AssertionError(f"unexpected request: {method} {path}")

    async def reachable(self: SandboxHandle, *, label: str = "sandbox") -> None:
        _ = (self, label)

    monkeypatch.setattr(provider, "_proxmox_request", request)
    monkeypatch.setattr(SandboxHandle, "check_reachable", reachable)

    sandbox = await provider.acquire(SandboxSpec(snapshot="gpu-free"))

    assert ("POST", "/nodes/seed3/qemu/101/status/shutdown") in calls
    assert (
        "POST",
        "/nodes/seed3/qemu/101/snapshot/clean%20baseline/rollback",
    ) in calls
    await provider.release(sandbox)


def test_static_provider_requires_named_baseline_for_rollback() -> None:
    config = _config()
    config["proxmox"]["rollback_before_acquire"] = True

    with pytest.raises(ValueError, match="baseline_snapshot is required"):
        StaticProvider(config)

    config["proxmox"]["baseline_snapshot"] = "current"
    with pytest.raises(ValueError, match="cannot be 'current'"):
        StaticProvider(config)


@pytest.mark.asyncio
async def test_static_provider_fixed_vm_lock_is_fail_fast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = StaticProvider(_config())
    second = StaticProvider(_config())

    async def running(
        method: str, path: str, *, data: dict[str, Any] | None = None,
    ) -> Any:
        _ = (method, path, data)
        return {"status": "running"}

    async def reachable(self: SandboxHandle, *, label: str = "sandbox") -> None:
        _ = (self, label)

    monkeypatch.setattr(first, "_proxmox_request", running)
    monkeypatch.setattr(second, "_proxmox_request", running)
    monkeypatch.setattr(SandboxHandle, "check_reachable", reachable)

    sandbox = await first.acquire(SandboxSpec(snapshot="cpu-free"))
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            await second.acquire(SandboxSpec(snapshot="cpu-free"))
    finally:
        await first.release(sandbox)


@pytest.mark.asyncio
async def test_static_provider_acquire_failure_releases_fixed_vm_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config["ready_timeout_s"] = 0.003
    first = StaticProvider(config)
    second = StaticProvider(config)

    async def running(
        method: str, path: str, *, data: dict[str, Any] | None = None,
    ) -> Any:
        _ = (method, path, data)
        return {"status": "running"}

    async def unreachable(
        self: SandboxHandle, *, label: str = "sandbox",
    ) -> None:
        _ = (self, label)
        raise RuntimeError("CUA down")

    monkeypatch.setattr(first, "_proxmox_request", running)
    monkeypatch.setattr(second, "_proxmox_request", running)
    monkeypatch.setattr(SandboxHandle, "check_reachable", unreachable)
    with pytest.raises(RuntimeError, match="CUA is not reachable"):
        await first.acquire(SandboxSpec(snapshot="cpu-free"))

    async def reachable(self: SandboxHandle, *, label: str = "sandbox") -> None:
        _ = (self, label)

    monkeypatch.setattr(SandboxHandle, "check_reachable", reachable)
    sandbox = await second.acquire(SandboxSpec(snapshot="cpu-free"))
    await second.release(sandbox)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshots", "message"),
    [
        ([{"name": "different", "vmstate": 0}], "was not found exactly once"),
        (
            [{"name": "clean-baseline", "vmstate": "local-lvm:vmstate"}],
            "contains RAM state",
        ),
    ],
)
async def test_static_provider_rejects_missing_or_stateful_baseline(
    monkeypatch: pytest.MonkeyPatch,
    snapshots: list[dict[str, Any]],
    message: str,
) -> None:
    config = _config()
    config["proxmox"].update(
        baseline_snapshot="clean-baseline",
        rollback_before_acquire=True,
    )
    provider = StaticProvider(config)
    calls: list[tuple[str, str]] = []

    async def request(
        method: str, path: str, *, data: dict[str, Any] | None = None,
    ) -> Any:
        _ = data
        calls.append((method, path))
        if path.endswith("/snapshot"):
            return snapshots
        raise AssertionError("validation must fail before changing VM power or disks")

    monkeypatch.setattr(provider, "_proxmox_request", request)
    with pytest.raises(RuntimeError, match=message):
        await provider.acquire(SandboxSpec(snapshot="gpu-free"))
    assert calls == [("GET", "/nodes/seed3/qemu/101/snapshot")]


@pytest.mark.asyncio
async def test_static_provider_release_can_gracefully_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config["proxmox"]["release_action"] = "shutdown"
    provider = StaticProvider(config)
    calls: list[tuple[str, str, dict[str, Any] | None]] = []
    states = iter(["running", "running", "stopped"])

    async def request(
        method: str, path: str, *, data: dict[str, Any] | None = None,
    ) -> Any:
        calls.append((method, path, data))
        if path.endswith("/status/current"):
            return {"status": next(states)}
        if path.endswith("/status/shutdown"):
            return "UPID:seed3:shutdown"
        if "/tasks/" in path:
            return {"status": "stopped", "exitstatus": "OK"}
        raise AssertionError(f"unexpected request: {method} {path}")

    async def reachable(self: SandboxHandle, *, label: str = "sandbox") -> None:
        _ = (self, label)

    monkeypatch.setattr(provider, "_proxmox_request", request)
    monkeypatch.setattr(SandboxHandle, "check_reachable", reachable)
    sandbox = await provider.acquire(SandboxSpec(snapshot="cpu-free"))
    await provider.release(sandbox)

    assert (
        "POST",
        "/nodes/seed3/qemu/101/status/shutdown",
        {"timeout": 1},
    ) in calls


@pytest.mark.asyncio
async def test_static_provider_rejects_missing_baseline_before_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config["proxmox"].update(
        baseline_snapshot="missing-baseline",
        rollback_before_acquire=True,
    )
    provider = StaticProvider(config)
    calls: list[tuple[str, str]] = []

    async def request(
        method: str,
        path: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> Any:
        _ = data
        calls.append((method, path))
        if path.endswith("/snapshot"):
            return [{"name": "some-other-baseline", "vmstate": 0}]
        raise AssertionError("VM state must not be changed for a missing baseline")

    monkeypatch.setattr(provider, "_proxmox_request", request)

    with pytest.raises(RuntimeError, match="was not found exactly once"):
        await provider.acquire(SandboxSpec(snapshot="gpu-free"))

    assert calls == [("GET", "/nodes/seed3/qemu/101/snapshot")]


@pytest.mark.asyncio
async def test_static_provider_rejects_ram_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    config["proxmox"].update(
        baseline_snapshot="suspended-baseline",
        rollback_before_acquire=True,
    )
    provider = StaticProvider(config)

    async def request(
        method: str,
        path: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> Any:
        _ = (method, data)
        if path.endswith("/snapshot"):
            return [{"name": "suspended-baseline", "vmstate": 1}]
        raise AssertionError("RAM snapshot validation must fail before shutdown")

    monkeypatch.setattr(provider, "_proxmox_request", request)

    with pytest.raises(RuntimeError, match="contains RAM state"):
        await provider.acquire(SandboxSpec(snapshot="gpu-free"))


@pytest.mark.asyncio
async def test_static_provider_lock_prevents_concurrent_fixed_vm_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = StaticProvider(_config())
    second = StaticProvider(_config())

    async def request(
        method: str,
        path: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> Any:
        _ = (method, path, data)
        return {"status": "running"}

    async def reachable(self: SandboxHandle, *, label: str = "sandbox") -> None:
        _ = (self, label)

    monkeypatch.setattr(first, "_proxmox_request", request)
    monkeypatch.setattr(second, "_proxmox_request", request)
    monkeypatch.setattr(SandboxHandle, "check_reachable", reachable)

    first_sandbox = await first.acquire(SandboxSpec(snapshot="gpu-free"))
    with pytest.raises(RuntimeError, match="already in use"):
        await second.acquire(SandboxSpec(snapshot="gpu-free"))

    await first.release(first_sandbox)
    second_sandbox = await second.acquire(SandboxSpec(snapshot="gpu-free"))
    await second.release(second_sandbox)


@pytest.mark.asyncio
async def test_static_provider_releases_lock_when_acquire_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failed = StaticProvider(_config())
    successor = StaticProvider(_config())

    async def fail_request(
        method: str,
        path: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> Any:
        _ = (method, path, data)
        raise RuntimeError("simulated API failure")

    async def running_request(
        method: str,
        path: str,
        *,
        data: dict[str, Any] | None = None,
    ) -> Any:
        _ = (method, path, data)
        return {"status": "running"}

    async def reachable(self: SandboxHandle, *, label: str = "sandbox") -> None:
        _ = (self, label)

    monkeypatch.setattr(failed, "_proxmox_request", fail_request)
    monkeypatch.setattr(successor, "_proxmox_request", running_request)
    monkeypatch.setattr(SandboxHandle, "check_reachable", reachable)

    with pytest.raises(RuntimeError, match="simulated API failure"):
        await failed.acquire(SandboxSpec(snapshot="gpu-free"))

    sandbox = await successor.acquire(SandboxSpec(snapshot="gpu-free"))
    await successor.release(sandbox)
