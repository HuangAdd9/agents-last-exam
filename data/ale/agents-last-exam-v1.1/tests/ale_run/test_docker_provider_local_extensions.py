from __future__ import annotations

import json
from pathlib import Path

import pytest

from ale_run.base_interface import SandboxSpec
from ale_run.environments.providers import docker as docker_module
from ale_run.environments.providers.docker import DockerProvider, DockerProviderConfig
from ale_run.orchestration.config_loader import load_experiment


@pytest.fixture(autouse=True)
def mock_resource_limits(monkeypatch):
    from unittest.mock import AsyncMock

    monkeypatch.setattr(docker_module, "_get_resource_limits", AsyncMock(return_value={
        "cpu_quota": 4.0, "memory_limit_bytes": 15 * 1024 ** 3,
    }))


def test_docker_network_mode_defaults_to_bridge() -> None:
    provider = DockerProvider({"image": "ale-ubuntu22-docker"})
    assert provider.config.network_mode == "bridge"
    assert provider.config.apparmor_unconfined is False
    assert provider.config.gpus == ""


async def test_apparmor_unconfined_is_explicit(monkeypatch) -> None:
    docker_calls: list[tuple[str, ...]] = []

    async def fake_run_docker(*args: str) -> tuple[int, str, str]:
        docker_calls.append(args)
        if args[:2] == ("image", "inspect"):
            return 0, "sha256:" + "b" * 64, ""
        return 0, "0123456789abcdef", ""

    async def fake_get_host_port(container_name: str, internal_port: int) -> int:
        return {5000: 15000, 6901: 16901}[internal_port]

    async def fake_wait_cua_ready(url: str, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(docker_module, "_run_docker", fake_run_docker)
    monkeypatch.setattr(docker_module, "_get_host_port", fake_get_host_port)
    monkeypatch.setattr(docker_module, "_wait_cua_ready", fake_wait_cua_ready)
    provider = DockerProvider({
        "image": "ale-ubuntu22-docker",
        "apparmor_unconfined": True,
    })

    sandbox = await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu"))

    run_args = docker_calls[0]
    security_index = run_args.index("--security-opt")
    assert run_args[security_index:security_index + 2] == (
        "--security-opt", "apparmor=unconfined",
    )
    assert "--privileged" not in run_args
    assert sandbox.metadata["apparmor_unconfined"] is True


async def test_gpu_request_is_snapshot_scoped(monkeypatch) -> None:
    docker_calls: list[tuple[str, ...]] = []

    async def fake_run_docker(*args: str) -> tuple[int, str, str]:
        docker_calls.append(args)
        return 0, "0123456789abcdef", ""

    async def fake_get_host_port(container_name: str, internal_port: int) -> int:
        return {5000: 15000, 6901: 16901}[internal_port]

    async def fake_wait_cua_ready(url: str, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(docker_module, "_run_docker", fake_run_docker)
    monkeypatch.setattr(docker_module, "_get_host_port", fake_get_host_port)
    monkeypatch.setattr(docker_module, "_wait_cua_ready", fake_wait_cua_ready)
    provider = DockerProvider({"snapshots": {
        "cpu-free-ubuntu": {"image": "ale-ubuntu22-docker"},
        "gpu-free-ubuntu": {"image": "ale-ubuntu22-docker", "gpus": "all"},
    }})

    await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu"))
    await provider.acquire(SandboxSpec(snapshot="gpu-free-ubuntu"))

    assert "--gpus" not in docker_calls[0]
    gpu_args = docker_calls[1]
    assert gpu_args[gpu_args.index("--gpus") + 1] == "all"


def test_gpu_request_rejects_unknown_value() -> None:
    with pytest.raises(ValueError, match="docker.gpus"):
        DockerProviderConfig(gpus="all,all")


async def test_qizhi_entrypoint_override_is_run_scoped(monkeypatch, tmp_path: Path) -> None:
    docker_calls: list[tuple[str, ...]] = []

    async def fake_run_docker(*args: str) -> tuple[int, str, str]:
        docker_calls.append(args)
        return 0, "0123456789abcdef", ""

    async def fake_wait_cua_ready(url: str, **kwargs: object) -> bool:
        return True

    override = tmp_path / "entrypoint.sh"
    override.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("ALE_QZ_ENTRYPOINT_HOST", str(override))
    monkeypatch.setattr(docker_module, "_run_docker", fake_run_docker)
    monkeypatch.setattr(docker_module, "_wait_cua_ready", fake_wait_cua_ready)
    provider = DockerProvider({"image": "ale-ubuntu22-docker", "network_mode": "host"})

    await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu"))

    run_args = docker_calls[0]
    mount = run_args.index("-v")
    assert run_args[mount + 1] == f"{override}:/dockerstartup/entrypoint.sh:ro"
    assert run_args[-2:] == ("agentslastexam/ale-ubuntu22-docker:latest", "--wait")


def test_loader_keeps_distinct_docker_snapshot_images(tmp_path: Path) -> None:
    (tmp_path / "agent.yaml").write_text(
        "harness: dummy\nmodel: test\n", encoding="utf-8"
    )
    (tmp_path / "environment.yaml").write_text(
        """
snapshots:
  engineering-a:
    provider: docker
    image: ale-ubuntu22-docker
    docker:
      image_ref: ale/task-a:1
  engineering-b:
    provider: docker
    image: ale-ubuntu22-docker
    docker:
      image_ref: ale/task-b:1
task_data_source: local:task-data
output_path: null
""",
        encoding="utf-8",
    )
    (tmp_path / "experiment.yaml").write_text(
        """
name: docker-snapshots
agent: agent.yaml
environment: environment.yaml
tasks:
  - path: demo/hello
""",
        encoding="utf-8",
    )

    spec = load_experiment(tmp_path / "experiment.yaml")
    snapshots = spec.environment.provider_specs["docker"].config["snapshots"]
    assert snapshots["engineering-a"]["image_ref"] == "ale/task-a:1"
    assert snapshots["engineering-b"]["image_ref"] == "ale/task-b:1"


@pytest.mark.parametrize("enabled", [True, False])
async def test_qizhi_claude_mount_is_file_scoped_readonly(monkeypatch, tmp_path, enabled):
    from unittest.mock import AsyncMock

    claude = tmp_path / "claude"
    claude.write_text("#!/bin/sh\n")
    if enabled:
        monkeypatch.setenv("ALE_QZ_CLAUDE_HOST", str(claude))
    else:
        monkeypatch.delenv("ALE_QZ_CLAUDE_HOST", raising=False)
    monkeypatch.delenv("ALE_QZ_ENTRYPOINT_HOST", raising=False)
    run = AsyncMock(return_value=(0, "container-id", ""))
    monkeypatch.setattr(docker_module, "_run_docker", run)
    monkeypatch.setattr(docker_module, "_wait_cua_ready", AsyncMock(return_value=True))
    provider = DockerProvider({"image": "ale-ubuntu22-docker", "network_mode": "host"})
    await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu"))
    args = run.call_args.args
    assert (f"{claude}:/usr/local/bin/claude:ro" in args) is enabled
    assert args.count("-v") == int(enabled)


@pytest.mark.parametrize("exists", [True, False])
async def test_qizhi_claude_rejects_directory_or_missing_file(monkeypatch, tmp_path, exists):
    from unittest.mock import AsyncMock

    source = tmp_path / "claude"
    if exists:
        source.mkdir()
    monkeypatch.setenv("ALE_QZ_CLAUDE_HOST", str(source))
    run = AsyncMock()
    monkeypatch.setattr(docker_module, "_run_docker", run)
    provider = DockerProvider({"image": "ale-ubuntu22-docker", "network_mode": "host"})
    with pytest.raises((ValueError, FileNotFoundError)):
        await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu"))
    run.assert_not_awaited()


@pytest.mark.parametrize("enabled", [True, False])
async def test_qizhi_claude_tmp_mount_is_run_scoped_readwrite(monkeypatch, tmp_path, enabled):
    from unittest.mock import AsyncMock

    source = tmp_path / "claude-tmp"
    source.mkdir()
    if enabled:
        monkeypatch.setenv("ALE_QZ_CLAUDE_TMP_HOST", str(source))
    else:
        monkeypatch.delenv("ALE_QZ_CLAUDE_TMP_HOST", raising=False)
    monkeypatch.delenv("ALE_QZ_ENTRYPOINT_HOST", raising=False)
    monkeypatch.delenv("ALE_QZ_CLAUDE_HOST", raising=False)
    run = AsyncMock(return_value=(0, "container-id", ""))
    monkeypatch.setattr(docker_module, "_run_docker", run)
    monkeypatch.setattr(docker_module, "_wait_cua_ready", AsyncMock(return_value=True))
    provider = DockerProvider({"image": "ale-ubuntu22-docker", "network_mode": "host"})
    await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu"))
    args = run.call_args.args
    assert (f"{source}:/tmp/claude-1000:rw" in args) is enabled
    assert args.count("-v") == int(enabled)


@pytest.mark.parametrize("exists", [True, False])
async def test_qizhi_claude_tmp_rejects_file_or_missing_directory(monkeypatch, tmp_path, exists):
    from unittest.mock import AsyncMock

    source = tmp_path / "claude-tmp"
    if exists:
        source.write_text("not a directory")
    monkeypatch.setenv("ALE_QZ_CLAUDE_TMP_HOST", str(source))
    run = AsyncMock()
    monkeypatch.setattr(docker_module, "_run_docker", run)
    provider = DockerProvider({"image": "ale-ubuntu22-docker", "network_mode": "host"})
    with pytest.raises((ValueError, FileNotFoundError)):
        await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu"))
    run.assert_not_awaited()


async def test_provider_selects_image_ref_by_snapshot(monkeypatch) -> None:
    docker_calls: list[tuple[str, ...]] = []

    async def fake_run_docker(*args: str) -> tuple[int, str, str]:
        docker_calls.append(args)
        if args[:2] == ("image", "inspect"):
            return 0, "sha256:" + "b" * 64, ""
        return 0, "0123456789abcdef", ""

    async def fake_get_host_port(container_name: str, internal_port: int) -> int:
        return {5000: 15000, 6901: 16901}[internal_port]

    async def fake_wait_cua_ready(url: str, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(docker_module, "_run_docker", fake_run_docker)
    monkeypatch.setattr(docker_module, "_get_host_port", fake_get_host_port)
    monkeypatch.setattr(docker_module, "_wait_cua_ready", fake_wait_cua_ready)
    provider = DockerProvider({
        "snapshots": {
            "engineering-a": {
                "image": "ale-ubuntu22-docker",
                "image_ref": "ale/task-a:1",
            },
            "engineering-b": {
                "image": "ale-ubuntu22-docker",
                "image_ref": "ale/task-b:1",
                "expected_image_id": "sha256:" + "b" * 64,
                "apparmor_unconfined": True,
            },
        }
    })

    sandbox = await provider.acquire(SandboxSpec(snapshot="engineering-b"))

    assert docker_calls[0][:3] == ("image", "inspect", "ale/task-b:1")
    run_call = docker_calls[1]
    assert run_call[-2:] == ("ale/task-b:1", "--wait")
    option = run_call.index("--security-opt")
    assert run_call[option:option + 2] == (
        "--security-opt", "apparmor=unconfined"
    )
    assert sandbox.metadata["image_ref"] == "ale/task-b:1"


async def test_provider_rejects_unregistered_snapshot() -> None:
    provider = DockerProvider({
        "snapshots": {
            "engineering-a": {
                "image": "ale-ubuntu22-docker",
                "image_ref": "ale/task-a:1",
            }
        }
    })
    with pytest.raises(KeyError, match="not in Docker provider config"):
        await provider.acquire(SandboxSpec(snapshot="engineering-missing"))


async def test_provider_rejects_retagged_snapshot_image(monkeypatch) -> None:
    async def fake_run_docker(*args: str) -> tuple[int, str, str]:
        assert args[:2] == ("image", "inspect")
        return 0, "sha256:" + "a" * 64, ""

    monkeypatch.setattr(docker_module, "_run_docker", fake_run_docker)
    provider = DockerProvider({
        "snapshots": {
            "engineering-a": {
                "image": "ale-ubuntu22-docker",
                "image_ref": "ale/task-a:1",
                "expected_image_id": "sha256:" + "b" * 64,
            }
        }
    })
    with pytest.raises(RuntimeError, match="expected image"):
        await provider.acquire(SandboxSpec(snapshot="engineering-a"))


@pytest.mark.parametrize("mode", ["none", "qizhi", "HOST "])
def test_docker_network_mode_rejects_unknown_value(mode: str) -> None:
    with pytest.raises(ValueError, match="network_mode"):
        DockerProviderConfig(network_mode=mode)  # type: ignore[arg-type]


async def test_host_network_uses_fixed_ports_without_publish_or_inspect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker_calls: list[tuple[str, ...]] = []
    ready_urls: list[str] = []

    async def fake_run_docker(*args: str) -> tuple[int, str, str]:
        docker_calls.append(args)
        return 0, "0123456789abcdef", ""

    async def fail_get_host_port(container_name: str, internal_port: int) -> int:
        pytest.fail(f"host mode must not inspect a port mapping: {internal_port}")

    async def fake_wait_cua_ready(url: str, **kwargs: object) -> bool:
        ready_urls.append(url)
        return True

    monkeypatch.setattr(docker_module, "_run_docker", fake_run_docker)
    monkeypatch.setattr(docker_module, "_get_host_port", fail_get_host_port)
    monkeypatch.setattr(docker_module, "_wait_cua_ready", fake_wait_cua_ready)

    provider = DockerProvider({
        "image": "ale-ubuntu22-docker",
        "network_mode": "host",
    })
    sandbox = await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu"))

    run_args = docker_calls[0]
    network_index = run_args.index("--network")
    assert run_args[network_index:network_index + 2] == ("--network", "host")
    assert "-p" not in run_args
    assert sandbox.endpoint == "http://127.0.0.1:5000"
    assert sandbox.metadata["cua_port"] == 5000
    assert sandbox.metadata["vnc_port"] == 6901
    assert sandbox.metadata["network_mode"] == "host"
    assert ready_urls == ["http://127.0.0.1:5000"]


async def test_bridge_network_keeps_random_port_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docker_calls: list[tuple[str, ...]] = []
    inspected_ports: list[int] = []
    ready_urls: list[str] = []

    async def fake_run_docker(*args: str) -> tuple[int, str, str]:
        docker_calls.append(args)
        return 0, "fedcba9876543210", ""

    async def fake_get_host_port(container_name: str, internal_port: int) -> int:
        inspected_ports.append(internal_port)
        return {5000: 15000, 6901: 16901}[internal_port]

    async def fake_wait_cua_ready(url: str, **kwargs: object) -> bool:
        ready_urls.append(url)
        return True

    monkeypatch.setattr(docker_module, "_run_docker", fake_run_docker)
    monkeypatch.setattr(docker_module, "_get_host_port", fake_get_host_port)
    monkeypatch.setattr(docker_module, "_wait_cua_ready", fake_wait_cua_ready)

    provider = DockerProvider({"image": "ale-ubuntu22-docker"})
    sandbox = await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu"))

    run_args = docker_calls[0]
    assert "--network" not in run_args
    assert run_args.count("-p") == 2
    assert "127.0.0.1:0:5000" in run_args
    assert "127.0.0.1:0:6901" in run_args
    assert inspected_ports == [5000, 6901]
    assert sandbox.endpoint == "http://127.0.0.1:15000"
    assert ready_urls == [sandbox.endpoint]
    assert sandbox.metadata["network_mode"] == "bridge"


@pytest.mark.parametrize("ipv6_first", [True, False])
async def test_port_lookup_selects_ipv4_loopback_by_address(monkeypatch, ipv6_first):
    bindings = [
        {"HostIp": "::", "HostPort": "16000"},
        {"HostIp": "127.0.0.1", "HostPort": "15000"},
    ]
    if not ipv6_first:
        bindings.reverse()

    async def fake_run(*args):
        assert args == (
            "inspect", "--format", '{{json (index .NetworkSettings.Ports "5000/tcp")}}',
            "test-container",
        )
        return 0, json.dumps(bindings), ""

    monkeypatch.setattr(docker_module, "_run_docker", fake_run)
    assert await docker_module._get_host_port("test-container", 5000) == 15000


@pytest.mark.parametrize("bindings", [
    None,
    [],
    [{"HostIp": "::", "HostPort": "16000"}],
    [{"HostIp": "0.0.0.0", "HostPort": "15000"}],
    [{"HostIp": "127.0.0.1", "HostPort": "bad"}],
    [{"HostIp": "127.0.0.1", "HostPort": "0"}],
    [{"HostIp": "127.0.0.1", "HostPort": "65536"}],
    [{"HostIp": "127.0.0.1"}],
    [{"HostIp": "127.0.0.1", "HostPort": "15000"}] * 2,
])
async def test_port_lookup_rejects_missing_or_invalid_loopback(monkeypatch, bindings):
    async def fake_run(*args):
        return 0, json.dumps(bindings), ""

    monkeypatch.setattr(docker_module, "_run_docker", fake_run)
    with pytest.raises(RuntimeError, match="IPv4 loopback binding"):
        await docker_module._get_host_port("test-container", 5000)


async def test_failed_port_inspection_cleans_up_new_container(monkeypatch):
    calls = []

    async def fake_run(*args):
        calls.append(args)
        if args[0] == "inspect":
            return 1, "", "inspection failed"
        return 0, "container-id", ""

    monkeypatch.setattr(docker_module, "_run_docker", fake_run)
    provider = DockerProvider({"image": "ale-ubuntu22-docker"})
    with pytest.raises(RuntimeError, match="inspection failed"):
        await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu"))
    name = calls[0][calls[0].index("--name") + 1]
    assert calls[-1] == ("rm", "-f", name)


async def test_cua_startup_retry_saves_diagnostics_before_removal(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock

    calls = []

    async def fake_run(*args):
        calls.append(args)
        if args[0] == "logs":
            return 0, "server failed to bind", "startup stderr"
        if args[0] == "inspect":
            return 0, json.dumps({"Running": False, "OOMKilled": True, "ExitCode": 137}), ""
        return 0, "container-id", ""

    monkeypatch.setenv("ALE_DOCKER_DIAGNOSTICS_DIR", str(tmp_path))
    monkeypatch.setattr(docker_module, "_run_docker", fake_run)
    monkeypatch.setattr(docker_module, "_get_host_port", AsyncMock(return_value=15000))
    ready = AsyncMock(side_effect=[False, True])
    monkeypatch.setattr(docker_module, "_wait_cua_ready", ready)
    monkeypatch.setattr(docker_module.asyncio, "sleep", AsyncMock())
    provider = DockerProvider({
        "image": "ale-ubuntu22-docker", "disable_proxy": True,
        "cua_ready_timeout": 600, "startup_attempts": 3,
    })

    await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu"))

    runs = [args for args in calls if args[0] == "run"]
    assert len(runs) == 2
    assert runs[0][runs[0].index("--name") + 1] != runs[1][runs[1].index("--name") + 1]
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        assert f"{key}=" in runs[0]
    assert ready.await_count == 2
    assert ready.await_args_list[0].kwargs == {"timeout": 600}
    actions = [args[0] for args in calls]
    assert actions.index("logs") < actions.index("inspect") < actions.index("rm")
    state = json.loads(next(tmp_path.glob("*.state.json")).read_text())
    assert state["state"]["OOMKilled"] is True
    assert "startup stderr" in next(tmp_path.glob("*.log")).read_text()


async def test_cua_startup_attempts_are_bounded(monkeypatch):
    from unittest.mock import AsyncMock

    run = AsyncMock(return_value=(0, "container-id", ""))
    monkeypatch.delenv("ALE_DOCKER_DIAGNOSTICS_DIR", raising=False)
    monkeypatch.setattr(docker_module, "_run_docker", run)
    monkeypatch.setattr(docker_module, "_get_host_port", AsyncMock(return_value=15000))
    monkeypatch.setattr(docker_module, "_wait_cua_ready", AsyncMock(return_value=False))
    monkeypatch.setattr(docker_module.asyncio, "sleep", AsyncMock())
    provider = DockerProvider({"image": "ale-ubuntu22-docker", "startup_attempts": 3})
    with pytest.raises(RuntimeError, match="did not become ready"):
        await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu"))
    actions = [call.args[0] for call in run.await_args_list]
    assert actions.count("run") == actions.count("rm") == 3


@pytest.mark.parametrize("config", [
    {"cua_ready_timeout": 0}, {"cua_ready_timeout": float("inf")},
    {"startup_attempts": 0}, {"startup_attempts": 11},
])
def test_invalid_cua_startup_settings_rejected(config):
    with pytest.raises(ValueError):
        DockerProvider(config)
