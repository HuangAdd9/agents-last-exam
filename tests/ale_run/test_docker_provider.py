from __future__ import annotations

import pytest

from ale_run.base_interface import SandboxSpec
from ale_run.environments.providers import docker as docker_module
from ale_run.environments.providers.docker import DockerProvider, DockerProviderConfig


def test_docker_network_mode_defaults_to_bridge() -> None:
    provider = DockerProvider({"image": "ale-ubuntu22-docker"})
    assert provider.config.network_mode == "bridge"


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

    async def fake_run_docker(*args: str) -> tuple[int, str, str]:
        docker_calls.append(args)
        return 0, "fedcba9876543210", ""

    async def fake_get_host_port(container_name: str, internal_port: int) -> int:
        inspected_ports.append(internal_port)
        return {5000: 15000, 6901: 16901}[internal_port]

    async def fake_wait_cua_ready(url: str, **kwargs: object) -> bool:
        return True

    monkeypatch.setattr(docker_module, "_run_docker", fake_run_docker)
    monkeypatch.setattr(docker_module, "_get_host_port", fake_get_host_port)
    monkeypatch.setattr(docker_module, "_wait_cua_ready", fake_wait_cua_ready)

    provider = DockerProvider({"image": "ale-ubuntu22-docker"})
    sandbox = await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu"))

    run_args = docker_calls[0]
    assert "--network" not in run_args
    assert run_args.count("-p") == 2
    assert "0:5000" in run_args
    assert "0:6901" in run_args
    assert inspected_ports == [5000, 6901]
    assert sandbox.endpoint == "http://localhost:15000"
    assert sandbox.metadata["network_mode"] == "bridge"
