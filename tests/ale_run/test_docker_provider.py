import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from ale_run.base_interface import SandboxSpec
from ale_run.environments import images
from ale_run.environments.providers import docker
from ale_run.environments.providers import docker as docker_module
from ale_run.environments.providers.docker import DockerProvider, DockerProviderConfig
from ale_run.orchestration.config_loader import _build_per_snapshot_env

_IMAGE_ID = "sha256:" + "1" * 64
_REPO_DIGEST = "sha256:" + "2" * 64


@pytest.fixture(autouse=True)
def cpu_scheduler(monkeypatch, tmp_path):
    monkeypatch.setattr(docker, "_CPUSET_LOCK_PATH", tmp_path / "cpuset.lock")
    monkeypatch.setattr(docker.os, "sched_getaffinity", lambda _: set(range(32)))
    monkeypatch.setattr(docker, "_occupied_cpus", AsyncMock(return_value=set()))


@pytest.mark.asyncio
@pytest.mark.parametrize("image_name", ["ale-kasm", "ale-ubuntu22-docker-v1-1"])
@pytest.mark.parametrize("ready", [True, False])
async def test_acquire_uses_one_loopback_address_family(monkeypatch, image_name, ready):
    image = images.get(image_name)
    selected_digest = image.docker_image.rsplit("@", 1)[-1] if "@" in image.docker_image else None
    commands = AsyncMock(
        side_effect=[
            (0, "container-id", ""),
            (0, _IMAGE_ID, ""),
            (0, json.dumps([f"example/image@{selected_digest}"] if selected_digest else []), ""),
            (0, "32809", ""),
            (0, "32808", ""),
            (0, json.dumps({"NanoCpus": 4_000_000_000, "Memory": 15 * 1024 ** 3}), ""),
        ]
    )
    readiness = AsyncMock(return_value=ready)
    monkeypatch.setattr(docker, "_run_docker", commands)
    monkeypatch.setattr(docker, "_wait_cua_ready", readiness)
    provider = docker.DockerProvider({"image": image_name})
    spec = SandboxSpec(snapshot="cpu-free-ubuntu", machine_type="c4-standard-4")

    if ready:
        sandbox = await provider.acquire(spec)
        assert sandbox.endpoint == "http://127.0.0.1:32809"
        assert sandbox.metadata["cua_port"] == 32809
        assert sandbox.metadata["vnc_port"] == 32808
        assert sandbox.metadata["resource_limits"] == {
            "cpu_quota": 4.0,
            "memory_limit_bytes": 15 * 1024 ** 3,
            "memory_and_swap_limit_bytes": 0,
        }
        assert sandbox.metadata["image_provenance"] == {
            "provider": "docker",
            "image": image_name,
            "image_revision": selected_digest or _IMAGE_ID,
            "image_ref": image.docker_image,
            "image_id": _IMAGE_ID,
        }
        assert commands.await_count == 6
    else:
        with pytest.raises(RuntimeError, match=r"127\.0\.0\.1:32809.*did not become ready"):
            await provider.acquire(spec)
        assert commands.await_args_list[-1].args[:2] == ("rm", "-f")

    readiness.assert_awaited_once_with("http://127.0.0.1:32809")
    arguments = commands.await_args_list[0].args
    published = [arguments[index + 1] for index, value in enumerate(arguments) if value == "-p"]
    assert published == [f"127.0.0.1:0:{image.cua_server_port}", "127.0.0.1:0:6901"]
    assert arguments[arguments.index("--cpus") + 1] == "4.0"
    assert arguments[arguments.index("--cpuset-cpus") + 1] == "0,1,2,3"
    assert arguments[arguments.index("--memory") + 1] == "15g"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("container_ref", "expected_revision", "recorded_revision"),
    [
        ("example/image:latest", None, _IMAGE_ID),
        ("example/image:latest", _IMAGE_ID, _IMAGE_ID),
        ("example/image:latest", _REPO_DIGEST, _REPO_DIGEST),
        (f"example/image:v1@{_REPO_DIGEST}", None, _REPO_DIGEST),
        (f"example/image:v1@{_REPO_DIGEST}", _IMAGE_ID, _REPO_DIGEST),
    ],
)
async def test_acquire_verifies_revision_and_preserves_selected_digest(
    monkeypatch, container_ref, expected_revision, recorded_revision,
):
    commands = AsyncMock(side_effect=[
        (0, "container-id", ""),
        (0, _IMAGE_ID, ""),
        (0, json.dumps([f"example/other@{'sha256:' + '3' * 64}", f"example/image@{_REPO_DIGEST}"]), ""),
        (0, "32809", ""),
        (0, "32808", ""),
        (0, json.dumps({"NanoCpus": 4_000_000_000, "Memory": 1024 ** 3}), ""),
    ])
    monkeypatch.setattr(docker, "_run_docker", commands)
    monkeypatch.setattr(docker, "_wait_cua_ready", AsyncMock(return_value=True))
    provider = docker.DockerProvider({
        "image": "ale-kasm", "image_ref": container_ref, "image_revision": expected_revision,
    })

    sandbox = await provider.acquire(SandboxSpec(snapshot="cpu-free"))

    assert json.loads(json.dumps(sandbox.metadata["image_provenance"])) == {
        "provider": "docker", "image": "ale-kasm", "image_revision": recorded_revision,
        "image_ref": container_ref, "image_id": _IMAGE_ID,
    }
    assert commands.await_args_list[0].args[-2] == container_ref
    assert commands.await_args_list[2].args == (
        "image", "inspect", "--format", "{{json .RepoDigests}}", _IMAGE_ID,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["expected_pin", "selected_digest", "inspect", "image_inspect"])
async def test_acquire_cleans_up_when_image_identity_cannot_be_verified(monkeypatch, failure):
    config = {"image": "ale-kasm", "image_ref": "example/image:latest"}
    if failure == "expected_pin":
        config["image_revision"] = "sha256:" + "9" * 64
    if failure == "selected_digest":
        config["image_ref"] = f"example/image@{_REPO_DIGEST}"
    responses = [(0, "container-id", "")]
    if failure == "inspect":
        responses.append((1, "", "inspect unavailable"))
    else:
        responses.append((0, _IMAGE_ID, ""))
        responses.append(
            (1, "", "image inspect unavailable") if failure == "image_inspect" else (0, "[]", "")
        )
    responses.append((0, "removed", ""))
    commands = AsyncMock(side_effect=responses)
    readiness = AsyncMock(return_value=True)
    monkeypatch.setattr(docker, "_run_docker", commands)
    monkeypatch.setattr(docker, "_wait_cua_ready", readiness)

    with pytest.raises(RuntimeError, match="mismatch|unavailable"):
        await docker.DockerProvider(config).acquire(SandboxSpec(snapshot="cpu-free"))

    assert commands.await_args_list[-1].args[:2] == ("rm", "-f")
    readiness.assert_not_awaited()


@pytest.mark.parametrize("revision", ["latest", "sha256:short", 0, f"example/image@{_REPO_DIGEST}"])
def test_config_rejects_invalid_image_revision(revision):
    with pytest.raises(ValueError, match="docker image_revision must be a sha256"):
        docker.DockerProvider({"image_revision": revision})


def test_loader_forwards_docker_image_revision():
    environment = _build_per_snapshot_env({
        "snapshots": {
            "cpu-free": {
                "provider": "docker",
                "image": "ale-kasm",
                "docker": {"image_ref": "example/image:latest", "image_revision": _REPO_DIGEST},
            },
        },
    }, "environment.yaml")

    provider = docker.DockerProvider(environment.provider_specs["docker"].config)

    assert provider.config.image_revision == _REPO_DIGEST


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["port", "readiness", "limits", "credentials", "cancel"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_acquire_cleans_up_after_post_create_failure(monkeypatch, stage, cleanup_fails):
    failure = asyncio.CancelledError() if stage == "cancel" else RuntimeError(f"failed {stage}")
    commands = AsyncMock(side_effect=[
        (0, "container-id", ""),
        RuntimeError("cleanup failed") if cleanup_fails else (0, "removed", ""),
    ])
    monkeypatch.setattr(docker, "_run_docker", commands)
    monkeypatch.setattr(docker, "_get_image_provenance", AsyncMock(return_value={}))
    port = AsyncMock(return_value=32809)
    readiness = AsyncMock(return_value=True)
    limits = AsyncMock(return_value={})
    credentials = AsyncMock()
    failing_call = {
        "port": port, "readiness": readiness, "limits": limits,
        "credentials": credentials, "cancel": readiness,
    }[stage]
    failing_call.side_effect = failure
    monkeypatch.setattr(docker, "_get_host_port", port)
    monkeypatch.setattr(docker, "_wait_cua_ready", readiness)
    monkeypatch.setattr(docker, "_get_resource_limits", limits)
    monkeypatch.setattr(docker.DockerProvider, "_inject_gcs_credentials", credentials)
    provider = docker.DockerProvider({"gcs_sa_key": "/unused/key.json"})

    with pytest.raises(type(failure)) as caught:
        await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu"))

    assert caught.value is failure
    arguments = commands.await_args_list[0].args
    container_name = arguments[arguments.index("--name") + 1]
    assert commands.await_args_list[-1].args == ("rm", "-f", container_name)
    assert commands.await_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("memory_override", ["", "2g"])
@pytest.mark.parametrize("allow_swap", [False, True])
async def test_container_swap_limits(monkeypatch, memory_override, allow_swap):
    memory_gib = 2 if memory_override else 3
    limits = {
        "NanoCpus": 2_000_000_000,
        "Memory": memory_gib * 1024 ** 3,
        "MemorySwap": (2 if allow_swap else 1) * memory_gib * 1024 ** 3,
    }
    commands = AsyncMock(side_effect=[
        (0, "container-id", ""), (0, json.dumps(limits), ""),
    ])
    monkeypatch.setattr(docker, "_run_docker", commands)
    monkeypatch.setattr(docker, "_get_image_provenance", AsyncMock(return_value={}))
    monkeypatch.setattr(docker, "_get_host_port", AsyncMock(return_value=32809))
    monkeypatch.setattr(docker, "_wait_cua_ready", AsyncMock(return_value=True))
    provider = docker.DockerProvider({
        "allow_swap": allow_swap, "memory": memory_override,
    })

    sandbox = await provider.acquire(SandboxSpec(snapshot="cpu-free", vcpus=2, memory_gb=3))

    args = commands.await_args_list[0].args
    assert args[args.index("--memory") + 1] == f"{memory_gib}g"
    assert "--pids-limit" not in args
    if allow_swap:
        assert "--memory-swap" not in args
    else:
        assert args[args.index("--memory-swap") + 1] == f"{memory_gib}g"
    assert sandbox.metadata["resource_limits"]["memory_and_swap_limit_bytes"] == limits["MemorySwap"]


@pytest.mark.asyncio
async def test_missing_requested_swap_limit_fails_and_removes_container(monkeypatch):
    limits = {
        "NanoCpus": 4_000_000_000, "Memory": 15 * 1024 ** 3,
    }
    commands = AsyncMock(side_effect=[
        (0, "container-id", ""), (0, json.dumps(limits), ""), (0, "removed", ""),
    ])
    monkeypatch.setattr(docker, "_run_docker", commands)
    monkeypatch.setattr(docker, "_get_image_provenance", AsyncMock(return_value={}))
    monkeypatch.setattr(docker, "_get_host_port", AsyncMock(return_value=32809))
    monkeypatch.setattr(docker, "_wait_cua_ready", AsyncMock(return_value=True))

    with pytest.raises(RuntimeError, match="requested swap limit"):
        await docker.DockerProvider({"allow_swap": False}).acquire(
            SandboxSpec(snapshot="cpu-free"),
        )

    assert commands.await_args_list[-1].args[:2] == ("rm", "-f")


@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_config_rejects_non_boolean_swap_policy(value):
    with pytest.raises(TypeError, match="allow_swap must be a boolean"):
        docker.DockerProvider({"allow_swap": value})


@pytest.mark.asyncio
async def test_lxcfs_binds_resource_views_readonly(monkeypatch, tmp_path):
    paths = [f"proc/{name}" for name in (
        "cpuinfo", "meminfo", "stat", "swaps", "uptime", "diskstats", "slabinfo",
    )] + ["sys/devices/system/cpu/online"]
    for relative in paths:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("fixture")
    commands = AsyncMock(return_value=(0, "container-id", ""))
    monkeypatch.setattr(docker, "_run_docker", commands)
    monkeypatch.setattr(docker, "_get_image_provenance", AsyncMock(return_value={}))
    monkeypatch.setattr(docker, "_get_host_port", AsyncMock(return_value=32809))
    monkeypatch.setattr(docker, "_wait_cua_ready", AsyncMock(return_value=True))
    monkeypatch.setattr(docker, "_get_resource_limits", AsyncMock(return_value={}))

    handle = await docker.DockerProvider({"lxcfs_root": str(tmp_path)}).acquire(
        SandboxSpec(snapshot="cpu-free"),
    )

    args = commands.await_args_list[0].args
    mounts = [args[i + 1] for i, value in enumerate(args) if value == "--mount"]
    assert set(mounts) == {
        f"type=bind,src={tmp_path / path},dst=/{path},readonly" for path in paths
    }
    assert handle.metadata["lxcfs_root"] == str(tmp_path)
    assert "--pids-limit" not in args


@pytest.mark.asyncio
async def test_lxcfs_missing_view_fails_before_container_start(monkeypatch, tmp_path):
    commands = AsyncMock()
    monkeypatch.setattr(docker, "_run_docker", commands)
    with pytest.raises(ValueError, match="LXCFS resource view is unavailable"):
        await docker.DockerProvider({"lxcfs_root": str(tmp_path)}).acquire(
            SandboxSpec(snapshot="cpu-free"),
        )
    commands.assert_not_awaited()


@pytest.mark.parametrize("root", ["relative/lxcfs", "/var/lib/lxcfs,readonly"])
def test_lxcfs_rejects_invalid_mount_root(root):
    with pytest.raises(ValueError, match="absolute path without commas"):
        docker._lxcfs_mount_args(root)
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
    monkeypatch.setattr(docker_module, "_get_image_provenance", AsyncMock(return_value={}))
    monkeypatch.setattr(docker_module, "_get_resource_limits", AsyncMock(return_value={}))

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
    monkeypatch.setattr(docker_module, "_get_image_provenance", AsyncMock(return_value={}))
    monkeypatch.setattr(docker_module, "_get_resource_limits", AsyncMock(return_value={}))

    provider = DockerProvider({"image": "ale-ubuntu22-docker"})
    sandbox = await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu"))

    run_args = docker_calls[0]
    assert "--network" not in run_args
    assert run_args.count("-p") == 2
    assert any(value.endswith(":0:5000") for value in run_args)
    assert any(value.endswith(":0:6901") for value in run_args)
    assert inspected_ports == [5000, 6901]
    assert sandbox.endpoint == "http://127.0.0.1:15000"
    assert sandbox.metadata["network_mode"] == "bridge"
