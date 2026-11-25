from __future__ import annotations

import importlib.util
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ale_run.environments import qemu_control
from ale_run.environments.providers import qemu

spec = importlib.util.spec_from_file_location(
    "qemu_proxy",
    Path(__file__).resolve().parents[2] / "scripts/qemu_proxy.py",
)
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)


@pytest.mark.parametrize("network", [None, "ale-qemu-egress"])
async def test_runner_joins_only_requested_bridge(tmp_path, monkeypatch, network):
    snapshot = qemu._build_snapshot_config(
        {
            "image": "ale-ubuntu22",
            "disk_source": "/unused.qcow2",
            "network": network,
        }
    )
    docker = AsyncMock(return_value=(0, "bridge", ""))
    monkeypatch.setattr(qemu, "_run_docker", docker)
    await qemu.QemuProvider._start_container(
        snapshot=snapshot,
        name="test-vm",
        task_id="demo/hello",
        base_qcow2=tmp_path / "base",
        storage_dir=tmp_path,
        exchange_dir=tmp_path,
        cua_port=5000,
        vcpus=4,
        memory_gb=8,
    )
    args = docker.await_args.args
    assert args[-1] == snapshot.runner_image
    assert "--device=/dev/kvm" in args
    assert "127.0.0.1:0:5000" in args
    assert not any("PROXY=" in arg.upper() for arg in args)
    if network:
        assert args[args.index("--network") + 1] == network
        assert docker.await_count == 2
    else:
        assert "--network" not in args
        assert docker.await_count == 1


@pytest.mark.parametrize("network", ["host", "none", "container:abc", "", " x", 123])
def test_rejects_network_modes_that_bypass_bridge_isolation(network):
    with pytest.raises(ValueError, match="Docker bridge"):
        qemu._build_snapshot_config(
            {
                "image": "ale-ubuntu22",
                "disk_source": "/unused",
                "network": network,
            }
        )


async def test_rejects_non_bridge_driver_before_starting_vm(tmp_path, monkeypatch):
    snapshot = qemu._build_snapshot_config(
        {
            "image": "ale-ubuntu22",
            "disk_source": "/unused",
            "network": "other",
        }
    )
    docker = AsyncMock(return_value=(0, "macvlan", ""))
    monkeypatch.setattr(qemu, "_run_docker", docker)
    with pytest.raises(ValueError, match="bridge driver"):
        await qemu.QemuProvider._start_container(
            snapshot=snapshot,
            name="test",
            task_id="demo/hello",
            base_qcow2=tmp_path,
            storage_dir=tmp_path,
            exchange_dir=tmp_path,
            cua_port=5000,
            vcpus=4,
            memory_gb=8,
        )
    assert docker.await_count == 1


def test_stop_leaves_gateway_untouched_while_vm_attached(monkeypatch):
    monkeypatch.setattr(proxy, "inspect_network", lambda: {"Containers": {"vm": {}}})
    calls = []
    monkeypatch.setattr(proxy, "run", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(RuntimeError, match="still has containers"):
        proxy.stop()
    assert calls == []


def test_stop_does_not_remove_rules_if_container_attaches_during_cleanup(monkeypatch):
    monkeypatch.setattr(proxy, "inspect_network", lambda: {"Containers": {}})
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        raise RuntimeError("network has active endpoints")

    monkeypatch.setattr(proxy, "run", run)
    with pytest.raises(RuntimeError, match="active endpoints"):
        proxy.stop()
    assert calls == [("docker", "network", "rm", proxy.NETWORK)]


def test_does_not_take_over_unowned_network(monkeypatch):
    network = {"Labels": None, "Driver": "bridge", "Options": {}}
    monkeypatch.setattr(
        proxy,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, json.dumps([network]), ""),
    )
    with pytest.raises(RuntimeError, match="not this script"):
        proxy.inspect_network()


def test_repeated_start_keeps_existing_gateway_and_configuration(monkeypatch):
    monkeypatch.setattr(proxy, "inspect_network", lambda: {"Containers": {"vm": {}}})
    monkeypatch.setattr(proxy, "healthy", lambda: True)
    monkeypatch.setattr(proxy, "stop", lambda: pytest.fail("must not interrupt existing VM"))
    proxy.start("unused", [])


def test_docker_daemon_failure_is_not_treated_as_missing_network(monkeypatch):
    monkeypatch.setattr(
        proxy,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 1, "", "Cannot connect to Docker"
        ),
    )
    with pytest.raises(RuntimeError, match="Cannot connect"):
        proxy.inspect_network()


async def test_diagnostics_record_runner_network_without_container_env(tmp_path, monkeypatch):
    monkeypatch.setattr(
        qemu_control, "inspect_health", AsyncMock(return_value={"guest": "unknown"})
    )
    query = AsyncMock(return_value={"returncode": 0, "stdout": "network evidence"})
    monkeypatch.setattr(qemu_control, "docker_query", query)
    await qemu_control.collect_diagnostics(SimpleNamespace(id="vm", metadata={}), tmp_path)
    query.assert_any_await("inspect", "--format", "{{json .NetworkSettings.Networks}}", "vm")
    report = json.loads((tmp_path / "qemu.json").read_text())
    assert report["runner_network"]["stdout"] == "network evidence"
    assert not any(".Env" in str(call) for call in query.await_args_list)


@pytest.mark.parametrize("fail_firewall", [False, True])
def test_start_never_connects_to_transparent_port_and_rolls_back(
    tmp_path, monkeypatch, fail_firewall
):
    networks = iter([None, {"IPAM": {"Config": [{"Gateway": "10.42.0.1"}]}}])
    monkeypatch.setattr(proxy, "inspect_network", lambda: next(networks))
    monkeypatch.setattr(proxy.shutil, "which", lambda binary: "/test/mihomo")
    monkeypatch.setattr(proxy, "STATE", tmp_path)
    stops = []
    monkeypatch.setattr(proxy, "stop", lambda: stops.append(True))

    class Upstream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def sendall(self, value):
            assert value == b"\x05\x01\x00"

        def makefile(self, mode):
            return io.BytesIO(b"\x05\x00")

    def connect(address, **kwargs):
        assert address == ("127.0.0.1", 7890), "must not probe transparent port with TCP"
        return Upstream()

    monkeypatch.setattr(proxy.socket, "create_connection", connect)
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        if fail_firewall and args[0] == "iptables":
            raise RuntimeError("firewall setup failed")
        output = "LISTEN 0 4096 10.42.0.1:17890 0.0.0.0:*\n" if args[0] == "ss" else ""
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(proxy, "run", run)
    if fail_firewall:
        with pytest.raises(RuntimeError, match="firewall setup failed"):
            proxy.start("mihomo", ["maas.yicloud.com.cn"])
        assert len(stops) == 2
    else:
        proxy.start("mihomo", ["maas.yicloud.com.cn"])
        assert len(stops) == 1
        assert calls[-1][:7] == ("iptables", "-w", "-t", "nat", "-I", "PREROUTING", "1")
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["rules"][0] == "IP-CIDR,10.42.0.1/32,REJECT,no-resolve"
    assert config["proxies"][0]["port"] == 7890
