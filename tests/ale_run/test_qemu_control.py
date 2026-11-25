from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ale_run.agents.dummy.config import DummyConfig
from ale_run.agents.dummy.deployer import DummyDeployer
from ale_run.base_interface import AgentRunResult
from ale_run.environments import qemu_control as control
from ale_run.executors import sandbox as sandbox_module
from ale_run.executors._guest_control import GuestControl
from ale_run.executors.sandbox import SandboxExecutor


@dataclass
class ControlConfig:
    model: str = "test"


class ControlDeployer:
    def __init__(self, executor):
        self.executor = executor

    async def install(self):
        pass

    async def launch(self, prompt):
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(60)",
            start_new_session=True,
        )
        (Path(self.executor.work_dir) / "child.pid").write_text(str(child.pid))
        await asyncio.Event().wait()
        return AgentRunResult(status="completed")


def record(directory, name, token="current", **values):
    (directory / name).write_text(json.dumps({"run_token": token, **values}))


def test_heartbeat_requires_advancement_and_expires_without_guest_clock():
    watch = control.HeartbeatWatch()
    report = {"sequence": 1, "state": "running", "timestamp": 999999999999}
    assert not watch.fresh(report, now=0)
    assert not watch.fresh(report, now=5)
    report["sequence"] = 2
    assert watch.fresh(report, now=10)
    assert watch.fresh(report, now=54)
    assert not watch.fresh(report, now=56)
    report["sequence"] = 1
    assert not watch.fresh(report, now=57)
    report["sequence"] = True
    assert not watch.fresh(report, now=58)
    report.update(sequence=3, state="stopped")
    assert not watch.fresh(report, now=59)
    report.update(sequence=100, state="running")
    assert not watch.fresh(report, now=500)
    report["sequence"] = 101
    assert watch.fresh(report, now=506)
    assert not watch.fresh({"state": [], "sequence": 102}, now=507)


@pytest.mark.parametrize("value", ["invalid", "[]", '{"run_token":"old"}', "x" * 140000])
def test_invalid_or_foreign_control_is_rejected(tmp_path, value):
    (tmp_path / "heartbeat.json").write_text(value)
    assert control.read_control(tmp_path, "heartbeat.json", "current") is None


@pytest.mark.parametrize("qmp_error", [False, True])
async def test_qmp_handshake_ignores_events_and_unmatched_ids(monkeypatch, qmp_error):
    original = asyncio.create_subprocess_exec
    code = """
import sys,json
print(json.dumps({"QMP": {"version": {}}}), flush=True)
for line in sys.stdin:
 command=json.loads(line)
 print(json.dumps({"event":"STOP"}), flush=True)
 print(json.dumps({"return":{},"id":"unrelated"}), flush=True)
 value={"qmp_capabilities":{},"query-status":{"status":"running","running":True},"query-kvm":{"enabled":True}}
 response={"id": command["id"], "return": value[command["execute"]]}
 if ERROR and command['execute']=='query-status': response={"id":command["id"],"error":{"class":"GenericError"}}
 print(json.dumps(response), flush=True)
""".replace("ERROR", repr(qmp_error))
    children = []

    async def spawn(*args, **kwargs):
        assert args[:4] == ("docker", "exec", "-i", "test-vm")
        process = await original(sys.executable, "-u", "-c", code, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(control.asyncio, "create_subprocess_exec", spawn)
    if qmp_error:
        with pytest.raises(RuntimeError, match="GenericError"):
            await control.query_qmp("test-vm")
    else:
        report = await control.query_qmp("test-vm")
        assert report["status"]["running"] is True
        assert report["kvm"]["enabled"] is True
    assert children[0].returncode is not None


@pytest.mark.parametrize("query", ["qmp", "docker"])
async def test_query_timeout_reaps_client(monkeypatch, query):
    original = asyncio.create_subprocess_exec
    children = []

    async def spawn(*args, **kwargs):
        child = await original(sys.executable, "-c", "import time; time.sleep(60)", **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(control.asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(TimeoutError):
        if query == "qmp":
            await control.query_qmp("test-vm", timeout=0.1)
        else:
            await control.docker_query("inspect", "test-vm", timeout=0.1)
    assert children[0].returncode is not None
    assert not Path(f"/proc/{children[0].pid}").exists()


async def test_docker_query_caps_output(monkeypatch):
    original = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        return await original(sys.executable, "-c", "print('x'*200000)", **kwargs)

    monkeypatch.setattr(control.asyncio, "create_subprocess_exec", spawn)
    with pytest.raises(ValueError, match="exceeded limit"):
        await control.docker_query("logs", "test-vm")


async def test_qmp_running_does_not_claim_guest_healthy(monkeypatch):
    monkeypatch.setattr(
        control,
        "docker_query",
        AsyncMock(
            return_value={
                "returncode": 0,
                "stdout": json.dumps({"Running": True, "OOMKilled": False, "Env": ["SECRET"]}),
            }
        ),
    )
    monkeypatch.setattr(control, "query_qmp", AsyncMock(return_value={"status": {"running": True}}))
    report = await control.inspect_health(SimpleNamespace(id="vm", metadata={}))
    assert report["guest"] == "unknown"
    assert report["container"]["Running"] is True
    assert "SECRET" not in json.dumps(report)


@pytest.mark.parametrize("os_type", ["linux", "windows"])
async def test_shared_control_handshake(tmp_path, os_type):
    token = "current"
    metadata = {
        "provider": "qemu",
        "exchange_host_dir": str(tmp_path),
        "exchange_guest_share": "//host/Data" if os_type == "linux" else r"\\host\Data",
    }
    commands = []

    async def run(command, **kwargs):
        commands.append(command)
        (tmp_path / "control" / token / "guest-probe").write_text(token)
        return subprocess.CompletedProcess(command, 0, "", "")

    sandbox = SimpleNamespace(is_linux=os_type == "linux", metadata=metadata, run_command=run)
    host, guest = await control.prepare_control(sandbox, token)
    assert host == tmp_path / "control" / token
    assert token in guest
    if os_type == "linux":
        assert "cache=none,actimeo=0,soft" in commands[0]
    assert metadata["control_host_dir"] == str(host)


async def test_failed_handshake_blocks_launch(tmp_path):
    sandbox = SimpleNamespace(
        is_linux=True,
        metadata={
            "provider": "qemu",
            "exchange_host_dir": str(tmp_path),
            "exchange_guest_share": "//host/Data",
        },
        run_command=AsyncMock(return_value=subprocess.CompletedProcess("mount", 1, "", "denied")),
    )
    with pytest.raises(RuntimeError, match="handshake failed"):
        await control.prepare_control(sandbox, "current")


@pytest.fixture
def executor(tmp_path, monkeypatch):
    sandbox = SimpleNamespace(
        is_linux=True,
        id="vm",
        endpoint="unused",
        os="linux",
        work_dir_base="/work",
        task_data_root="/data",
        python="python3",
        node="node",
        mcp_server_dir="/mcp",
        cua_server_port=5000,
        metadata={"provider": "qemu"},
        mkdir=AsyncMock(),
        rm=AsyncMock(),
        write_file=AsyncMock(),
        exists=AsyncMock(return_value=False),
        read_text=AsyncMock(side_effect=RuntimeError("CUA unavailable")),
    )
    instance = SandboxExecutor(config=DummyConfig(), work_dir="/work/run", sandbox=sandbox)
    monkeypatch.setattr(instance, "_ship_ale_subtree", AsyncMock())
    monkeypatch.setattr(instance, "_read_pid", AsyncMock(return_value=123))
    monkeypatch.setattr(instance, "_tail_log", AsyncMock(return_value="unavailable"))
    monkeypatch.setattr(
        sandbox_module, "prepare_control", AsyncMock(return_value=(tmp_path, "/share"))
    )
    monkeypatch.setattr(
        sandbox_module, "inspect_health", AsyncMock(return_value={"guest": "unknown"})
    )
    return instance


async def test_cua_loss_uses_advancing_heartbeat_and_shared_result(executor, tmp_path, monkeypatch):
    token = executor._run_token
    record(tmp_path, "heartbeat.json", token, sequence=1, state="running")

    async def command(command, **kwargs):
        if command.startswith("kill -0"):
            record(tmp_path, "heartbeat.json", token, sequence=2, state="running")
            return subprocess.CompletedProcess(command, -1, "", "CUA unavailable")
        return subprocess.CompletedProcess(command, 0, "", "")

    async def tick(delay):
        record(tmp_path, "result.json", token, ok=True, status="completed")
        record(tmp_path, "done.json", token)

    executor.sandbox.run_command = command
    monkeypatch.setattr(sandbox_module.asyncio, "sleep", tick)
    monkeypatch.setattr(sandbox_module, "_LIVENESS_TRANSPORT_GRACE_S", 0)
    result = await executor._run_deployer(deployer_cls=DummyDeployer, prompt="test", timeout_s=60)
    assert result.status == "completed"
    executor.sandbox.read_text.assert_not_awaited()
    assert len(executor.sandbox.metadata["health_probe_history"]) == 1


async def test_qmp_without_live_heartbeat_still_fails_unknown(executor, tmp_path, monkeypatch):
    record(tmp_path, "heartbeat.json", "foreign", sequence=100, state="running")
    executor.sandbox.run_command = AsyncMock(
        side_effect=[
            subprocess.CompletedProcess("spawn", 0, "", ""),
            subprocess.CompletedProcess("alive", -1, "", "unavailable"),
        ]
    )
    monkeypatch.setattr(
        sandbox_module,
        "inspect_health",
        AsyncMock(
            return_value={
                "qmp": {"status": {"running": True}},
                "guest": "unknown",
            }
        ),
    )
    monkeypatch.setattr(sandbox_module, "_LIVENESS_TRANSPORT_GRACE_S", 0)
    result = await executor._run_deployer(deployer_cls=DummyDeployer, prompt="test", timeout_s=60)
    assert result.status == "failed"
    assert "process state unknown" in result.error


async def test_fresh_heartbeat_never_extends_wall_budget(executor, tmp_path, monkeypatch):
    elapsed = [0.0]
    sequence = [1]
    record(tmp_path, "heartbeat.json", executor._run_token, sequence=1, state="running")

    async def tick(delay):
        elapsed[0] += delay
        sequence[0] += 1
        record(
            tmp_path, "heartbeat.json", executor._run_token, sequence=sequence[0], state="running"
        )

    executor.sandbox.run_command = AsyncMock(
        return_value=subprocess.CompletedProcess("spawn", 0, "", "")
    )
    monkeypatch.setattr(sandbox_module, "time", SimpleNamespace(monotonic=lambda: elapsed[0]))
    monkeypatch.setattr(sandbox_module.asyncio, "sleep", tick)
    result = await executor._run_deployer(deployer_cls=DummyDeployer, prompt="test", timeout_s=15)
    assert result.status == "timeout"
    assert "wall budget" in result.error
    assert elapsed[0] <= 25
    executor.sandbox.run_command.assert_awaited_once()


@pytest.mark.parametrize("stopped", [False, True])
async def test_shared_stop_confirms_token_and_empty_tree(executor, tmp_path, stopped):
    executor._control_host = tmp_path
    executor._spawn_attempted = True
    executor.sandbox.write_file.side_effect = RuntimeError("CUA unavailable")
    record(tmp_path, "stopped.json", "old", stopped=True)
    operation = asyncio.create_task(executor.stop_deployer())
    await asyncio.sleep(0.02)
    assert not operation.done()
    assert (tmp_path / "stop.request").read_text() == executor._run_token
    record(tmp_path, "stopped.json", executor._run_token, stopped=stopped, error="live processes")
    if stopped:
        await asyncio.wait_for(operation, 1)
    else:
        with pytest.raises(RuntimeError, match="termination unconfirmed"):
            await asyncio.wait_for(operation, 1)


async def test_cua_can_confirm_stop_when_host_share_write_fails(executor, tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("blocked")
    executor._control_host = blocked
    executor._spawn_attempted = True
    executor.sandbox.read_text = AsyncMock(
        return_value=json.dumps(
            {
                "run_token": executor._run_token,
                "stopped": True,
            }
        )
    )
    await asyncio.wait_for(executor.stop_deployer(), 1)


def test_guest_control_reports_progress_and_terminal_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(GuestControl, "_diagnostics", lambda self: {"run_token": self.token})
    channel = GuestControl(str(tmp_path), "current")
    channel.tick(1, "running", 123)
    (tmp_path / "stop.request").write_text("old")
    assert not channel.stop_requested.wait(0.05)
    (tmp_path / "stop.request").write_text("current")
    assert channel.stop_requested.wait(2)
    heartbeat = control.read_control(tmp_path, "heartbeat.json", "current")
    assert heartbeat["sequence"] == 1
    assert heartbeat["worker_pid"] == 123
    channel.finish({"run_token": "current", "stopped": True}, {"ok": True})
    assert channel.published.wait(1)
    assert control.read_control(tmp_path, "stopped.json", "current")["stopped"] is True
    assert control.read_control(tmp_path, "result.json", "current")["ok"] is True
    assert control.read_control(tmp_path, "done.json", "current") is not None
    channel.thread.join(1)
    assert not channel.thread.is_alive()


def test_blocked_share_does_not_block_supervisor_tick_or_finish(tmp_path, monkeypatch):
    blocked = threading.Event()
    release = threading.Event()

    def write(self, name, value):
        blocked.set()
        release.wait(10)

    monkeypatch.setattr(GuestControl, "_write", write)
    monkeypatch.setattr(GuestControl, "_diagnostics", lambda self: {})
    channel = GuestControl(str(tmp_path), "current")
    try:
        assert blocked.wait(1)
        channel.tick(10, "running", 123)
        channel.finish({"run_token": "current", "stopped": True}, {"ok": True})
        assert not channel.published.is_set()
    finally:
        release.set()
        channel.thread.join(2)


async def test_diagnostics_persist_when_qmp_and_logs_fail(tmp_path, monkeypatch):
    sandbox = SimpleNamespace(id="vm", metadata={})
    monkeypatch.setattr(
        control,
        "inspect_health",
        AsyncMock(
            return_value={
                "guest": "unknown",
                "qmp": {"error": "timeout"},
            }
        ),
    )
    monkeypatch.setattr(control, "docker_query", AsyncMock(side_effect=TimeoutError("timeout")))
    report = await control.collect_diagnostics(sandbox, tmp_path)
    persisted = json.loads(Path(report["path"]).read_text())
    assert persisted["qmp"]["error"] == "timeout"
    assert "TimeoutError" in persisted["runner_log"]["error"]
    assert "not initialized" in persisted["control"]["error"]


async def test_real_supervisor_accepts_shared_stop_and_confirms_descendants(tmp_path):
    if sys.platform != "linux":
        pytest.skip("Linux process containment")
    work = tmp_path / "work"
    share = tmp_path / "control"
    work.mkdir()
    share.mkdir()
    spec = {
        "work_dir": str(work),
        "control_guest_dir": str(share),
        "run_token": "current",
        "timeout_s": 20,
        "install_agent_deps": False,
        "sandbox_local": False,
        "config_module": __name__,
        "config_class": "ControlConfig",
        "config_kwargs": {},
        "deployer_module": __name__,
        "deployer_class": "ControlDeployer",
        "prompt": "test",
        "sandbox_kwargs": {
            "id": "fake",
            "endpoint": "unused",
            "os": "linux",
            "work_dir_base": str(work),
            "task_data_root": "/data",
            "python": sys.executable,
            "node": "unused",
            "mcp_server_dir": "unused",
        },
    }
    spec_path = work / "_spec.json"
    spec_path.write_text(json.dumps(spec))
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([str(Path(__file__).parent), str(Path.cwd())])
    with (work / "entry.log").open("wb") as log:
        supervisor = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "ale_run.executors._sandbox_entry",
            str(spec_path),
            env=environment,
            stdout=log,
            stderr=log,
        )
        try:
            async with asyncio.timeout(8):
                while not (work / "child.pid").exists() or not (share / "heartbeat.json").exists():
                    assert supervisor.returncode is None, (work / "entry.log").read_text()
                    await asyncio.sleep(0.02)
            child = int((work / "child.pid").read_text())
            assert Path(f"/proc/{child}").exists()
            (share / "stop.request").write_text("current")
            await asyncio.wait_for(supervisor.wait(), 8)
            assert supervisor.returncode == 0, (work / "entry.log").read_text()
            assert not Path(f"/proc/{child}").exists()
            assert control.read_control(share, "stopped.json", "current")["stopped"] is True
            assert control.read_control(share, "done.json", "current") is not None
            assert (
                "solver stop requested"
                in control.read_control(share, "result.json", "current")["error"]
            )
        finally:
            (work / "_stop.request").write_text("current")
            try:
                await asyncio.wait_for(supervisor.wait(), 3)
            except TimeoutError:
                supervisor.kill()
                await supervisor.wait()
            if (work / "child.pid").exists():
                try:
                    os.kill(int((work / "child.pid").read_text()), 9)
                except ProcessLookupError:
                    pass


@pytest.mark.parametrize("diagnostics_fail", [False, True])
async def test_lifecycle_diagnoses_before_release_preserving_primary_error(
    tmp_path,
    monkeypatch,
    diagnostics_fail,
):
    from ale_run.orchestration import lifecycle
    from ale_run.orchestration.experiment_spec import AgentSpec, RunUnit

    monkeypatch.chdir(tmp_path)
    task = tmp_path / "tasks/demo/test"
    task.mkdir(parents=True)
    (task / "main.py").write_text("value = 1\n")
    events = []
    sandbox = SimpleNamespace(metadata={"provider": "qemu"}, id="vm")
    env = SimpleNamespace(
        sandbox=sandbox,
        current_phase="task_setup",
        reset_async=AsyncMock(),
        set_phase=lambda phase: None,
        close_async=AsyncMock(side_effect=lambda **kwargs: events.append("release")),
    )
    provider = SimpleNamespace(collect_diagnostics=AsyncMock())

    async def diagnostics(*args):
        events.append("diagnostics")
        assert args[0] is sandbox
        if diagnostics_fail:
            raise TimeoutError("diagnostics timeout")
        return {"path": "qemu.json"}

    provider.collect_diagnostics.side_effect = diagnostics
    monkeypatch.setattr(lifecycle, "resolve_agent", lambda spec: (DummyDeployer, DummyConfig))
    monkeypatch.setattr(lifecycle, "ALEEnv", lambda **kwargs: env)
    monkeypatch.setattr(
        lifecycle,
        "TaskLoader",
        lambda path: SimpleNamespace(
            load=lambda variant: {
                "description": "test",
                "snapshot_name": "fake",
            }
        ),
    )
    monkeypatch.setattr(
        lifecycle, "_stage_task_data", AsyncMock(side_effect=RuntimeError("primary failure"))
    )
    result = await lifecycle.run_one_unit(
        unit=RunUnit("fake", AgentSpec("fake", "fake", executor="sandbox"), "demo/test", 0),
        router=SimpleNamespace(provider_for=lambda snapshot: provider),
        output_root=tmp_path,
    )
    assert result.status == "failed"
    assert "primary failure" in result.error
    assert "diagnostics timeout" not in result.error
    assert events == ["diagnostics", "release"]
