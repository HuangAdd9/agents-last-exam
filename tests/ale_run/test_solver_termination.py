from __future__ import annotations

import asyncio
import ctypes
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ale_run.base_interface import AgentRunResult, GatherReport, SandboxHandle
from ale_run.executors import _sandbox_entry as entry
from ale_run.executors import sandbox as sandbox_module
from ale_run.executors.local import LocalExecutor
from ale_run.executors.docker import DockerExecutor
from ale_run.executors import docker as docker_module
from ale_run.executors.sandbox import SandboxExecutor


@dataclass
class TreeConfig:
    mode: str = "completed"
    model: str = "test"
    orphan: bool = True
    tool_duration_s: float = 0.5


class TreeDeployer:
    hot_artifacts = ()

    def __init__(self, executor):
        self.executor = executor

    async def install(self):
        if self.executor.config.mode == "install_timeout":
            raise TimeoutError("plugin inspection timed out")
        if self.executor.config.mode == "install_failure":
            raise RuntimeError("plugin installation failed")
        if self.executor.config.mode == "install_hang":
            await asyncio.Event().wait()

    async def launch(self, prompt):
        work = Path(self.executor.work_dir)
        (work / "endpoint").write_text(self.executor.sandbox.endpoint)
        (work / "worker.pid").write_text(str(os.getpid()))
        grandchild = (
            "import os, pathlib, signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            f"work = pathlib.Path({str(work)!r})\n"
            "(work / 'grandchild.pid').write_text(str(os.getpid()))\n"
            f"deadline = time.monotonic() + {self.executor.config.tool_duration_s!r}\n"
            "while time.monotonic() < deadline:\n"
            "    with (work / 'heartbeat').open('a') as stream: stream.write('x')\n"
            "    time.sleep(0.02)\n"
            "(work / 'output.complete').write_text('finished')\n"
        )
        child = (
            "import os, pathlib, subprocess, sys, time\n"
            f"work = pathlib.Path({str(work)!r})\n"
            "(work / 'child.pid').write_text(str(os.getpid()))\n"
            f"subprocess.Popen([sys.executable, '-c', {grandchild!r}], "
            "start_new_session=True, stdin=subprocess.DEVNULL)\n"
        )
        if not self.executor.config.orphan:
            child += f"time.sleep({self.executor.config.tool_duration_s!r})\n"
        tool = subprocess.Popen(
            [sys.executable, "-c", child], start_new_session=True,
            stdin=subprocess.DEVNULL,
        )
        if self.executor.config.orphan:
            tool.wait(timeout=5)
        deadline = time.monotonic() + 5
        while not (work / "heartbeat").exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("grandchild did not start")
            await asyncio.sleep(0.01)
        (work / "ready").write_text("ready")
        if self.executor.config.mode == "failed":
            raise RuntimeError("deployer failed with a background tool")
        if self.executor.config.mode == "hang":
            while True:
                try:
                    await asyncio.sleep(30)
                except asyncio.CancelledError:
                    continue
        return AgentRunResult(status="completed")

    @staticmethod
    def parse_artifacts(**kwargs):
        pass


class DiskSandbox(SandboxHandle):
    async def mkdir(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    async def rm(self, paths):
        for path in paths:
            Path(path).unlink(missing_ok=True)

    async def write_file(self, path, content):
        Path(path).write_bytes(content.encode() if isinstance(content, str) else content)

    async def read_text(self, path):
        return Path(path).read_text()

    async def exists(self, path):
        return Path(path).exists()

    async def run_command(self, command, *, timeout=60):
        if command.startswith("chmod +x "):
            spec_path = Path(shlex.split(command)[-1]).with_name("_spec.json")
            returncode = await asyncio.to_thread(
                entry.launch, spec_path, json.loads(spec_path.read_text()),
            )
            return subprocess.CompletedProcess(command, returncode, "", "")
        return await asyncio.to_thread(
            subprocess.run, command, shell=True, capture_output=True, text=True, timeout=timeout,
        )


@pytest.fixture
def disk_executor(tmp_path, monkeypatch):
    if sys.platform != "linux":
        pytest.skip("local shell transport is Linux-only")
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(Path(__file__).parent), str(Path.cwd())]),
    )
    monkeypatch.setattr(entry, "_SYSTEMD_RUNTIME", tmp_path / "no-systemd")
    sandbox = DiskSandbox(
        id="test", endpoint="unused", os="linux", work_dir_base=str(tmp_path),
        task_data_root=str(tmp_path), python=sys.executable, node="unused", mcp_server_dir="unused",
    )
    executor = SandboxExecutor(config=TreeConfig(), work_dir=str(tmp_path / "work"), sandbox=sandbox)
    monkeypatch.setattr(executor, "_ship_ale_subtree", AsyncMock())
    monkeypatch.setattr(sandbox_module, "_POLL_INTERVAL_S", 0.02)
    monkeypatch.setattr(sandbox_module, "_STOP_POLL_S", 0.01)
    yield executor
    work = Path(executor.work_dir)
    if work.exists():
        (work / "_stop.request").write_text(executor._run_token)
    for pid_file in work.glob("*.pid"):
        try:
            os.kill(int(pid_file.read_text()), signal.SIGKILL)
        except ProcessLookupError:
            pass


async def wait_ready(executor):
    async with asyncio.timeout(8):
        while not (Path(executor.work_dir) / "ready").exists():
            await asyncio.sleep(0.01)


def assert_tree_stopped(executor):
    work = Path(executor.work_dir)
    for name in ("worker.pid", "child.pid", "grandchild.pid"):
        assert not Path(f"/proc/{(work / name).read_text()}").exists(), name
    report = json.loads((work / "_stopped.json").read_text())
    assert report["stopped"] is True
    assert report["run_token"] == executor._run_token


@pytest.fixture(params=["sandbox", "local"])
def contained_executor(request, disk_executor):
    if request.param == "local":
        return LocalExecutor(
            config=TreeConfig(), work_dir=disk_executor.work_dir, sandbox=disk_executor.sandbox,
        )
    return disk_executor


@pytest.mark.parametrize("mode", ["completed", "failed", "hang"])
@pytest.mark.parametrize("orphan", [False, True])
async def test_real_detached_orphan_cleanup(contained_executor, mode, orphan):
    executor = contained_executor
    executor.config.mode = mode
    executor.config.orphan = orphan
    if mode == "hang":
        executor.config.tool_duration_s = 30
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        result = await asyncio.wait_for(executor.run_deployer(
            deployer_cls=TreeDeployer, prompt="test", timeout_s=5 if mode == "hang" else 10,
        ), timeout=12)
        assert result.status == ("timeout" if mode == "hang" else mode)
        assert_tree_stopped(executor)
        assert (Path(executor.work_dir) / "output.complete").exists() is (mode != "hang")
        if executor.type == "local":
            assert (Path(executor.work_dir) / "endpoint").read_text() == executor.sandbox.endpoint
        assert unrelated.poll() is None
        heartbeat = (Path(executor.work_dir) / "heartbeat").read_bytes()
        await asyncio.sleep(0.06)
        assert (Path(executor.work_dir) / "heartbeat").read_bytes() == heartbeat
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


@pytest.mark.parametrize("orphan", [False, True])
async def test_background_output_finishes_after_session_exit(contained_executor, orphan):
    executor = contained_executor
    executor.config.orphan = orphan
    executor.config.tool_duration_s = 1.5
    work = Path(executor.work_dir)
    operation = asyncio.create_task(executor.run_deployer(
        deployer_cls=TreeDeployer, prompt="test", timeout_s=8,
    ))
    async with asyncio.timeout(5):
        while not (work / "_worker_result.json").exists():
            await asyncio.sleep(0.01)
    assert not operation.done()
    assert not (work / "_done.marker").exists()
    assert not (work / "_stop.request").exists()
    assert (await operation).status == "completed"
    assert (work / "output.complete").read_text() == "finished"
    assert_tree_stopped(executor)


async def test_background_work_uses_remaining_budget(contained_executor):
    executor = contained_executor
    executor.config.tool_duration_s = 30
    work = Path(executor.work_dir)
    started = time.monotonic()
    result = await asyncio.wait_for(executor.run_deployer(
        deployer_cls=TreeDeployer, prompt="test", timeout_s=5,
    ), timeout=9)
    assert result.status == "timeout"
    assert time.monotonic() - started >= 5
    assert not (work / "output.complete").exists()
    assert_tree_stopped(executor)


@pytest.mark.parametrize("mode", ["completed", "failed"])
async def test_finished_worker_reclaims_detached_tools_after_drain_grace(disk_executor, mode):
    executor = disk_executor
    work = Path(executor.work_dir)
    work.mkdir()
    spec_path = work / "_spec.json"
    spec_path.write_text(json.dumps({
        "work_dir": str(work), "run_token": executor._run_token, "timeout_s": 20,
        "deployer_module": __name__, "deployer_class": "TreeDeployer",
        "config_module": __name__, "config_class": "TreeConfig",
        "config_kwargs": asdict(TreeConfig(mode=mode, tool_duration_s=30)),
        "sandbox_kwargs": asdict(executor.sandbox), "prompt": "test",
        "install_agent_deps": False,
    }))
    bootstrap = (
        "from ale_run.executors import _sandbox_entry as entry; "
        "entry._BACKGROUND_DRAIN_TIMEOUT_S = 0.25; "
        "raise SystemExit(entry.main())"
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        result = await asyncio.to_thread(
            subprocess.run, [sys.executable, "-c", bootstrap, str(spec_path)],
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, result.stderr
        out = json.loads((work / "_result.json").read_text())
        assert out["status"] == mode
        assert out["duration_s"] < 10
        assert out["ok"] is (mode == "completed")
        assert not (work / "output.complete").exists()
        assert (work / "_done.marker").exists()
        assert "reclaiming remaining background processes" in result.stderr
        assert_tree_stopped(executor)
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


@pytest.mark.parametrize("mode", ["install_timeout", "install_failure", "install_hang"])
async def test_real_install_failure_never_marks_launch(contained_executor, mode):
    executor = contained_executor
    executor.config.mode = mode
    result = await asyncio.wait_for(executor.run_deployer(
        deployer_cls=TreeDeployer, prompt="test", timeout_s=2,
    ), timeout=8)
    assert result.status == ("timeout" if mode == "install_hang" else "failed")
    work = Path(executor.work_dir)
    assert not (work / "_agent_started").exists()
    assert not (work / "worker.pid").exists()
    assert json.loads((work / "_stopped.json").read_text())["stopped"] is True
    if mode != "install_hang":
        assert json.loads((work / "_worker_result.json").read_text())["phase"] == "install"


@pytest.mark.parametrize("mode", ["install_timeout", "install_failure", "install_hang"])
async def test_lifecycle_real_install_failure_skips_evaluation(
    tmp_path, monkeypatch, contained_executor, mode,
):
    from ale_run.orchestration import lifecycle
    from ale_run.orchestration.experiment_spec import AgentSpec, RunUnit

    monkeypatch.chdir(tmp_path)
    task_directory = tmp_path / "tasks" / "demo" / "test"
    task_directory.mkdir(parents=True)
    (task_directory / "main.py").write_text("value = 1\n")
    executor = contained_executor
    executor.config.mode = mode
    env = SimpleNamespace(
        sandbox=executor.sandbox, session=None, current_phase=None, reset_async=AsyncMock(),
        close_async=AsyncMock(), reset_session=Mock(),
    )
    env.set_phase = lambda phase: setattr(env, "current_phase", phase)
    driver = SimpleNamespace(
        setup=AsyncMock(), close=AsyncMock(), evaluate=AsyncMock(), evaluation_timeout_s=None,
    )
    executor.gather_dir = AsyncMock(return_value=GatherReport(transport="local"))
    reference = AsyncMock()
    monkeypatch.setattr(lifecycle, "resolve_agent", lambda spec: (TreeDeployer, TreeConfig))
    monkeypatch.setattr(lifecycle, "ALEEnv", lambda **kwargs: env)
    monkeypatch.setattr(lifecycle, "TaskLoader", lambda path: SimpleNamespace(load=lambda variant: {
        "description": "test", "snapshot_name": "fake",
    }))
    monkeypatch.setattr(lifecycle, "TaskDriver", lambda **kwargs: driver)
    monkeypatch.setattr(lifecycle, "_build_executor", lambda **kwargs: executor)
    monkeypatch.setattr(lifecycle, "_stage_task_data", AsyncMock())
    monkeypatch.setattr(lifecycle, "stage_reference", reference)
    result = await asyncio.wait_for(lifecycle.run_one_unit(
        unit=RunUnit("fake", AgentSpec("fake", "fake", executor=executor.type), "demo/test", 0),
        router=SimpleNamespace(provider_for=lambda snapshot: None),
        output_root=tmp_path, wall_time_s=3,
    ), timeout=10)
    metadata = json.loads((result.run_dir / "run.json").read_text())
    evaluation = json.loads((result.run_dir / "eval_result.json").read_text())
    assert result.status == metadata["status"] == "failed"
    assert result.score is None and metadata["score"] is None
    assert result.eval_status == "not_executed"
    assert evaluation["eval_status"] == "not_executed" and evaluation["score"] is None
    reference.assert_not_awaited()
    driver.evaluate.assert_not_awaited()
    executor.gather_dir.assert_awaited_once()
    env.close_async.assert_awaited_once()


@pytest.mark.parametrize("finishes", [False, True])
async def test_sigkilled_worker_without_result_preserves_live_orphan(disk_executor, finishes):
    executor = disk_executor
    work = Path(executor.work_dir)
    work.mkdir()
    budget_s = 4
    spec_path = work / "_spec.json"
    spec_path.write_text(json.dumps({
        "work_dir": str(work), "run_token": executor._run_token, "timeout_s": budget_s,
        "deployer_module": __name__, "deployer_class": "TreeDeployer",
        "config_module": __name__, "config_class": "TreeConfig",
        "config_kwargs": asdict(TreeConfig(
            mode="hang", tool_duration_s=1.5 if finishes else 30,
        )),
        "sandbox_kwargs": asdict(executor.sandbox), "prompt": "test",
        "install_agent_deps": False,
    }))
    with (work / "_entry.log").open("wb") as log:
        supervisor = subprocess.Popen(
            [sys.executable, "-m", "ale_run.executors._sandbox_entry", str(spec_path)],
            stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True,
        )
    try:
        await wait_ready(executor)
        os.kill(int((work / "worker.pid").read_text()), signal.SIGKILL)
        async with asyncio.timeout(2):
            while "without _worker_result.json" not in (work / "_entry.log").read_text():
                await asyncio.sleep(0.01)
        heartbeat = (work / "heartbeat").read_bytes()
        await asyncio.sleep(0.2)
        assert (work / "heartbeat").read_bytes() != heartbeat
        assert supervisor.poll() is None
        assert not (work / "_worker_result.json").exists()
        assert not (work / "_done.marker").exists()
        assert not (work / "_stopped.json").exists()
        assert not (work / "_stop.request").exists()
        await asyncio.to_thread(supervisor.wait, timeout=budget_s + 5)
        assert supervisor.returncode == 0
        result = json.loads((work / "_result.json").read_text())
        assert result["status"] == ("failed" if finishes else "timeout")
        assert result["exit_code"] == -signal.SIGKILL
        assert "worker exited with code -9" in result["error"]
        assert "without _worker_result.json" in result["error"]
        assert (work / "output.complete").exists() is finishes
        if not finishes:
            assert result["duration_s"] >= budget_s
            assert "solver budget exceeded" in result["error"]
        assert_tree_stopped(executor)
        report = json.loads((work / "_stopped.json").read_text())
        assert "worker exited with code -9" in report["error"]
    finally:
        (work / "_stop.request").write_text(executor._run_token)
        try:
            await asyncio.to_thread(supervisor.wait, timeout=5)
        except subprocess.TimeoutExpired:
            supervisor.kill()
            await asyncio.to_thread(supervisor.wait, timeout=5)


@pytest.mark.parametrize("cause", ["cancel", "outer_timeout", "transport_error"])
async def test_real_cleanup_on_host_interruption(disk_executor, monkeypatch, cause):
    executor = disk_executor
    executor.config.mode = "hang"
    operation = asyncio.create_task(executor.run_deployer(
        deployer_cls=TreeDeployer, prompt="test", timeout_s=20,
    ))
    await wait_ready(executor)
    work = Path(executor.work_dir)
    assert os.getpgid(int((work / "grandchild.pid").read_text())) != os.getpgid(
        int((work / "worker.pid").read_text()),
    )
    if cause == "transport_error":
        monkeypatch.setattr(executor.sandbox, "run_command", AsyncMock(side_effect=RuntimeError("rpc")))
        with pytest.raises(RuntimeError, match="rpc"):
            await asyncio.wait_for(operation, 8)
    elif cause == "outer_timeout":
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(operation, 0.01)
    else:
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
    assert_tree_stopped(executor)


@pytest.mark.parametrize("os_type", ["linux", "windows"])
@pytest.mark.parametrize("report", [None, {}, {"run_token": "stale", "stopped": True},
                                     {"stopped": False}, {"stopped": "true"}])
async def test_unconfirmed_stop_fails_closed(tmp_path, monkeypatch, os_type, report):
    sandbox = SimpleNamespace(is_linux=os_type == "linux", write_file=AsyncMock())
    executor = SandboxExecutor(config=TreeConfig(), work_dir=str(tmp_path), sandbox=sandbox)
    executor._spawn_attempted = True
    if report is None:
        sandbox.read_text = AsyncMock(side_effect=RuntimeError("transport unavailable"))
    else:
        report = {"run_token": executor._run_token, **report}
        sandbox.read_text = AsyncMock(return_value=json.dumps(report))
    monkeypatch.setattr(sandbox_module, "_STOP_TIMEOUT_S", 0.04)
    monkeypatch.setattr(sandbox_module, "_STOP_POLL_S", 0.005)
    with pytest.raises(RuntimeError, match="termination unconfirmed"):
        await executor.stop_deployer()


async def test_repeated_cancellation_waits_for_confirmation(tmp_path, monkeypatch):
    sandbox = SimpleNamespace(is_linux=True, write_file=AsyncMock())
    executor = SandboxExecutor(config=TreeConfig(), work_dir=str(tmp_path), sandbox=sandbox)
    executor._spawn_attempted = True
    entered, release = asyncio.Event(), asyncio.Event()

    async def read_report(path):
        entered.set()
        await release.wait()
        return json.dumps({"run_token": executor._run_token, "stopped": True})

    sandbox.read_text = read_report
    operation = asyncio.create_task(executor.stop_deployer())
    await entered.wait()
    operation.cancel()
    await asyncio.sleep(0)
    operation.cancel()
    await asyncio.sleep(0)
    assert not operation.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await operation
    await executor.stop_deployer()


@pytest.mark.parametrize("missing_pid", [False, True])
async def test_stop_during_spawn_before_pid_is_known(disk_executor, monkeypatch, missing_pid):
    executor = disk_executor
    executor.config.mode = "hang"
    spawned = asyncio.Event()
    original = executor.sandbox.run_command

    async def spawn_without_response(command, *, timeout=60):
        result = await original(command, timeout=timeout)
        spawned.set()
        await asyncio.Event().wait()
        return result

    if missing_pid:
        monkeypatch.setattr(executor, "_read_pid", AsyncMock(return_value=None))
        operation = asyncio.create_task(executor.run_deployer(
            deployer_cls=TreeDeployer, prompt="test", timeout_s=10,
        ))
        result = await asyncio.wait_for(operation, 10)
        assert "usable PID" in result.error
    else:
        monkeypatch.setattr(executor.sandbox, "run_command", spawn_without_response)
        operation = asyncio.create_task(executor.run_deployer(
            deployer_cls=TreeDeployer, prompt="test", timeout_s=10,
        ))
        await asyncio.wait_for(spawned.wait(), 5)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
    report = json.loads((Path(executor.work_dir) / "_stopped.json").read_text())
    assert report["stopped"] is True


async def test_late_launcher_cannot_restart_stopped_solver(disk_executor):
    executor = disk_executor
    result = await executor.run_deployer(deployer_cls=TreeDeployer, prompt="test", timeout_s=10)
    assert result.status == "completed"
    work = Path(executor.work_dir)
    heartbeat = (work / "heartbeat").read_bytes()
    duplicate = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-m", "ale_run.executors._sandbox_entry", str(work / "_spec.json")],
        capture_output=True, timeout=8,
    )
    assert duplicate.returncode == 0, duplicate.stderr
    assert (work / "heartbeat").read_bytes() == heartbeat
    assert_tree_stopped(executor)


@pytest.mark.parametrize("mode", [
    "completed", "hang", "failed", "cancel", "unconfirmed", "failed_unconfirmed",
    "sandbox_unconfirmed", "sandbox_exception_unconfirmed", "sandbox_timeout",
    "sandbox_completed_recovery", "sandbox_failed_recovery", "sandbox_failed_recovery_failed",
    "sandbox_timeout_recovery_failed", "sandbox_timeout_gather_failed", "sandbox_timeout_parse_failed",
    "install_timeout", "install_failure", "install_outer_timeout", "sandbox_install_timeout",
    "launch_marker_unavailable",
])
async def test_lifecycle_stops_before_reference_and_evaluation(tmp_path, monkeypatch, mode):
    from ale_run.orchestration import lifecycle
    from ale_run.orchestration.experiment_spec import AgentSpec, RunUnit

    monkeypatch.chdir(tmp_path)
    task_directory = tmp_path / "tasks" / "demo" / "test"
    task_directory.mkdir(parents=True)
    (task_directory / "main.py").write_text("value = 1\n")
    expected_revision = lifecycle.task_revision(task_directory)
    events = []
    started = asyncio.Event()
    sandbox = SimpleNamespace(id="fake", metadata={"image_provenance": {
        "provider": "qemu", "image_revision": "test-image",
    }}, os="linux")
    env = SimpleNamespace(sandbox=sandbox, session=None, current_phase=None,
                          reset_async=AsyncMock(), close_async=AsyncMock(), reset_session=Mock())
    env.set_phase = lambda phase: setattr(env, "current_phase", phase)
    if mode.startswith("sandbox_"):
        executor = SandboxExecutor(config=TreeConfig(), work_dir=str(tmp_path), sandbox=sandbox)
        executor.stopped = False
    else:
        executor = SimpleNamespace(work_dir=str(tmp_path), stopped=False)

    async def run(**kwargs):
        events.append("run")
        started.set()
        if mode in ("hang", "cancel", "install_outer_timeout"):
            await asyncio.Event().wait()
        if mode in ("install_timeout", "install_failure", "sandbox_install_timeout",
                    "launch_marker_unavailable"):
            return AgentRunResult(status="failed", error="installation failed")
        if mode in ("failed", "failed_unconfirmed", "sandbox_exception_unconfirmed"):
            raise RuntimeError("launch failure")
        if mode == "sandbox_unconfirmed":
            return AgentRunResult(
                status="failed", error="original worker failure", pid=42, exit_code=-9,
            )
        if mode.startswith("sandbox_timeout"):
            return AgentRunResult(status="timeout", error="agent timed out")
        if mode.startswith("sandbox_failed_recovery"):
            return AgentRunResult(status="failed", error="gateway exited")
        return AgentRunResult(status="completed")

    async def stop():
        events.append("stop")
        await asyncio.sleep(0.01)
        if mode.endswith("unconfirmed"):
            records = [
                json.loads(line)
                for line in next(tmp_path.rglob("events.jsonl")).read_text().splitlines()
            ]
            assert any(record["type"] == "agent_run_result" for record in records)
            raise RuntimeError(f"termination unconfirmed on stop {events.count('stop')}")
        executor.stopped = True
        events.append("stopped")

    async def reference(**kwargs):
        assert executor.stopped
        events.append("reference")

    async def evaluate():
        assert executor.stopped
        events.append("evaluate")
        return {"score": 1.0}

    if isinstance(executor, SandboxExecutor):
        executor._run_deployer = run
    else:
        executor.run_deployer = run
    executor.stop_deployer = stop
    executor.gather_dir = AsyncMock(return_value=GatherReport(transport="local"))
    if mode.startswith(("sandbox_timeout", "sandbox_completed_recovery", "sandbox_failed_recovery")):
        async def recover(sandbox, work_dir):
            assert executor.stopped
            events.append("recover")
            if mode.endswith("recovery_failed"):
                raise RuntimeError("transcript recovery failed")
            return {"transcript.jsonl": "expected"}
        def validate(work_dir, manifest):
            events.append("validate")
            assert manifest == {"transcript.jsonl": "expected"}
            if mode.endswith("gather_failed"):
                raise RuntimeError("transcript gather failed")
        monkeypatch.setattr(TreeDeployer, "recover_stopped_artifacts", recover, raising=False)
        monkeypatch.setattr(TreeDeployer, "validate_recovered_artifacts", validate, raising=False)
        if mode == "sandbox_timeout_parse_failed":
            def fail_parse(**kwargs):
                raise ValueError("malformed transcript")
            monkeypatch.setattr(TreeDeployer, "parse_artifacts", fail_parse)
    driver = SimpleNamespace(
        setup=AsyncMock(), close=AsyncMock(), evaluate=evaluate, evaluation_timeout_s=None,
    )
    monkeypatch.setattr(lifecycle, "resolve_agent", lambda spec: (TreeDeployer, TreeConfig))
    monkeypatch.setattr(lifecycle, "ALEEnv", lambda **kwargs: env)
    monkeypatch.setattr(lifecycle, "TaskLoader", lambda path: SimpleNamespace(load=lambda variant: {
        "description": "test", "snapshot_name": "fake",
    }))
    monkeypatch.setattr(lifecycle, "TaskDriver", lambda **kwargs: driver)
    monkeypatch.setattr(lifecycle, "_build_executor", lambda **kwargs: executor)
    monkeypatch.setattr(lifecycle, "_stage_task_data", AsyncMock())
    monkeypatch.setattr(lifecycle, "pull_agent_output", AsyncMock())
    monkeypatch.setattr(lifecycle, "stage_reference", reference)
    launch_check = AsyncMock(return_value=mode not in (
        "install_timeout", "install_failure", "install_outer_timeout", "sandbox_install_timeout",
    ))
    if mode == "launch_marker_unavailable":
        launch_check.side_effect = RuntimeError("sandbox transport unavailable")
    monkeypatch.setattr(lifecycle, "_agent_launch_started", launch_check)
    operation = asyncio.create_task(lifecycle.run_one_unit(
        unit=RunUnit("fake", AgentSpec("fake", "fake", executor="sandbox"), "demo/test", 0),
        router=SimpleNamespace(provider_for=lambda snapshot: None),
        output_root=tmp_path, wall_time_s=1, cleanup_mode="keep",
    ))
    await started.wait()
    if mode == "cancel":
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert executor.stopped
    else:
        result = await asyncio.wait_for(operation, 5)
        metadata = json.loads((result.run_dir / "run.json").read_text())
        assert metadata["task"]["revision"] == expected_revision
        assert metadata["environment"]["image_revision"] == "test-image"
        if mode.endswith("unconfirmed"):
            assert result.status == "failed"
            assert result.eval_status == "not_executed"
            records = [
                json.loads(line)
                for line in (result.run_dir / "events.jsonl").read_text().splitlines()
            ]
            record_types = [record["type"] for record in records]
            assert record_types.index("agent_run_result") < record_types.index(
                "agent_stop_failed",
            )
            assert "agent_finished" not in record_types
            traceback_text = metadata["termination"]["error"]["traceback"]
            if mode == "sandbox_unconfirmed":
                primary = next(
                    record["data"] for record in records if record["type"] == "agent_run_result"
                )
                assert primary["error"] == "original worker failure"
                assert primary["pid"] == 42
                assert primary["exit_code"] == -9
                assert "original worker failure" in traceback_text
            if mode in ("failed_unconfirmed", "sandbox_exception_unconfirmed"):
                assert "launch failure" in traceback_text
            if mode.startswith("sandbox_"):
                assert "unconfirmed on stop 1" in traceback_text
                assert "unconfirmed on stop 2" in traceback_text
        elif mode in ("sandbox_timeout_recovery_failed", "sandbox_timeout_gather_failed",
                      "sandbox_failed_recovery_failed",
                      "install_timeout", "install_failure", "install_outer_timeout",
                      "sandbox_install_timeout", "launch_marker_unavailable"):
            assert result.status == "failed"
            assert result.eval_status == "not_executed"
            assert result.score is None
            assert "reference" not in events and "evaluate" not in events
            if mode in ("sandbox_timeout_recovery_failed", "sandbox_failed_recovery_failed"):
                executor.gather_dir.assert_awaited_once()
            if mode in ("install_timeout", "install_failure", "install_outer_timeout",
                        "sandbox_install_timeout", "launch_marker_unavailable"):
                assert metadata["termination"]["phase"] == "agent_start"
                assert "agent launch not confirmed" in result.error
                executor.gather_dir.assert_awaited_once()
                records = [json.loads(line) for line in (result.run_dir / "events.jsonl").read_text().splitlines()]
                finished = next(record["data"] for record in records if record["type"] == "agent_finished")
                assert finished["status"] == "failed"
                assert finished["launch_confirmed"] is False
        elif mode == "sandbox_timeout_parse_failed":
            assert result.status == "failed"
            assert result.eval_status == "failed"
            assert result.score is None
            assert "evaluate" in events
        else:
            assert result.status == {"hang": "timeout", "failed": "failed", "sandbox_timeout": "timeout",
                                     "sandbox_failed_recovery": "failed"}.get(mode, "completed")
            assert events.index("stopped") < events.index("reference") < events.index("evaluate")
            if mode == "sandbox_timeout":
                assert events.index("stopped") < events.index("recover") < events.index("reference")
                assert events.index("recover") < events.index("validate") < events.index("reference")
                assert metadata["termination"]["phase"] == "agent_run"
            if mode in ("sandbox_completed_recovery", "sandbox_failed_recovery"):
                assert events.index("stopped") < events.index("recover") < events.index("validate") < events.index("reference")
    if mode == "cancel" or mode.endswith("unconfirmed"):
        assert "reference" not in events
        assert "evaluate" not in events
    env.close_async.assert_awaited_once_with(mode="keep")


@pytest.mark.parametrize("executor_type", ["local", "docker"])
async def test_launch_marker_host_executors(tmp_path, executor_type):
    from ale_run.orchestration.lifecycle import _agent_launch_started

    executor_class = LocalExecutor if executor_type == "local" else DockerExecutor
    executor = executor_class(config=None, work_dir=str(tmp_path), sandbox=None)
    (tmp_path / "_spec.json").write_text(json.dumps({"run_token": "current"}))
    marker = tmp_path / "_agent_started"
    assert not await _agent_launch_started(executor)
    marker.write_text("stale")
    with pytest.raises(RuntimeError, match="invalid agent launch marker"):
        await _agent_launch_started(executor)
    marker.write_text("current")
    assert await _agent_launch_started(executor)


@pytest.mark.parametrize("os_type", ["linux", "windows"])
async def test_launch_marker_sandbox_executor(tmp_path, os_type):
    from ale_run.orchestration.lifecycle import _agent_launch_started

    sandbox = SimpleNamespace(is_linux=os_type == "linux", read_text=AsyncMock())
    executor = SandboxExecutor(config=None, work_dir="work", sandbox=sandbox)
    sandbox.read_text.side_effect = FileNotFoundError
    assert not await _agent_launch_started(executor)
    sandbox.read_text.side_effect = None
    sandbox.read_text.return_value = "stale"
    with pytest.raises(RuntimeError, match="invalid agent launch marker"):
        await _agent_launch_started(executor)
    sandbox.read_text.return_value = executor._run_token
    assert await _agent_launch_started(executor)
    sandbox.read_text.assert_awaited_with(
        "work/_agent_started" if os_type == "linux" else "work\\_agent_started",
    )
    sandbox.read_text.side_effect = RuntimeError("transport unavailable")
    with pytest.raises(RuntimeError, match="transport unavailable"):
        await _agent_launch_started(executor)


async def test_local_executor_without_launch_is_safe_to_stop(tmp_path):
    executor = LocalExecutor(config=None, work_dir=str(tmp_path), sandbox=None)
    await executor.stop_deployer()


@pytest.mark.parametrize("cause", ["cancel", "outer_timeout"])
async def test_local_interruption_stops_actual_tools(disk_executor, cause):
    executor = LocalExecutor(
        config=TreeConfig(mode="hang"), work_dir=disk_executor.work_dir,
        sandbox=disk_executor.sandbox, env={"MODEL_API_KEY": "test-secret"},
    )
    operation = asyncio.create_task(executor.run_deployer(
        deployer_cls=TreeDeployer, prompt="test", timeout_s=20,
    ))
    await wait_ready(executor)
    if cause == "cancel":
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
    else:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(operation, 0.01)
    assert_tree_stopped(executor)
    assert not (Path(executor.work_dir) / "_secrets.json").exists()


@pytest.mark.parametrize("mode", ["completed", "failed", "timeout", "cancel"])
async def test_docker_removes_and_verifies_container(tmp_path, monkeypatch, mode):
    events = []
    running = asyncio.Event()
    created = False
    worker = SimpleNamespace(returncode=0)

    def command(argv, **kwargs):
        nonlocal created
        events.append(argv[:3])
        if argv[1] == "create":
            created = True
            return subprocess.CompletedProcess(argv, 0, "container-id", "")
        if argv[1] == "rm":
            created = False
        if argv[1:3] == ["container", "ls"]:
            assert not created
        return subprocess.CompletedProcess(argv, 0, "", "")

    async def communicate():
        running.set()
        if mode in ("timeout", "cancel") and created:
            if mode == "timeout":
                raise asyncio.TimeoutError
            await asyncio.Event().wait()
        (tmp_path / "_result.json").write_text(json.dumps({
            "ok": mode != "failed", "status": "failed" if mode == "failed" else "completed",
        }))
        return b"", b""

    worker.communicate = communicate
    monkeypatch.setattr(docker_module, "_docker_available", lambda: True)
    monkeypatch.setattr(docker_module, "_image_present", lambda image: True)
    monkeypatch.setattr(docker_module.subprocess, "run", command)
    monkeypatch.setattr(docker_module.asyncio, "create_subprocess_exec", AsyncMock(return_value=worker))
    sandbox = SandboxHandle(
        id="fake", endpoint="http://remote:5000", os="linux", work_dir_base="/work",
        task_data_root="/data", python=sys.executable, node="unused", mcp_server_dir="unused",
    )
    executor = DockerExecutor(config=TreeConfig(), work_dir=str(tmp_path), sandbox=sandbox)
    operation = asyncio.create_task(executor.run_deployer(
        deployer_cls=TreeDeployer, prompt="test", timeout_s=10,
    ))
    await running.wait()
    if mode == "cancel":
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
    else:
        assert (await operation).status == mode
    assert events[-2][1] == "rm"
    assert events[-1][1:3] == ["container", "ls"]
    assert not (tmp_path / "_secrets.json").exists()


@pytest.mark.parametrize("inspection", ["alive", "unavailable"])
async def test_docker_stop_requires_successful_absence_check(tmp_path, monkeypatch, inspection):
    def command(argv, **kwargs):
        if argv[1:3] == ["container", "ls"]:
            return subprocess.CompletedProcess(
                argv, 0 if inspection == "alive" else 1,
                "container-id" if inspection == "alive" else "", "daemon unavailable",
            )
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(docker_module.subprocess, "run", command)
    executor = DockerExecutor(config=None, work_dir=str(tmp_path), sandbox=None)
    executor._container_name = "ale-test"
    with pytest.raises(RuntimeError, match="termination unconfirmed"):
        await executor.stop_deployer()


async def test_docker_cancellation_during_create_cannot_leave_late_container(tmp_path, monkeypatch):
    release = asyncio.Event()
    events = []

    async def create():
        await release.wait()
        events.append("created")

    def command(argv, **kwargs):
        events.append(argv[1])
        assert "created" in events
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(docker_module.subprocess, "run", command)
    executor = DockerExecutor(config=None, work_dir=str(tmp_path), sandbox=None)
    executor._container_name = "ale-test"
    executor._create_task = asyncio.create_task(create())
    operation = asyncio.create_task(executor.stop_deployer())
    await asyncio.sleep(0)
    operation.cancel()
    await asyncio.sleep(0)
    assert not operation.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert events == ["created", "rm", "container"]


@pytest.fixture
def windows_kernel(monkeypatch):
    kernel = SimpleNamespace(**{name: Mock(return_value=1) for name in (
        "CreateJobObjectW", "SetInformationJobObject", "OpenProcess",
        "AssignProcessToJobObject", "TerminateJobObject", "QueryInformationJobObject", "CloseHandle",
    )})
    monkeypatch.setattr(ctypes, "WinDLL", Mock(return_value=kernel), raising=False)
    monkeypatch.setattr(ctypes, "WinError", lambda: OSError("Windows API failed"), raising=False)
    return kernel


def test_windows_job_disallows_breakaway_and_waits_for_empty(windows_kernel):
    observed = []

    def set_limits(handle, info_class, pointer, size):
        assert info_class == 9
        assert pointer._obj.BasicLimitInformation.LimitFlags == 0x2000
        return 1

    def query(handle, info_class, pointer, size, returned):
        observed.append("query")
        pointer._obj.ActiveProcesses = 1 if len(observed) == 1 else 0
        return 1

    windows_kernel.SetInformationJobObject.side_effect = set_limits
    windows_kernel.QueryInformationJobObject.side_effect = query
    job = entry._WindowsJob()
    job.assign(123)
    job.stop()
    assert len(observed) == 2
    windows_kernel.AssignProcessToJobObject.assert_called_once()
    windows_kernel.TerminateJobObject.assert_called_once()


def test_windows_install_failure_stops_remaining_processes(tmp_path, monkeypatch, windows_kernel):
    worker = Mock(pid=123, returncode=0)
    worker.poll.return_value = 0
    worker.stdin.closed = False
    monkeypatch.setattr(entry.sys, "platform", "win32")
    monkeypatch.setattr(entry.subprocess, "Popen", Mock(return_value=worker))
    (tmp_path / "_worker_result.json").write_text(json.dumps({
        "ok": False, "status": "failed", "phase": "install", "error": "plugin timed out",
    }))
    result = entry.supervise(tmp_path / "_spec.json", {
        "work_dir": str(tmp_path), "run_token": "test", "timeout_s": 30,
    })
    assert result["status"] == "failed"
    assert result["phase"] == "install"
    windows_kernel.TerminateJobObject.assert_called_once()
    assert json.loads((tmp_path / "_stopped.json").read_text())["stopped"] is True


@pytest.mark.parametrize("failure", ["CreateJobObjectW", "SetInformationJobObject",
                                    "OpenProcess", "AssignProcessToJobObject",
                                    "TerminateJobObject", "QueryInformationJobObject"])
def test_windows_job_api_errors_fail_closed(windows_kernel, failure):
    getattr(windows_kernel, failure).return_value = 0
    with pytest.raises(OSError, match="Windows API failed"):
        job = entry._WindowsJob()
        job.assign(123)
        job.stop()


def test_windows_job_live_descendants_fail_closed(windows_kernel, monkeypatch):
    def query(handle, info_class, pointer, size, returned):
        pointer._obj.ActiveProcesses = 1
        return 1

    windows_kernel.QueryInformationJobObject.side_effect = query
    monkeypatch.setattr(entry, "_PROCESS_STOP_TIMEOUT_S", 0.03)
    with pytest.raises(RuntimeError, match="processes remain"):
        entry._WindowsJob().stop()
    windows_kernel.CloseHandle.assert_called_once()


@pytest.mark.parametrize("finishes", [False, True])
@pytest.mark.parametrize("missing_result", [False, True])
def test_windows_waits_for_background_work_before_cleanup(
    tmp_path, monkeypatch, windows_kernel, finishes, missing_result,
):
    worker = Mock(pid=123)
    worker.returncode = -9 if missing_result else 0
    worker.poll.return_value = 0
    worker.stdin.closed = False
    elapsed = [0.0]
    monkeypatch.setattr(entry.sys, "platform", "win32")
    monkeypatch.setattr(entry.subprocess, "Popen", Mock(return_value=worker))
    monkeypatch.setattr(entry.time, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(entry.time, "sleep", lambda delay: elapsed.__setitem__(0, elapsed[0] + delay))

    def query(handle, info_class, pointer, size, returned):
        pointer._obj.ActiveProcesses = int(
            not windows_kernel.TerminateJobObject.called
            and (not finishes or elapsed[0] < 0.5)
        )
        return 1

    windows_kernel.QueryInformationJobObject.side_effect = query
    if not missing_result:
        (tmp_path / "_worker_result.json").write_text(json.dumps({
            "ok": True, "status": "completed", "transcript_path": "transcript.jsonl",
        }))
    out = entry.supervise(tmp_path / "_spec.json", {
        "work_dir": str(tmp_path), "run_token": "test", "timeout_s": 1,
    })
    assert out["status"] == (
        ("failed" if missing_result else "completed") if finishes else "timeout"
    )
    if missing_result:
        assert "worker exited with code -9" in out["error"]
    else:
        assert out["transcript_path"] == "transcript.jsonl"
    if finishes:
        assert 0.5 <= out["duration_s"] < 1
        windows_kernel.TerminateJobObject.assert_not_called()
    else:
        assert out["duration_s"] >= 1
        windows_kernel.TerminateJobObject.assert_called_once()
    assert json.loads((tmp_path / "_stopped.json").read_text())["stopped"] is True


@pytest.mark.parametrize("missing_result", [False, True])
def test_worker_result_and_cleanup_exceptions_remain_visible(
    tmp_path, monkeypatch, windows_kernel, missing_result,
):
    worker = Mock(pid=123, returncode=-9)
    worker.poll.return_value = -9
    worker.stdin.closed = False
    monkeypatch.setattr(entry.sys, "platform", "win32")
    monkeypatch.setattr(entry.subprocess, "Popen", Mock(return_value=worker))
    windows_kernel.QueryInformationJobObject.side_effect = RuntimeError("containment query failed")
    windows_kernel.TerminateJobObject.return_value = 0
    if not missing_result:
        (tmp_path / "_worker_result.json").write_text("invalid JSON")

    out = entry.supervise(tmp_path / "_spec.json", {
        "work_dir": str(tmp_path), "run_token": "test", "timeout_s": 1,
    })

    assert out["status"] == "failed"
    assert "process cleanup" in out["error"]
    assert "Windows API failed" in out["traceback"]
    if missing_result:
        assert "worker exited with code -9" in out["error"]
        assert "containment query failed" in out["traceback"]
    else:
        assert "JSONDecodeError" in out["traceback"]
        assert "without _worker_result.json" not in out["error"]
    assert json.loads((tmp_path / "_stopped.json").read_text())["stopped"] is False


@pytest.mark.parametrize("assignment_fails", [False, True])
def test_windows_worker_is_gated_until_job_assignment(
    tmp_path, monkeypatch, windows_kernel, assignment_fails,
):
    events = []
    worker = Mock(pid=123)
    worker.poll.return_value = 0
    worker.stdin.closed = False
    worker.stdin.write.side_effect = lambda data: events.append("released")
    monkeypatch.setattr(entry.sys, "platform", "win32")
    monkeypatch.setattr(entry.subprocess, "Popen", Mock(return_value=worker))

    def assign(*args):
        events.append("assigned")
        return 0 if assignment_fails else 1

    windows_kernel.AssignProcessToJobObject.side_effect = assign
    (tmp_path / "_worker_result.json").write_text(json.dumps({"ok": True, "status": "completed"}))
    out = entry.supervise(tmp_path / "_spec.json", {
        "work_dir": str(tmp_path), "run_token": "test", "timeout_s": 1,
    })
    assert events == (["assigned"] if assignment_fails else ["assigned", "released"])
    assert out["ok"] is not assignment_fails
    if assignment_fails:
        windows_kernel.TerminateJobObject.assert_called_once()
    else:
        windows_kernel.TerminateJobObject.assert_not_called()
    assert json.loads((tmp_path / "_stopped.json").read_text())["stopped"] is True
