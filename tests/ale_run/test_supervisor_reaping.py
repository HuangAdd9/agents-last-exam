from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper regression")

_TOOL = """
import json, os, pathlib, sys, time
work = pathlib.Path(sys.argv[1])
children = []
for _ in range(24):
    pid = os.fork()
    if pid == 0:
        os.setsid()
        time.sleep(0.2)
        os._exit(0)
    children.append(pid)
pid = os.fork()
if pid == 0:
    os.setsid()
    while not (work / 'background.release').exists():
        with (work / 'heartbeat').open('a') as stream:
            stream.write('x')
        time.sleep(0.02)
(work / 'descendants.json').write_text(json.dumps({
    'short_lived': children, 'background': pid,
}))
os._exit(17)
"""

_WORKER = """
import json, os, pathlib, subprocess, sys, time
work = pathlib.Path(sys.argv[1])
assert sys.stdin.buffer.read(1) == b'1'
(work / 'worker.pid').write_text(str(os.getpid()))
tool = subprocess.run([sys.executable, str(work / 'tool.py'), str(work)])
(work / 'tool.exit').write_text(str(tool.returncode))
(work / 'ready').touch()
while not (work / 'release').exists():
    time.sleep(0.02)
if sys.argv[2] == 'worker_failure':
    os._exit(23)
(work / '_worker_result.json').write_text(json.dumps({
    'ok': True, 'status': 'completed',
}))
"""

_SUPERVISOR = """
import json, pathlib, sys
from subprocess import Popen
from ale_run.executors import _sandbox_entry as entry
work = pathlib.Path(sys.argv[1])
mode = sys.argv[2]
entry.subprocess.Popen = lambda args, **kwargs: Popen(
    [sys.executable, str(work / 'worker.py'), str(work), mode], **kwargs,
)
out = entry.supervise(work / '_spec.json', {
    'work_dir': str(work), 'run_token': 'reaping-test',
    'timeout_s': 7 if mode == 'timeout' else 20,
})
(work / 'result.json').write_text(json.dumps(out))
"""


def test_reaping_preserves_already_exited_worker_status():
    code = """
import pathlib, subprocess, sys, time
from ale_run.executors import _sandbox_entry as entry
tree = entry._LinuxProcessTree()
worker = subprocess.Popen([sys.executable, '-c', 'raise SystemExit(23)'])
try:
    deadline = time.monotonic() + 5
    while pathlib.Path(f'/proc/{worker.pid}/stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z':
        assert time.monotonic() < deadline, 'worker did not exit'
        time.sleep(0.01)
    tree.reap_exited_children(exclude_pid=worker.pid)
    assert worker.wait(timeout=5) == 23, 'reaper consumed the worker exit status'
    assert tree.is_empty()
finally:
    if worker.poll() is None:
        worker.kill()
    worker.wait(timeout=5)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", ["completed", "worker_failure", "timeout", "stop"])
def test_running_worker_reaps_orphans_and_preserves_background_work(tmp_path, mode):
    (tmp_path / "tool.py").write_text(_TOOL)
    (tmp_path / "worker.py").write_text(_WORKER)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(__file__).resolve().parents[2]),
            env.get("PYTHONPATH", ""),
        ]
    )
    with (tmp_path / "supervisor.log").open("w+") as log:
        supervisor = subprocess.Popen(
            [sys.executable, "-c", _SUPERVISOR, str(tmp_path), mode],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            env=env,
        )
        try:
            deadline = time.monotonic() + 5
            while not (tmp_path / "ready").exists():
                assert supervisor.poll() is None, (tmp_path / "supervisor.log").read_text()
                assert time.monotonic() < deadline, "worker did not start"
                time.sleep(0.02)
            descendants = json.loads((tmp_path / "descendants.json").read_text())
            short_lived = [Path(f"/proc/{pid}") for pid in descendants["short_lived"]]
            worker_pid = int((tmp_path / "worker.pid").read_text())
            deadline = time.monotonic() + 3
            while any(path.exists() for path in short_lived) and time.monotonic() < deadline:
                time.sleep(0.02)

            assert not any(path.exists() for path in short_lived), (
                "exited adopted children were not reaped while worker was running"
            )
            assert Path(f"/proc/{worker_pid}").exists()
            assert supervisor.poll() is None
            assert not (tmp_path / "_worker_result.json").exists()
            assert not (tmp_path / "_stopped.json").exists()
            assert (tmp_path / "tool.exit").read_text() == "17"
            heartbeat = (tmp_path / "heartbeat").read_bytes()
            time.sleep(0.1)
            assert (tmp_path / "heartbeat").read_bytes() != heartbeat

            if mode in ("completed", "worker_failure"):
                (tmp_path / "background.release").touch()
                (tmp_path / "release").touch()
            elif mode == "stop":
                (tmp_path / "_stop.request").write_text("reaping-test")
            assert supervisor.wait(timeout=10) == 0, (tmp_path / "supervisor.log").read_text()
            out = json.loads((tmp_path / "result.json").read_text())
            assert (
                out["status"]
                == {
                    "completed": "completed",
                    "worker_failure": "failed",
                    "timeout": "timeout",
                    "stop": "failed",
                }[mode]
            )
            if mode == "worker_failure":
                assert out["exit_code"] == 23
                assert "worker exited with code 23" in out["error"]
            elif mode == "stop":
                assert "solver stop requested" in out["error"]
            elif mode == "timeout":
                assert "solver budget exceeded" in out["error"]
            assert json.loads((tmp_path / "_stopped.json").read_text()) == {
                "run_token": "reaping-test",
                "stopped": True,
                "error": out.get("error"),
            }
            for pid in [worker_pid, descendants["background"], *descendants["short_lived"]]:
                assert not Path(f"/proc/{pid}").exists(), f"process {pid} survived cleanup"
        finally:
            (tmp_path / "_stop.request").write_text("reaping-test")
            try:
                supervisor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                supervisor.kill()
                supervisor.wait(timeout=5)
                raise
