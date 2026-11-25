import asyncio
import fcntl
import os

import pytest

from ale_run.environments.providers import docker


@pytest.fixture
def sleeping_docker(monkeypatch, tmp_path):
    executable = tmp_path / "docker"
    pidfile = tmp_path / "query.pid"
    executable.write_text('#!/bin/sh\nprintf "%s" "$$" > "$ALE_TEST_PIDFILE"\nexec sleep 60\n')
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("ALE_TEST_PIDFILE", str(pidfile))
    monkeypatch.setattr(docker, "_DOCKER_QUERY_TIMEOUT", 0.1)
    monkeypatch.setattr(docker, "_CPUSET_LOCK_PATH", tmp_path / "allocation.lock")
    return pidfile


async def test_stuck_scan_reaps_cli_and_releases_allocation_lock(sleeping_docker):
    with pytest.raises(RuntimeError, match="docker ps --quiet exceeded"):
        await asyncio.wait_for(
            docker._start_container("unused", ["run", "--name", "unused", "image"], 1),
            timeout=2,
        )
    with pytest.raises(ProcessLookupError):
        os.kill(int(sleeping_docker.read_text()), 0)
    with docker._CPUSET_LOCK_PATH.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_UN)


async def test_cancelled_query_reaps_cli(sleeping_docker, monkeypatch):
    monkeypatch.setattr(docker, "_DOCKER_QUERY_TIMEOUT", 60)
    task = asyncio.create_task(docker._run_docker("inspect", "unused"))
    try:
        for _ in range(100):
            if sleeping_docker.exists():
                break
            await asyncio.sleep(0.005)
        assert sleeping_docker.exists()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        with pytest.raises(ProcessLookupError):
            os.kill(int(sleeping_docker.read_text()), 0)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_completed_query_preserves_output(monkeypatch, tmp_path):
    executable = tmp_path / "docker"
    executable.write_text('#!/bin/sh\nprintf "docker-ready\\n"\n')
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    assert await docker._run_docker("ps", "--quiet") == (0, "docker-ready", "")
