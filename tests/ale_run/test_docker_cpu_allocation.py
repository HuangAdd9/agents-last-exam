import asyncio
import json
import sys
from unittest.mock import AsyncMock

import pytest

from ale_run.base_interface import SandboxHandle
from ale_run.environments import images
from ale_run.environments.providers import docker


@pytest.fixture
def engine(monkeypatch, tmp_path):
    containers = {}
    commands = []

    async def run(*args):
        commands.append(args)
        await asyncio.sleep(0)
        if args[0] == "ps":
            return 0, "\n".join(containers), ""
        if args[0] == "inspect":
            return 0, "\n".join(json.dumps(containers[name]) for name in args[3:]), ""
        if args[0] == "run":
            name = args[args.index("--name") + 1]
            containers[name] = args[args.index("--cpuset-cpus") + 1]
            return 0, name, ""
        if args[0] in {"rm", "stop"}:
            containers.pop(args[-1], None)
            return 0, args[-1], ""
        raise AssertionError(args)

    monkeypatch.setattr(docker, "_run_docker", run)
    monkeypatch.setattr(docker, "_CPUSET_LOCK_PATH", tmp_path / "cpuset.lock")
    monkeypatch.setattr(docker, "_CPUSET_POLL_INTERVAL", 0.001)
    monkeypatch.setattr(docker.os, "sched_getaffinity", lambda _: {1, 3, 5, 7})
    return containers, commands


@pytest.mark.parametrize(
    "quota,expected", [(2, "3,7"), (1.5, "3,7"), (0.25, "3")],
)
async def test_allocation_respects_affinity_existing_ranges_and_fractional_quota(
    engine, quota, expected,
):
    containers, commands = engine
    containers["existing"] = "1,5-6"

    result = await docker._start_container(
        "new", ["run", "-d", "--name", "new", "--cpus", str(quota), "image"], quota,
    )

    assert result == ("new", expected)
    assert containers["new"] == expected
    arguments = next(args for args in commands if args[0] == "run")
    assert arguments[arguments.index("--cpus") + 1] == str(quota)
    assert arguments[-1] == "image"


async def test_simultaneous_starts_are_disjoint_and_wait_for_release(engine):
    containers, commands = engine
    results = await asyncio.gather(*(
        docker._start_container(name, ["run", "-d", "--name", name, "image"], 2)
        for name in ("first", "second")
    ))
    first_cpus, second_cpus = (set(result[1].split(",")) for result in results)
    assert first_cpus.isdisjoint(second_cpus)
    assert first_cpus | second_cpus == {"1", "3", "5", "7"}

    waiter = asyncio.create_task(docker._start_container(
        "third", ["run", "-d", "--name", "third", "image"], 2,
    ))
    try:
        await asyncio.sleep(0.02)
        assert not waiter.done()
        assert len([args for args in commands if args[0] == "run"]) == 2
        await docker.DockerProvider({}).release(
            SandboxHandle(
                id="first", endpoint="http://localhost", os="linux",
                **images.get("ale-kasm").sandbox_paths(),
            ),
        )
        _, cpu_set = await asyncio.wait_for(waiter, 2)
        assert cpu_set == results[0][1]
        assert "first" not in containers
    finally:
        if not waiter.done():
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.parametrize("mode,released", [("keep", False), ("stop", True), ("delete", True)])
async def test_release_modes_preserve_or_free_reservation(engine, mode, released):
    containers, _ = engine
    containers["existing"] = "1,3"
    handle = SandboxHandle(
        id="existing", endpoint="http://localhost", os="linux",
        **images.get("ale-kasm").sandbox_paths(),
    )

    await docker.DockerProvider({}).release(handle, mode=mode)

    assert (await docker._occupied_cpus() == set()) is released


@pytest.mark.parametrize("quota", [0, -1, float("inf"), float("nan"), 4.5])
async def test_impossible_or_invalid_requests_fail_without_starting(engine, quota):
    _, commands = engine
    with pytest.raises(ValueError, match="positive finite|only 4 are allowed"):
        await docker._start_container("new", ["run", "--name", "new", "image"], quota)
    assert not commands


async def test_cancel_while_waiting_does_not_create_a_container(engine):
    containers, commands = engine
    containers["existing"] = "1,3,5,7"
    waiter = asyncio.create_task(docker._start_container(
        "new", ["run", "--name", "new", "image"], 1,
    ))
    await asyncio.sleep(0.01)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not any(args[0] == "run" for args in commands)
    assert set(containers) == {"existing"}


async def test_failed_run_removes_partial_container_before_reusing_cpus(engine, monkeypatch):
    containers, _ = engine
    original = docker._run_docker

    async def fail_run(*args):
        result = await original(*args)
        if args[0] == "run":
            return 125, "", "startup failed"
        return result

    monkeypatch.setattr(docker, "_run_docker", fail_run)
    with pytest.raises(RuntimeError, match="startup failed"):
        await docker._start_container("new", ["run", "--name", "new", "image"], 4)
    assert not containers
    monkeypatch.setattr(docker, "_run_docker", original)
    _, cpu_set = await docker._start_container(
        "retry", ["run", "--name", "retry", "image"], 4,
    )
    assert cpu_set == "1,3,5,7"


async def test_cancel_during_run_waits_for_creation_then_removes_container(engine, monkeypatch):
    containers, _ = engine
    original = docker._run_docker
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def slow_run(*args):
        if args[0] == "run":
            entered.set()
            await finish.wait()
        return await original(*args)

    monkeypatch.setattr(docker, "_run_docker", slow_run)
    start = asyncio.create_task(docker._start_container(
        "new", ["run", "--name", "new", "image"], 4,
    ))
    await asyncio.wait_for(entered.wait(), 2)
    start.cancel()
    await asyncio.sleep(0)
    assert not start.done()
    finish.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(start, 2)
    assert not containers
    _, cpu_set = await docker._start_container(
        "retry", ["run", "--name", "retry", "image"], 4,
    )
    assert cpu_set == "1,3,5,7"


@pytest.mark.parametrize("stage", ["list", "inspect", "disappeared"])
async def test_docker_scan_errors_do_not_allocate_over_existing_cpus(monkeypatch, stage):
    commands = AsyncMock(side_effect=(
        [(1, "", "daemon unavailable")] if stage == "list" else [
            (0, "existing", ""),
            (1, "", "No such object: existing" if stage == "disappeared" else "inspect failed"),
        ]
    ))
    monkeypatch.setattr(docker, "_run_docker", commands)
    if stage == "disappeared":
        assert await docker._occupied_cpus() is None
    else:
        with pytest.raises(RuntimeError, match="daemon unavailable|inspect failed"):
            await docker._occupied_cpus()


async def test_rescan_after_concurrent_container_removal(engine, monkeypatch):
    monkeypatch.setattr(docker, "_occupied_cpus", AsyncMock(side_effect=[None, set()]))
    _, cpu_set = await docker._start_container(
        "new", ["run", "--name", "new", "image"], 4,
    )
    assert cpu_set == "1,3,5,7"


async def test_other_runner_process_holds_allocation_lock(engine, monkeypatch):
    scan = AsyncMock(return_value=set())
    monkeypatch.setattr(docker, "_occupied_cpus", scan)
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c",
        "import fcntl, sys\n"
        "with open(sys.argv[1], 'a') as lock:\n"
        "    fcntl.flock(lock, fcntl.LOCK_EX)\n"
        "    print('locked', flush=True)\n"
        "    sys.stdin.read(1)\n",
        str(docker._CPUSET_LOCK_PATH),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
    )
    start = None
    try:
        assert await asyncio.wait_for(process.stdout.readline(), 2) == b"locked\n"
        start = asyncio.create_task(docker._start_container(
            "new", ["run", "--name", "new", "image"], 2,
        ))
        await asyncio.sleep(0.02)
        scan.assert_not_awaited()
        assert not start.done()
        process.stdin.write(b"x")
        await process.stdin.drain()
        await asyncio.wait_for(process.wait(), 2)
        assert await asyncio.wait_for(start, 2) == ("new", "1,3")
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        if start is not None and not start.done():
            start.cancel()
            await asyncio.gather(start, return_exceptions=True)
