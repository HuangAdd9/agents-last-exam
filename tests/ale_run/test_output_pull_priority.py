import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ale_run.base_interface import TaskDataSpec
from ale_run.environments import output_pull


def fixture(entries):
    sandbox = SimpleNamespace(is_linux=True, task_data_root='/tasks', metadata={},
                              list_dir=AsyncMock(return_value=entries))
    task = TaskDataSpec(domain_name='test', task_name='one', variant_name='base')
    return sandbox, task


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for key in ('ALE_OUTPUT_PULL_TIMEOUT_S', 'ALE_OUTPUT_PULL_EXCLUDE_DEPENDENCIES',
                'ALE_OUTPUT_PULL_EXTERNAL'):
        monkeypatch.delenv(key, raising=False)


@pytest.mark.asyncio
@pytest.mark.parametrize('exclude', [False, True])
async def test_dependency_filter_is_opt_in_and_preserves_source(tmp_path, monkeypatch, exclude):
    names = ['answer.json', '.flash_runtime/packages/numpy/__init__.py',
             '.venv/lib/a.py', 'packages/my_algorithm.py', 'venv']
    sandbox, task = fixture([{'relpath': n, 'is_dir': False} for n in names])
    if exclude:
        monkeypatch.setenv('ALE_OUTPUT_PULL_EXCLUDE_DEPENDENCIES', '1')

    async def download(remote, local, timeout):
        Path(local).write_text('content')
        return True

    sandbox.download_to_local = AsyncMock(side_effect=download)
    report = await output_pull.pull_to_host(sandbox, task, dest_dir=tmp_path)
    assert report['files'] == (3 if exclude else 5)
    assert report['excluded_dependency_entries'] == (2 if exclude else 0)
    assert not report['errors']
    assert (tmp_path / 'packages/my_algorithm.py').exists()
    assert (tmp_path / 'venv').is_file()
    assert (tmp_path / '.flash_runtime').exists() is not exclude


@pytest.mark.asyncio
@pytest.mark.parametrize('raises', [False, True])
async def test_one_failed_file_does_not_abort_other_outputs(tmp_path, raises):
    sandbox, task = fixture([{'relpath': n, 'is_dir': False} for n in ['bad.txt', 'answer.json']])

    async def download(remote, local, timeout):
        if remote.endswith('bad.txt'):
            if raises:
                raise OSError('network')
            return False
        Path(local).write_text('answer')
        return True

    sandbox.download_to_local = AsyncMock(side_effect=download)
    report = await output_pull.pull_to_host(sandbox, task, dest_dir=tmp_path)
    assert report['files'] == 1
    assert report['bytes'] == 6
    assert len(report['errors']) == 1
    assert (tmp_path / 'bad.txt.unreadable').exists()


@pytest.mark.asyncio
@pytest.mark.parametrize('provider', ['remote', 'docker', 'qemu'])
async def test_collection_budget_includes_listing_after_failed_bulk_copy(
    tmp_path, monkeypatch, provider,
):
    sandbox, task = fixture([])
    if provider == 'docker':
        sandbox.metadata = {'provider': 'docker', 'container_name': 'container'}
        monkeypatch.setattr(output_pull, '_pull_via_docker_cp',
                            AsyncMock(side_effect=RuntimeError('copy failed')))
    if provider == 'qemu':
        sandbox.metadata = {'provider': 'qemu', 'exchange_host_dir': str(tmp_path),
                            'exchange_guest_share': 'share', 'exchange_guest_transport': '9p'}
        monkeypatch.setattr(output_pull, '_pull_via_qemu_share',
                            AsyncMock(side_effect=RuntimeError('copy failed')))

    async def stuck(path):
        await asyncio.Event().wait()

    sandbox.list_dir = stuck
    monkeypatch.setenv('ALE_OUTPUT_PULL_TIMEOUT_S', '.02')
    report = await asyncio.wait_for(output_pull.pull_to_host(sandbox, task, dest_dir=tmp_path), 1)
    assert report['errors']
    assert report['transport'] == 'partial'
    assert report['files'] == report['bytes'] == 0


@pytest.mark.asyncio
async def test_docker_bulk_copy_precedes_listing_and_uses_own_budget(tmp_path, monkeypatch):
    sandbox, task = fixture([])
    sandbox.metadata = {'provider': 'docker', 'container_name': 'container'}
    sandbox.list_dir.side_effect = AssertionError('CUA must not be queried')
    expected = {'transport': 'docker-cp', 'files': 1, 'bytes': 10}

    async def copy(*args):
        await asyncio.sleep(.03)
        return expected

    monkeypatch.setattr(output_pull, '_pull_via_docker_cp', copy)
    monkeypatch.setenv('ALE_OUTPUT_PULL_TIMEOUT_S', '1')
    assert await output_pull.pull_to_host(sandbox, task, dest_dir=tmp_path) == expected
    sandbox.list_dir.assert_not_awaited()


@pytest.mark.asyncio
async def test_timeout_keeps_completed_files_and_drains_cancelled_downloads(tmp_path, monkeypatch):
    sandbox, task = fixture([{'relpath': name} for name in ['ready.txt', 'blocked.txt']])
    cancelled = asyncio.Event()
    observed = []

    async def download(remote, local, timeout):
        observed.append(timeout)
        if remote.endswith('ready.txt'):
            Path(local).write_text('ready')
            return True
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    sandbox.download_to_local = download
    monkeypatch.setenv('ALE_OUTPUT_PULL_TIMEOUT_S', '.03')
    report = await output_pull.pull_to_host(sandbox, task, dest_dir=tmp_path)
    assert cancelled.is_set()
    assert report['transport'] == 'partial'
    assert report['files'] == 1 and report['bytes'] == 5
    assert max(observed) <= .03
    assert not (tmp_path / 'blocked.txt').exists()


@pytest.mark.asyncio
async def test_external_cancellation_propagates_and_drains_downloads(tmp_path):
    sandbox, task = fixture([{'relpath': 'blocked.txt'}])
    entered, stopped = asyncio.Event(), asyncio.Event()

    async def download(*args, **kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    sandbox.download_to_local = download
    collector = asyncio.create_task(output_pull.pull_to_host(sandbox, task, dest_dir=tmp_path))
    await entered.wait()
    collector.cancel()
    with pytest.raises(asyncio.CancelledError):
        await collector
    assert stopped.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize('value', ['0', '-1', 'nan', 'inf', 'bad'])
async def test_invalid_budget_rejected(tmp_path, monkeypatch, value):
    sandbox, task = fixture([])
    monkeypatch.setenv('ALE_OUTPUT_PULL_TIMEOUT_S', value)
    with pytest.raises(ValueError):
        await output_pull.pull_to_host(sandbox, task, dest_dir=tmp_path)
    sandbox.list_dir.assert_not_awaited()


@pytest.mark.asyncio
async def test_external_staging_is_explicit_and_does_not_touch_sandbox(tmp_path, monkeypatch):
    sandbox, task = fixture([])
    monkeypatch.setenv('ALE_OUTPUT_PULL_EXTERNAL', 'developer-stage')
    report = await output_pull.pull_to_host(sandbox, task, dest_dir=tmp_path)
    assert report['skipped'] and report['transport'] == 'developer-stage'
    sandbox.list_dir.assert_not_awaited()


@pytest.mark.asyncio
async def test_docker_copy_timeout_kills_and_reaps_process(tmp_path, monkeypatch):
    started = asyncio.Event()
    killed = []

    async def communicate():
        if not killed:
            started.set()
            await asyncio.Event().wait()
        return b'', b''

    proc = SimpleNamespace(returncode=None, communicate=AsyncMock(side_effect=communicate),
                           kill=lambda: killed.append(True))
    monkeypatch.setattr(output_pull.asyncio, 'create_subprocess_exec', AsyncMock(return_value=proc))
    monkeypatch.setattr(output_pull, '_DOCKER_COPY_TIMEOUT_S', .02)
    with pytest.raises(TimeoutError):
        await output_pull._pull_via_docker_cp('container', '/output', tmp_path)
    assert started.is_set() and killed
    assert proc.communicate.await_count == 2
