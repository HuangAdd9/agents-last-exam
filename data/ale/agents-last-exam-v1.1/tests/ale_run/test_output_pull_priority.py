import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ale_run.environments import output_pull


def fixture(entries):
    sandbox = SimpleNamespace(is_linux=True, task_data_root='/tasks', metadata={},
                              list_dir=AsyncMock(return_value=entries))
    task = SimpleNamespace(domain_name='test', task_name='one', variant_name='base')
    return sandbox, task


@pytest.mark.asyncio
async def test_dependency_filter_preserves_real_package_source(tmp_path):
    names = ['answer.json', '.flash_runtime/packages/numpy/__init__.py',
             '.venv/lib/a.py', 'packages/my_algorithm.py']
    sandbox, task = fixture([{'relpath': n, 'is_dir': False} for n in names])
    async def download(remote, local, timeout):
        Path(local).write_text('content')
        return True
    sandbox.download_to_local = AsyncMock(side_effect=download)
    report = await output_pull.pull_to_host(sandbox, task, dest_dir=tmp_path)
    assert report['files'] == 2
    assert report['excluded_dependency_entries'] == 2
    assert not report['errors']
    assert (tmp_path / 'packages/my_algorithm.py').exists()
    assert not (tmp_path / '.flash_runtime').exists()


@pytest.mark.asyncio
async def test_one_failed_file_does_not_abort_other_outputs(tmp_path):
    sandbox, task = fixture([{'relpath': n, 'is_dir': False} for n in ['bad.txt', 'answer.json']])
    async def download(remote, local, timeout):
        if remote.endswith('bad.txt'):
            raise OSError('network')
        Path(local).write_text('answer')
        return True
    sandbox.download_to_local = AsyncMock(side_effect=download)
    report = await output_pull.pull_to_host(sandbox, task, dest_dir=tmp_path)
    assert report['files'] == 1
    assert len(report['errors']) == 1


@pytest.mark.asyncio
async def test_collection_has_a_total_budget(tmp_path, monkeypatch):
    sandbox, task = fixture([])
    async def stuck(path):
        await asyncio.sleep(10)
    sandbox.list_dir = stuck
    monkeypatch.setenv('ALE_OUTPUT_PULL_TIMEOUT_S', '.02')
    report = await output_pull.pull_to_host(sandbox, task, dest_dir=tmp_path)
    assert report['errors']
    assert report['transport'] == 'partial'
