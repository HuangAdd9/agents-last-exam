from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ale_run.environments import output_pull
from ale_run.tasks import driver


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['file_exists', 'directory_exists'])
async def test_file_probe_failure_is_not_false(command):
    interface = SimpleNamespace(
        _send_command=AsyncMock(return_value={'success': False, 'error': 'permission denied'}),
        run_command=AsyncMock(),
    )
    driver.install_resilient_cua_commands(interface)
    with pytest.raises(RuntimeError, match='permission denied'):
        await interface._send_command(command, {'path': '/reference/judge.py'})


@pytest.mark.asyncio
@pytest.mark.parametrize('invalid', [None, {}, {'success': True}, {'success': True, 'exists': 'false'}])
async def test_malformed_probe_retries_then_returns_real_absence(monkeypatch, invalid):
    send = AsyncMock(side_effect=[invalid, {'success': True, 'exists': False}])
    interface = SimpleNamespace(_send_command=send, run_command=AsyncMock())
    monkeypatch.setattr(driver, '_sleep_before_retry', AsyncMock())
    driver.install_resilient_cua_commands(interface)
    assert await interface._send_command('file_exists', {}) == {'success': True, 'exists': False}
    assert send.await_count == 2


@pytest.mark.asyncio
async def test_docker_output_does_not_depend_on_cua_listing(monkeypatch, tmp_path):
    sandbox = SimpleNamespace(is_linux=True, task_data_root='/data',
                              metadata={'provider': 'docker', 'container_name': 'test-container'},
                              list_dir=AsyncMock(side_effect=RuntimeError('CUA is down')))
    task = SimpleNamespace(domain_name='computing_math', task_name='test', variant_name='base')
    copy = AsyncMock(return_value={'transport': 'docker-cp', 'files': 1, 'bytes': 42})
    monkeypatch.setattr(output_pull, '_pull_via_docker_cp', copy)
    result = await output_pull.pull_to_host(sandbox, task, dest_dir=tmp_path)
    assert result['files'] == 1
    sandbox.list_dir.assert_not_awaited()
    copy.assert_awaited_once()


@pytest.mark.asyncio
async def test_docker_copy_failure_still_falls_back_to_cua(monkeypatch, tmp_path):
    sandbox = SimpleNamespace(is_linux=True, task_data_root='/data',
                              metadata={'provider': 'docker', 'container_name': 'test-container'},
                              list_dir=AsyncMock(return_value=[]))
    task = SimpleNamespace(domain_name='computing_math', task_name='test', variant_name='base')
    monkeypatch.setattr(output_pull, '_pull_via_docker_cp', AsyncMock(side_effect=RuntimeError('copy failed')))
    result = await output_pull.pull_to_host(sandbox, task, dest_dir=tmp_path)
    assert result['skipped'] is True
    sandbox.list_dir.assert_awaited_once()
