from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
from types import SimpleNamespace

import pytest

from ale_run.agents.claude_code import deployer as module
from ale_run.agents.claude_code.config import ClaudeCodeConfig


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX CLI signal handling")
@pytest.mark.parametrize("success", [False, True])
@pytest.mark.parametrize("ignore_term", [False, True])
async def test_final_result_reclaims_hanging_cli_without_losing_outcome(
    tmp_path, monkeypatch, success, ignore_term,
):
    event = {
        "type": "result", "subtype": "success" if success else "error_during_execution",
        "is_error": not success,
    }
    cli = tmp_path / "claude"
    cli.write_text(
        f"#!{sys.executable}\n"
        "import json, signal, time\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
        + f"print(json.dumps({event!r}), flush=True)\n"
        "time.sleep(30)\n"
    )
    cli.chmod(0o755)
    config = ClaudeCodeConfig(model="test", provider="direct", api_key="test", otel_enabled=False)
    agent = module.ClaudeCodeDeployer(SimpleNamespace(
        config=config, work_dir=str(tmp_path), env={},
    ))
    agent._claude_path = str(cli)
    monkeypatch.setattr(module, "_POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(module, "_RESULT_EXIT_GRACE_S", 0.1)
    monkeypatch.setattr(module, "_TERM_GRACE_S", 0.1)
    result = await asyncio.wait_for(agent.launch("test"), timeout=5)
    assert result.status == ("completed" if success else "failed")
    assert result.exit_code == (-signal.SIGKILL if ignore_term else -signal.SIGTERM)
    assert result.duration_s < 5
    assert json.loads((tmp_path / "transcript.jsonl").read_text()) == event
    with pytest.raises(ProcessLookupError):
        os.kill(result.pid, 0)


async def test_assistant_text_is_not_a_final_result(tmp_path, monkeypatch):
    cli = tmp_path / "claude"
    cli.write_text(
        f"#!{sys.executable}\n"
        "import json, time\n"
        "print(json.dumps({'type': 'assistant', 'message': {'content': 'Done.'}}), flush=True)\n"
        "time.sleep(30)\n"
    )
    cli.chmod(0o755)
    agent = module.ClaudeCodeDeployer(SimpleNamespace(
        config=ClaudeCodeConfig(model="test", provider="direct", api_key="test", otel_enabled=False),
        work_dir=str(tmp_path), env={},
    ))
    agent._claude_path = str(cli)
    monkeypatch.setattr(module, "_POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(module, "_RESULT_EXIT_GRACE_S", 0.01)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(agent.launch("test"), timeout=0.3)
    pid = int((tmp_path / "claude.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
