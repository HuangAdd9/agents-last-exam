from __future__ import annotations

from types import SimpleNamespace

import pytest

from ale_run.agents.codex.config import CodexConfig
from ale_run.agents.codex.deployer import CodexDeployer


@pytest.mark.asyncio
async def test_chatgpt_auth_uses_isolated_codex_home_and_is_removed(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    auth_content = '{"auth_mode":"chatgpt","tokens":{"access_token":"secret"}}'
    sandbox = SimpleNamespace(
        is_linux=True,
        node="/usr/bin/node",
        mcp_server_dir="/opt/cua-mcp-server",
    )
    executor = SimpleNamespace(
        config=CodexConfig(
            provider="chatgpt",
            model="gpt-5.6-sol",
            proxy_url="http://192.168.0.14:17890",
        ),
        work_dir=str(tmp_path / ".ale" / "codex" / "run-id"),
        sandbox=sandbox,
        env={"CODEX_AUTH_JSON": auth_content, "OPENAI_API_KEY": "must-not-pass"},
        cua_bridge_url=lambda: "http://127.0.0.1:5000",
    )
    deployer = CodexDeployer(executor)

    await deployer._write_codex_config(executor.config)
    deployer._write_chatgpt_auth()
    child_env = deployer._build_env(executor.config)

    codex_home = tmp_path / ".ale" / "codex-account" / "run-id"
    auth_path = codex_home / "auth.json"
    assert auth_path.read_text(encoding="utf-8") == auth_content
    assert (codex_home / "config.toml").is_file()
    assert child_env["CODEX_HOME"] == str(codex_home)
    assert "CODEX_AUTH_JSON" not in child_env
    assert "OPENAI_API_KEY" not in child_env
    assert "CODEX_API_KEY" not in child_env
    assert child_env["HTTPS_PROXY"] == "http://192.168.0.14:17890"
    assert child_env["ALL_PROXY"] == "http://192.168.0.14:17890"
    assert "127.0.0.1" in child_env["NO_PROXY"].split(",")

    deployer._remove_chatgpt_auth()

    assert not auth_path.exists()


def test_chatgpt_auth_rejects_invalid_payload(tmp_path) -> None:
    executor = SimpleNamespace(
        config=CodexConfig(provider="chatgpt"),
        env={"CODEX_AUTH_JSON": "not-json"},
    )
    deployer = CodexDeployer(executor)
    deployer._codex_home = str(tmp_path / "codex-home")

    with pytest.raises(RuntimeError, match="invalid JSON"):
        deployer._write_chatgpt_auth()

    assert not (tmp_path / "codex-home" / "auth.json").exists()
