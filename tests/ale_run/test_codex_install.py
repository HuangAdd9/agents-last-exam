from __future__ import annotations

from types import SimpleNamespace

import pytest

from ale_run.agents import _bootstrap
from ale_run.agents.codex.config import CodexConfig
from ale_run.agents.codex.deployer import CodexDeployer


@pytest.mark.asyncio
async def test_empty_patch_url_installs_and_uses_pinned_npm_codex(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stock mode must not demand a fork after npm installation succeeds."""
    config = CodexConfig(
        codex_version="0.153.4",
        fork_version="",
        patched_binary_url="",
        patched_binary_url_windows="",
    )
    sandbox = SimpleNamespace(is_linux=True, mcp_server_dir="/opt/cua-mcp-server")
    executor = SimpleNamespace(
        config=config,
        sandbox=sandbox,
        work_dir=str(tmp_path / "codex-run"),
    )
    deployer = CodexDeployer(executor)
    installed = False

    async def fake_ensure_npm() -> str:
        return "/usr/bin/npm"

    async def fake_install(version: str) -> None:
        nonlocal installed
        assert version == "0.153.4"
        installed = True

    async def fake_ensure_cua(_sandbox) -> None:
        return None

    async def fake_write_config(_config) -> None:
        return None

    def fake_which(command: str) -> str | None:
        if command == "codex" and installed:
            return "/usr/local/bin/codex"
        return None

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="codex-cli 0.153.4\n", stderr="")

    monkeypatch.setattr(_bootstrap, "ensure_npm", fake_ensure_npm)
    monkeypatch.setattr(_bootstrap, "ensure_cua_mcp_server", fake_ensure_cua)
    monkeypatch.setattr("ale_run.agents.codex.deployer.shutil.which", fake_which)
    monkeypatch.setattr("ale_run.agents.codex.deployer.subprocess.run", fake_run)
    monkeypatch.setattr(deployer, "_npm_install_codex", fake_install)
    monkeypatch.setattr(deployer, "_write_codex_config", fake_write_config)

    await deployer.install()

    assert installed
    assert deployer._codex_path == "/usr/local/bin/codex"
    assert deployer.version == "0.153.4"
