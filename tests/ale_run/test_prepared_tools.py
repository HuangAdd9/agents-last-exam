from __future__ import annotations

import hashlib
import io
import json
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ale_run.agents import _bootstrap
from ale_run.agents.claude_code.config import ClaudeCodeConfig
from ale_run.agents.claude_code.deployer import ClaudeCodeDeployer
from ale_run.environments.prepared_tools import restore_tools, stage_tools, verify_tools
from ale_run.environments.providers.qemu import _build_snapshot_config


def make_tools(tmp_path):
    bridge = tmp_path / "cua_mcp_server"
    bridge.mkdir()
    (bridge / "package-lock.json").write_text('{}')
    manifest = {
        "claude_version": "2.1.241",
        "bridge_lock_sha256": hashlib.sha256(b'{}').hexdigest(),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    cli = tmp_path / "claude/node_modules/.bin/claude"
    cli.parent.mkdir(parents=True)
    cli.write_text("#!/bin/sh\nprintf '2.1.241 (Claude Code)\\n'\n")
    cli.chmod(0o755)
    node = tmp_path / "fake-node"
    node.write_text('#!/bin/sh\nprintf \'["key","screenshot"]\\n\'\n')
    node.chmod(0o755)
    return bridge, node


def test_verifier_rejects_changed_lock_and_wrong_cli(tmp_path):
    bridge, node = make_tools(tmp_path)
    assert verify_tools(tmp_path, str(node))["tools"] == ["key", "screenshot"]
    (bridge / "package-lock.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="lockfile hash"):
        verify_tools(tmp_path, str(node))
    (bridge / "package-lock.json").write_text('{}')
    cli = tmp_path / "claude/node_modules/.bin/claude"
    cli.write_text("#!/bin/sh\necho 2.1.170\n")
    with pytest.raises(ValueError, match="version mismatch"):
        verify_tools(tmp_path, str(node))


def test_restore_checks_hash_before_extraction_and_blocks_escape(tmp_path):
    archive = tmp_path / "bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        member = tarfile.TarInfo("../escaped")
        member.size = 1
        bundle.addfile(member, io.BytesIO(b'x'))
    destination = tmp_path / "tools"
    with pytest.raises(ValueError, match="archive hash"):
        restore_tools(archive, '0' * 64, destination, "node")
    assert not destination.exists()
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with pytest.raises(tarfile.FilterError):
        restore_tools(archive, digest, destination, "node")
    assert not (tmp_path / "escaped").exists()


@pytest.mark.parametrize("changed", [False, True])
async def test_offline_bridge_reuses_exact_manifest_without_npm(tmp_path, monkeypatch, changed):
    source = tmp_path / "source"
    target = tmp_path / "target"
    for directory in (source, target):
        (directory / "src").mkdir(parents=True)
        (directory / "src/index.js").write_text("// bridge")
        (directory / "package.json").write_text('{"dependencies":{"sdk":"1"}}')
        (directory / "package-lock.json").write_text('{}')
    (target / "node_modules/@modelcontextprotocol/sdk").mkdir(parents=True)
    if changed:
        (target / "package-lock.json").write_text('{"changed":true}')
    npm = AsyncMock(side_effect=AssertionError("Offline mode attempted npm"))
    monkeypatch.setattr(_bootstrap, "ensure_npm", npm)
    if changed:
        with pytest.raises(RuntimeError, match="manifests changed"):
            await _bootstrap._ensure_bridge_at(source, str(target), what="cua", offline=True)
    else:
        assert await _bootstrap._ensure_bridge_at(source, str(target), what="cua", offline=True) == str(target)
    npm.assert_not_awaited()


async def test_stage_requires_guest_verification_and_sets_explicit_cli(tmp_path):
    archive = tmp_path / "tools.tar.gz"
    archive.write_bytes(b'tools')
    exchange = tmp_path / "exchange"
    exchange.mkdir()
    sandbox = SimpleNamespace(
        is_linux=True, work_dir_base="/home/user/.ale", python="/opt/ale/python",
        node="/usr/local/bin/node", mcp_server_dir="/old/bridge",
        metadata={"exchange_host_dir": str(exchange), "exchange_guest_share": "//172.30.0.1/Data"},
        run_command=AsyncMock(return_value=SimpleNamespace(
            returncode=0, stdout='{"claude_version":"2.1.241"}', stderr="",
        )),
    )
    await stage_tools(sandbox, archive, 'a' * 64)
    assert (exchange / "prepared-tools.tar.gz").read_bytes() == b'tools'
    assert sandbox.metadata["prepared_tools"]["claude_path"].endswith("/claude/node_modules/.bin/claude")
    assert sandbox.mcp_server_dir.endswith("/cua_mcp_server")
    sandbox.run_command.return_value.returncode = 1
    with pytest.raises(RuntimeError, match="verification failed"):
        await stage_tools(sandbox, archive, 'a' * 64)


async def test_deployer_refuses_wrong_prepared_version_without_online_install():
    deployer = object.__new__(ClaudeCodeDeployer)
    deployer.config = ClaudeCodeConfig(cli_version="@anthropic-ai/claude-code@2.1.241")
    deployer.executor = SimpleNamespace(sandbox=SimpleNamespace(
        is_linux=True, metadata={"prepared_tools": {"claude_version": "2.1.170"}},
    ))
    deployer._auto_install_cli = AsyncMock()
    with pytest.raises(RuntimeError, match="pinned version"):
        await deployer.install()
    deployer._auto_install_cli.assert_not_awaited()


@pytest.mark.parametrize("extra", [{"tools_archive": "/tmp/a"}, {"tools_sha256": "a" * 64},
                                   {"tools_archive": "/tmp/a", "tools_sha256": "bad"}])
def test_config_requires_archive_and_digest_together(extra):
    with pytest.raises(ValueError, match="tools_sha256"):
        _build_snapshot_config({"image": "ale-ubuntu22-v1-1", "disk_source": "/tmp/disk", **extra})
