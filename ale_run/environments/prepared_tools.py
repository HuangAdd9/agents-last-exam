"""Verify and stage a pinned Linux Claude Code and CUA bridge bundle."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path


def verify_tools(directory: Path, node: str = "node") -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    bridge = directory / "cua_mcp_server"
    digest = hashlib.sha256((bridge / "package-lock.json").read_bytes()).hexdigest()
    if digest != manifest["bridge_lock_sha256"]:
        raise ValueError("Prepared bridge lockfile hash mismatch")
    cli = directory / "claude/node_modules/.bin/claude"
    version = subprocess.run(
        [str(cli), "--version"], capture_output=True, text=True, check=True, timeout=30,
    ).stdout.strip()
    if re.search(r"\b" + re.escape(manifest["claude_version"]) + r"\b", version) is None:
        raise ValueError(f"Prepared CLI version mismatch: {version}")
    # Initialize the actual stdio server and list tools without touching the desktop.
    probe = """
import {Client} from '@modelcontextprotocol/sdk/client/index.js';
import {StdioClientTransport} from '@modelcontextprotocol/sdk/client/stdio.js';
const client = new Client({name:'ale-preflight',version:'1.0.0'});
await client.connect(new StdioClientTransport({command:process.execPath,args:['src/index.js']}));
try {
  const result = await client.listTools();
  console.log(JSON.stringify(result.tools.map(t=>t.name)));
} finally { await client.close(); }
"""
    result = subprocess.run(
        [node, "--input-type=module", "-e", probe], cwd=bridge,
        capture_output=True, text=True, check=True, timeout=30,
    )
    tools = json.loads(result.stdout)
    if not {"key", "screenshot"}.issubset(tools):
        raise ValueError(f"Prepared CUA tool handshake incomplete: {tools}")
    return {**manifest, "cli_version_output": version, "tools": tools}


def restore_tools(archive: Path, sha256: str, destination: Path, node: str) -> dict:
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != sha256:
            raise ValueError("Prepared tools archive hash mismatch")
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive) as bundle:
        bundle.extractall(destination, filter="data")
    return verify_tools(destination, node)


async def stage_tools(sandbox, archive: Path, sha256: str) -> None:
    """Copy over the existing guest share; fail provisioning on failed verification."""
    if not sandbox.is_linux:
        raise ValueError("Prepared tools currently require a Linux guest")
    exchange = Path(sandbox.metadata["exchange_host_dir"])
    await asyncio.to_thread(shutil.copyfile, archive, exchange / "prepared-tools.tar.gz")
    mount = "/tmp/ale-prepared-tools"
    home = sandbox.work_dir_base.rstrip("/").rsplit("/", 1)[0]
    destination = f"{home}/.ale-prepared-tools/{sha256}"
    # This stdlib-only module is also the guest verifier, before ALE sources ship.
    code = Path(__file__).read_text() + (
        "\nprint(json.dumps(restore_tools(Path(" + repr(mount + "/prepared-tools.tar.gz")
        + "), " + repr(sha256) + ", Path(" + repr(destination) + "), "
        + repr(sandbox.node) + ")))\n"
    )
    command = (
        f"set -eu; mkdir -p {mount}; "
        f"sudo -n mount -t cifs {shlex.quote(sandbox.metadata['exchange_guest_share'])} "
        f"{mount} -o guest,vers=3.0,uid=$(id -u),gid=$(id -g); "
        f"{shlex.quote(sandbox.python)} -c {shlex.quote(code)}"
    )
    result = await sandbox.run_command(command, timeout=180)
    if result.returncode != 0:
        raise RuntimeError(f"Prepared tools verification failed: {result.stderr or result.stdout}")
    receipt = json.loads(result.stdout.strip().splitlines()[-1])
    sandbox.mcp_server_dir = destination + "/cua_mcp_server"
    sandbox.metadata["prepared_tools"] = {
        **receipt, "archive_sha256": sha256,
        "claude_path": destination + "/claude/node_modules/.bin/claude",
    }
