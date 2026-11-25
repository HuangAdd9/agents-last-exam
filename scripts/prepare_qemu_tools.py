#!/usr/bin/env python3
"""Build and verify an offline CLI/CUA bundle without changing the guest image."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from ale_run.environments.prepared_tools import verify_tools


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", default="https://registry.npmmirror.com")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    tree = args.output / "tree"
    bridge_source = Path(__file__).resolve().parents[1] / "ale_run/agents/_assets/cua_mcp_server"
    bridge = tree / "cua_mcp_server"
    shutil.copytree(bridge_source, bridge, ignore=shutil.ignore_patterns("node_modules"))
    shutil.copytree(args.claude_dir, tree / "claude", symlinks=True)
    version = json.loads(
        (tree / "claude/node_modules/@anthropic-ai/claude-code/package.json").read_text()
    )["version"]
    subprocess.run(
        ["npm", "ci", "--omit=dev", "--no-audit", "--no-fund", "--registry", args.registry],
        cwd=bridge, check=True, timeout=600,
    )
    manifest = {
        "claude_version": version,
        "bridge_lock_sha256": hashlib.sha256((bridge / "package-lock.json").read_bytes()).hexdigest(),
        "platform": "linux-x64",
    }
    (tree / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    receipt = verify_tools(tree)
    archive = args.output / "prepared-tools.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        for entry in sorted(tree.iterdir()):
            bundle.add(entry, arcname=entry.name)
    with archive.open("rb") as stream:
        receipt["archive_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    receipt["archive"] = str(archive.resolve())
    (args.output / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
