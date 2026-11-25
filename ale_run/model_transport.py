"""Credential-free TCP checks for the locally managed Kimi gateway deployment."""

from __future__ import annotations

import json
import socket
import time
from pathlib import Path
from urllib.parse import urlsplit


class ModelTransportUnavailable(RuntimeError):
    """The managed model transport is unavailable before provisioning."""


def check_managed_transport(config: dict, base_dir: Path, *, timeout_s: float = 5) -> dict:
    """Check declared transport; TCP success does not establish inference health.

    Only the deployment adjacent to the selected model catalog is considered.
    No keys or prompts are read or sent. Other deployments are left unchecked.
    """
    catalog = config.get("model_catalog_path")
    gateway = config.get("base_url")
    skipped = {"status": "not_applicable", "checks": []}
    if not catalog or not gateway:
        return skipped
    deployment = (base_dir / catalog).resolve().parent / "deployment.json"
    if not deployment.is_file():
        return skipped
    record = json.loads(deployment.read_text())
    if (record.get("service") != "kimi-responses.service"
            or record.get("model") != config.get("model")):
        return skipped
    if urlsplit(gateway).port != urlsplit(record["base_url"]).port:
        return skipped
    report = {"status": "reachable", "probe": "tcp_connect", "checks": []}
    for role, url in (("gateway", gateway), ("upstream", record["upstream"])):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Managed endpoint must be an HTTP(S) URL")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        item = {"role": role, "host": parsed.hostname, "port": port}
        start = time.monotonic()
        try:
            with socket.create_connection((parsed.hostname, port), timeout=timeout_s):
                pass
            item["connected"] = True
        except OSError as error:
            item.update(connected=False, error_type=type(error).__name__, errno=error.errno)
            report["status"] = "unreachable"
        item["elapsed_seconds"] = time.monotonic() - start
        report["checks"].append(item)
    return report
