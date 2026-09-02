"""Ephemeral full-OS sandboxes cloned from Proxmox VE templates."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import os
import re
import secrets
import shlex
import shutil
import signal
import tempfile
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

import requests
from urllib3.exceptions import InsecureRequestWarning

from ...base_interface import Provider, ReleaseMode, SandboxHandle, SandboxSpec

logger = logging.getLogger(__name__)

_DEFAULT_CUA_PORT = 5000
_DEFAULT_REQUEST_TIMEOUT_S = 30
_DEFAULT_TASK_TIMEOUT_S = 3600
_DEFAULT_READY_TIMEOUT_S = 900
_DEFAULT_POLL_INTERVAL_S = 3.0
_DEFAULT_NOVNC_PORT_MIN = 6080
_DEFAULT_NOVNC_PORT_MAX = 6180
_DEFAULT_NOVNC_GUEST_PORT = 5900
_DEFAULT_NOVNC_READY_TIMEOUT_S = 60


class ProxmoxApiError(RuntimeError):
    """One failed Proxmox REST request."""

    def __init__(self, status_code: int, method: str, path: str, detail: str):
        self.status_code = status_code
        self.method = method
        self.path = path
        self.detail = detail
        super().__init__(f"Proxmox API {method} {path} failed ({status_code}): {detail}")


class _ProxmoxApi:
    def __init__(
        self,
        *,
        api_url: str,
        token_id: str,
        token_secret: str,
        verify_ssl: bool | str,
        timeout_s: float,
    ) -> None:
        self._base_url = f"{api_url.rstrip('/')}/api2/json"
        self._verify_ssl = verify_ssl
        self._timeout_s = timeout_s
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"PVEAPIToken={token_id}={token_secret}"})

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        normalized_path = "/" + path.lstrip("/")
        try:
            # A trusted LAN deployment may explicitly opt into PVE's default
            # self-signed certificate with ``verify_ssl: false``.  urllib3
            # otherwise emits the same warning for every polling request,
            # obscuring useful lifecycle logs.  Keep the suppression scoped to
            # this request; verified requests and unrelated HTTP clients retain
            # their normal warning behavior.
            with warnings.catch_warnings():
                if self._verify_ssl is False:
                    warnings.simplefilter("ignore", InsecureRequestWarning)
                response = self._session.request(
                    method,
                    f"{self._base_url}{normalized_path}",
                    params=params,
                    data=data,
                    timeout=self._timeout_s,
                    verify=self._verify_ssl,
                )
        except requests.RequestException as exc:
            raise ProxmoxApiError(0, method, normalized_path, str(exc)) from exc

        try:
            payload = response.json()
        except ValueError:
            payload = None
        if not response.ok:
            detail: Any = payload
            if isinstance(payload, dict):
                detail = payload.get("errors") or payload.get("message") or payload
            if detail is None:
                detail = response.text[:500]
            raise ProxmoxApiError(
                response.status_code,
                method,
                normalized_path,
                str(detail)[:1000],
            )
        if not isinstance(payload, dict) or "data" not in payload:
            raise ProxmoxApiError(
                response.status_code,
                method,
                normalized_path,
                "response is missing the data field",
            )
        return payload["data"]


@dataclass(frozen=True, slots=True)
class ProxmoxSnapshotConfig:
    image: str
    node: str
    template_vmid: int
    full_clone: bool = False
    storage: str = ""
    pool: str = ""
    guest_ip: str = ""
    guest_cidr: str = ""
    vcpus: int = 0
    memory_gb: int = 0
    allow_gpu: bool = False
    name_prefix: str = "ale-pve"
    ready_timeout_s: int = _DEFAULT_READY_TIMEOUT_S
    task_timeout_s: int = _DEFAULT_TASK_TIMEOUT_S
    poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S
    novnc_enabled: bool = False
    novnc_bind_address: str = "127.0.0.1"
    novnc_public_host: str = ""
    novnc_port_min: int = _DEFAULT_NOVNC_PORT_MIN
    novnc_port_max: int = _DEFAULT_NOVNC_PORT_MAX
    novnc_guest_port: int = _DEFAULT_NOVNC_GUEST_PORT
    novnc_ready_timeout_s: int = _DEFAULT_NOVNC_READY_TIMEOUT_S
    novnc_websockify: str = "websockify"
    novnc_web_root: str = "/usr/share/novnc"
    novnc_runtime_root: str = "/tmp/ale-proxmox-novnc"
    novnc_tls_cert: str = ""
    novnc_tls_key: str = ""
    novnc_manage_guest_password: bool = False


@dataclass(frozen=True, slots=True)
class ProxmoxProviderConfig:
    api_url: str
    token_id: str
    token_secret: str
    verify_ssl: bool | str
    request_timeout_s: float
    snapshots: dict[str, ProxmoxSnapshotConfig]
    gcs_sa_key: str = ""


def _normalize_api_url(value: str) -> str:
    raw = value.strip().rstrip("/")
    for suffix in ("/api2/json", "/api2/extjs", "/api2/html"):
        raw = raw.removesuffix(suffix)
    parsed = urlsplit(raw)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("proxmox api_url must be an http(s) URL such as https://pve.example:8006")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _build_snapshot_config(raw: dict[str, Any]) -> ProxmoxSnapshotConfig:
    node = str(raw.get("node") or "").strip()
    if not node:
        raise KeyError("proxmox snapshot config missing required field `node`")
    template_vmid = int(raw.get("template_vmid") or 0)
    if template_vmid <= 0:
        raise ValueError("proxmox template_vmid must be a positive integer")

    vcpus = int(raw.get("vcpus") or 0)
    memory_gb = int(raw.get("memory_gb") or 0)
    ready_timeout_s = int(raw.get("ready_timeout_s") or _DEFAULT_READY_TIMEOUT_S)
    task_timeout_s = int(raw.get("task_timeout_s") or _DEFAULT_TASK_TIMEOUT_S)
    poll_interval_s = float(raw.get("poll_interval_s") or _DEFAULT_POLL_INTERVAL_S)
    if vcpus < 0 or memory_gb < 0:
        raise ValueError("proxmox vcpus and memory_gb must be non-negative")
    if ready_timeout_s <= 0 or task_timeout_s <= 0 or poll_interval_s <= 0:
        raise ValueError("proxmox timeout and poll interval values must be positive")

    novnc_port_min = int(raw.get("novnc_port_min") or _DEFAULT_NOVNC_PORT_MIN)
    novnc_port_max = int(raw.get("novnc_port_max") or _DEFAULT_NOVNC_PORT_MAX)
    novnc_guest_port = int(raw.get("novnc_guest_port") or _DEFAULT_NOVNC_GUEST_PORT)
    novnc_ready_timeout_s = int(raw.get("novnc_ready_timeout_s") or _DEFAULT_NOVNC_READY_TIMEOUT_S)
    if not (1 <= novnc_port_min <= novnc_port_max <= 65535):
        raise ValueError("proxmox noVNC port range must be between 1 and 65535")
    if not (1 <= novnc_guest_port <= 65535):
        raise ValueError("proxmox novnc_guest_port must be between 1 and 65535")
    if novnc_ready_timeout_s <= 0:
        raise ValueError("proxmox novnc_ready_timeout_s must be positive")

    novnc_tls_cert = str(raw.get("novnc_tls_cert") or "").strip()
    novnc_tls_key = str(raw.get("novnc_tls_key") or "").strip()
    if bool(novnc_tls_cert) != bool(novnc_tls_key):
        raise ValueError("proxmox novnc_tls_cert and novnc_tls_key must be set together")

    novnc_bind_address = str(raw.get("novnc_bind_address") or "127.0.0.1").strip()
    try:
        ipaddress.ip_address(novnc_bind_address)
    except ValueError as exc:
        raise ValueError("proxmox novnc_bind_address must be an IP address") from exc

    guest_ip = str(raw.get("guest_ip") or "").strip()
    if guest_ip:
        ipaddress.ip_address(guest_ip)
    guest_cidr = str(raw.get("guest_cidr") or "").strip()
    if guest_cidr:
        ipaddress.ip_network(guest_cidr, strict=False)

    name_prefix = str(raw.get("name_prefix") or "ale-pve").strip().lower()
    name_prefix = re.sub(r"[^a-z0-9-]+", "-", name_prefix).strip("-")
    if not name_prefix:
        raise ValueError("proxmox name_prefix must contain a letter or digit")

    return ProxmoxSnapshotConfig(
        image=str(raw["image"]),
        node=node,
        template_vmid=template_vmid,
        full_clone=bool(raw.get("full_clone", False)),
        storage=str(raw.get("storage") or "").strip(),
        pool=str(raw.get("pool") or "").strip(),
        guest_ip=guest_ip,
        guest_cidr=guest_cidr,
        vcpus=vcpus,
        memory_gb=memory_gb,
        allow_gpu=bool(raw.get("allow_gpu", False)),
        name_prefix=name_prefix,
        ready_timeout_s=ready_timeout_s,
        task_timeout_s=task_timeout_s,
        poll_interval_s=poll_interval_s,
        novnc_enabled=bool(raw.get("novnc_enabled", False)),
        novnc_bind_address=novnc_bind_address,
        novnc_public_host=str(raw.get("novnc_public_host") or "").strip(),
        novnc_port_min=novnc_port_min,
        novnc_port_max=novnc_port_max,
        novnc_guest_port=novnc_guest_port,
        novnc_ready_timeout_s=novnc_ready_timeout_s,
        novnc_websockify=str(raw.get("novnc_websockify") or "websockify").strip(),
        novnc_web_root=str(raw.get("novnc_web_root") or "/usr/share/novnc"),
        novnc_runtime_root=str(raw.get("novnc_runtime_root") or "/tmp/ale-proxmox-novnc"),
        novnc_tls_cert=str(Path(novnc_tls_cert).expanduser().resolve()) if novnc_tls_cert else "",
        novnc_tls_key=str(Path(novnc_tls_key).expanduser().resolve()) if novnc_tls_key else "",
        novnc_manage_guest_password=bool(raw.get("novnc_manage_guest_password", False)),
    )


def _build_provider_config(raw: dict[str, Any]) -> ProxmoxProviderConfig:
    api_url = _normalize_api_url(str(raw.get("api_url") or ""))
    token_id = str(raw.get("token_id") or "").strip()
    token_secret = str(raw.get("token_secret") or "").strip()
    if not token_id or "!" not in token_id:
        raise ValueError("proxmox token_id must use the user@realm!token-name form")
    if not token_secret:
        raise KeyError("proxmox provider config missing required field `token_secret`")

    verify_ssl_raw = raw.get("verify_ssl", True)
    if isinstance(verify_ssl_raw, bool):
        verify_ssl: bool | str = verify_ssl_raw
    elif isinstance(verify_ssl_raw, str) and verify_ssl_raw.strip():
        verify_ssl = str(Path(verify_ssl_raw).expanduser().resolve())
    else:
        raise TypeError("proxmox verify_ssl must be a boolean or CA bundle path")

    request_timeout_s = float(raw.get("request_timeout_s") or _DEFAULT_REQUEST_TIMEOUT_S)
    if request_timeout_s <= 0:
        raise ValueError("proxmox request_timeout_s must be positive")

    snapshots_raw = raw.get("snapshots")
    if not isinstance(snapshots_raw, dict) or not snapshots_raw:
        raise TypeError("proxmox provider config requires a non-empty `snapshots` mapping")
    snapshots = {
        str(tag): _build_snapshot_config(dict(snapshot_raw))
        for tag, snapshot_raw in snapshots_raw.items()
    }
    return ProxmoxProviderConfig(
        api_url=api_url,
        token_id=token_id,
        token_secret=token_secret,
        verify_ssl=verify_ssl,
        request_timeout_s=request_timeout_s,
        snapshots=snapshots,
        gcs_sa_key=str(Path(str(raw["gcs_sa_key"])).expanduser().resolve())
        if raw.get("gcs_sa_key")
        else "",
    )


def _vm_name(prefix: str, spec: SandboxSpec) -> str:
    task = re.sub(r"[^a-z0-9-]+", "-", spec.task_id.lower()).strip("-")
    task = task[-30:] or "task"
    return f"{prefix}-{task}-{uuid.uuid4().hex[:8]}"[:63].rstrip("-")


def _usable_agent_ips(payload: Any, guest_cidr: str) -> list[str]:
    if isinstance(payload, dict):
        interfaces = payload.get("result") or []
    else:
        interfaces = payload or []
    network = ipaddress.ip_network(guest_cidr, strict=False) if guest_cidr else None
    addresses: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for interface in interfaces if isinstance(interfaces, list) else []:
        if not isinstance(interface, dict):
            continue
        raw_addresses = interface.get("ip-addresses") or interface.get("ip_addresses") or []
        for raw_address in raw_addresses if isinstance(raw_addresses, list) else []:
            if not isinstance(raw_address, dict):
                continue
            value = raw_address.get("ip-address") or raw_address.get("ip_address")
            try:
                address = ipaddress.ip_address(str(value))
            except ValueError:
                continue
            if (
                address.is_loopback
                or address.is_link_local
                or address.is_multicast
                or address.is_unspecified
            ):
                continue
            if network and address not in network:
                continue
            addresses.append(address)
    addresses.sort(key=lambda address: (address.version != 4, str(address)))
    return [str(address) for address in addresses]


class ProxmoxProvider(Provider):
    """Clone one disposable Proxmox VM template per ALE run."""

    def __init__(self, config: ProxmoxProviderConfig | dict[str, Any]):
        if isinstance(config, dict):
            config = _build_provider_config(config)
        self._cfg = config
        self._api = _ProxmoxApi(
            api_url=config.api_url,
            token_id=config.token_id,
            token_secret=config.token_secret,
            verify_ssl=config.verify_ssl,
            timeout_s=config.request_timeout_s,
        )
        self._allocation_lock = asyncio.Lock()
        self._novnc_lock = asyncio.Lock()
        self._preflight_lock = asyncio.Lock()
        self._novnc_processes: dict[int, int] = {}
        self._verified_templates: set[ProxmoxSnapshotConfig] = set()

    @property
    def config(self) -> ProxmoxProviderConfig:
        return self._cfg

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        return await asyncio.to_thread(
            self._api.request,
            method,
            path,
            params=params,
            data=data,
        )

    async def _wait_task(
        self,
        node: str,
        upid: str,
        *,
        timeout_s: float,
        poll_interval_s: float,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        path = f"/nodes/{quote(node, safe='')}/tasks/{quote(upid, safe='')}/status"
        while True:
            status = await self._request("GET", path)
            if isinstance(status, dict) and status.get("status") == "stopped":
                exit_status = str(status.get("exitstatus") or "")
                if exit_status != "OK":
                    raise RuntimeError(f"Proxmox task failed: {exit_status or status}")
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Proxmox task did not finish within {timeout_s:g}s")
            await asyncio.sleep(poll_interval_s)

    async def _preflight(self, snapshot: ProxmoxSnapshotConfig) -> None:
        if snapshot in self._verified_templates:
            return
        async with self._preflight_lock:
            if snapshot in self._verified_templates:
                return
            await self._request("GET", "/version")
            node = quote(snapshot.node, safe="")
            template = await self._request(
                "GET",
                f"/nodes/{node}/qemu/{snapshot.template_vmid}/config",
            )
            if not isinstance(template, dict) or int(template.get("template") or 0) != 1:
                raise RuntimeError(
                    f"Proxmox VM {snapshot.template_vmid} on {snapshot.node} "
                    "must be converted to a template"
                )
            if snapshot.novnc_enabled:
                binary = (
                    shutil.which(snapshot.novnc_websockify)
                    if "/" not in snapshot.novnc_websockify
                    else snapshot.novnc_websockify
                )
                if not binary or not os.access(binary, os.X_OK):
                    raise RuntimeError(
                        f"Proxmox noVNC requires executable {snapshot.novnc_websockify!r}"
                    )
                web_root = Path(snapshot.novnc_web_root).expanduser()
                if not (web_root / "core" / "rfb.js").is_file():
                    raise RuntimeError(
                        f"Proxmox noVNC web root is missing {web_root / 'core' / 'rfb.js'}"
                    )
                viewer_path = (
                    Path(__file__).resolve().parents[1] / "images" / "proxmox_novnc" / "index.html"
                )
                if not viewer_path.is_file():
                    raise RuntimeError(f"Proxmox noVNC viewer is missing {viewer_path}")
                for tls_path in (snapshot.novnc_tls_cert, snapshot.novnc_tls_key):
                    if tls_path and not os.access(tls_path, os.R_OK):
                        raise RuntimeError(f"Proxmox noVNC TLS file is not readable: {tls_path}")
            self._verified_templates.add(snapshot)

    @staticmethod
    def _resolve_shape(
        snapshot: ProxmoxSnapshotConfig,
        spec: SandboxSpec,
    ) -> tuple[int, int]:
        from ..images import get as get_image
        from .gcloud import _parse_gce_machine_type

        default_machine_type = get_image(snapshot.image).default_machine_type
        shape = _parse_gce_machine_type(spec.machine_type or default_machine_type)
        vcpus = snapshot.vcpus or spec.vcpus or (shape.vcpus if shape else 4)
        memory_gb = snapshot.memory_gb or spec.memory_gb or (shape.memory_gb if shape else 8)
        return vcpus, memory_gb

    async def _resolve_guest_ip(
        self,
        snapshot: ProxmoxSnapshotConfig,
        vmid: int,
    ) -> str:
        if snapshot.guest_ip:
            return snapshot.guest_ip

        node = quote(snapshot.node, safe="")
        deadline = time.monotonic() + snapshot.ready_timeout_s
        last_error = "QEMU Guest Agent returned no usable address"
        while True:
            try:
                payload = await self._request(
                    "GET",
                    f"/nodes/{node}/qemu/{vmid}/agent/network-get-interfaces",
                )
                addresses = _usable_agent_ips(payload, snapshot.guest_cidr)
                if addresses:
                    return addresses[0]
            except ProxmoxApiError as exc:
                last_error = exc.detail

            status = await self._request(
                "GET",
                f"/nodes/{node}/qemu/{vmid}/status/current",
            )
            if isinstance(status, dict) and status.get("status") != "running":
                raise RuntimeError(
                    f"Proxmox VM {vmid} stopped before its guest address was available"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Proxmox VM {vmid} has no usable guest address after "
                    f"{snapshot.ready_timeout_s}s: {last_error}"
                )
            await asyncio.sleep(snapshot.poll_interval_s)

    async def _wait_vnc_ready(
        self,
        snapshot: ProxmoxSnapshotConfig,
        guest_ip: str,
    ) -> None:
        deadline = time.monotonic() + snapshot.novnc_ready_timeout_s
        last_error = "no RFB greeting"
        while True:
            writer: asyncio.StreamWriter | None = None
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(guest_ip, snapshot.novnc_guest_port),
                    timeout=3,
                )
                greeting = await asyncio.wait_for(reader.read(12), timeout=3)
                if greeting.startswith(b"RFB "):
                    return
                last_error = f"unexpected greeting {greeting[:12]!r}"
            except (OSError, TimeoutError) as exc:
                last_error = str(exc)
            finally:
                if writer is not None:
                    writer.close()
                    try:
                        await writer.wait_closed()
                    except OSError:
                        pass
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"VNC server at {guest_ip}:{snapshot.novnc_guest_port} "
                    f"did not become ready: {last_error}"
                )
            await asyncio.sleep(snapshot.poll_interval_s)

    @staticmethod
    def _novnc_pid_matches(pid: int, target_file: Path) -> bool:
        try:
            command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            return False
        return b"websockify" in command and str(target_file).encode() in command

    def _find_novnc_pid(self, target_file: Path) -> int:
        proc_root = Path("/proc")
        for entry in proc_root.iterdir():
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            if self._novnc_pid_matches(pid, target_file):
                return pid
        return 0

    async def _terminate_novnc_pid(self, pid: int, target_file: Path) -> None:
        if not self._novnc_pid_matches(pid, target_file):
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + 5
        while self._novnc_pid_matches(pid, target_file):
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.05)
        if self._novnc_pid_matches(pid, target_file):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                return

    async def _start_novnc_bridge(
        self,
        snapshot: ProxmoxSnapshotConfig,
        vmid: int,
        guest_ip: str,
        sandbox: SandboxHandle | None = None,
    ) -> dict[str, Any]:
        if not snapshot.novnc_enabled:
            return {}

        token = secrets.token_urlsafe(24)
        if snapshot.novnc_manage_guest_password:
            if sandbox is None or not sandbox.is_linux:
                raise RuntimeError("Proxmox managed VNC passwords require a Linux sandbox handle")
            vnc_password = token[:8]
            result = await sandbox.run_command(
                'install -d -m 700 "$HOME/.vnc" && '
                f"x11vnc -storepasswd {shlex.quote(vnc_password)} "
                '"$HOME/.vnc/passwd" >/dev/null && '
                'chmod 600 "$HOME/.vnc/passwd" && '
                "pkill -KILL -x x11vnc"
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout or "unknown error")[:500]
                raise RuntimeError(f"failed to rotate guest VNC password: {detail}")

        await self._wait_vnc_ready(snapshot, guest_ip)
        binary = (
            shutil.which(snapshot.novnc_websockify)
            if "/" not in snapshot.novnc_websockify
            else snapshot.novnc_websockify
        )
        if not binary:
            raise RuntimeError(f"Proxmox noVNC requires executable {snapshot.novnc_websockify!r}")

        runtime_root = Path(snapshot.novnc_runtime_root).expanduser().resolve()
        runtime_root.mkdir(parents=True, exist_ok=True)
        runtime_dir = Path(tempfile.mkdtemp(prefix=f"vm-{vmid}-", dir=runtime_root))
        target_file = runtime_dir / "targets"
        viewer_path = (
            Path(__file__).resolve().parents[1] / "images" / "proxmox_novnc" / "index.html"
        )
        await asyncio.to_thread(
            shutil.copytree,
            Path(snapshot.novnc_web_root).expanduser().resolve(),
            runtime_dir,
            dirs_exist_ok=True,
        )
        await asyncio.to_thread(shutil.copyfile, viewer_path, runtime_dir / "index.html")
        target_host = f"[{guest_ip}]" if ":" in guest_ip else guest_ip
        target_file.write_text(
            f"{token}: {target_host}:{snapshot.novnc_guest_port}\n",
            encoding="utf-8",
        )
        target_file.chmod(0o600)

        bind = snapshot.novnc_bind_address
        source_host = f"[{bind}]" if ":" in bind else bind
        connect_host = "127.0.0.1" if bind == "0.0.0.0" else bind
        last_error = "no free port"
        try:
            async with self._novnc_lock:
                for port in range(snapshot.novnc_port_min, snapshot.novnc_port_max + 1):
                    log_path = runtime_dir / "websockify.log"
                    command = [
                        binary,
                        "--daemon",
                        "--log-file",
                        str(log_path),
                        "--web",
                        str(runtime_dir),
                        "--file-only",
                        "--token-plugin",
                        "TokenFile",
                        "--token-source",
                        str(target_file),
                        "--heartbeat",
                        "30",
                    ]
                    if snapshot.novnc_tls_cert:
                        command.extend(
                            [
                                "--cert",
                                snapshot.novnc_tls_cert,
                                "--key",
                                snapshot.novnc_tls_key,
                                "--ssl-only",
                            ]
                        )
                    command.append(f"{source_host}:{port}")
                    process = await asyncio.create_subprocess_exec(
                        *command,
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                        start_new_session=True,
                    )
                    launcher_status = await process.wait()

                    started = False
                    for _ in range(20):
                        try:
                            _, writer = await asyncio.wait_for(
                                asyncio.open_connection(connect_host, port),
                                timeout=0.5,
                            )
                        except (OSError, TimeoutError):
                            await asyncio.sleep(0.05)
                            continue
                        writer.close()
                        try:
                            await writer.wait_closed()
                        except OSError:
                            pass
                        started = True
                        break

                    daemon_pid = self._find_novnc_pid(target_file)
                    if started and daemon_pid:
                        self._novnc_processes[vmid] = daemon_pid
                        public_host = snapshot.novnc_public_host or (
                            urlsplit(self._cfg.api_url).hostname or "127.0.0.1"
                        )
                        url_host = f"[{public_host}]" if ":" in public_host else public_host
                        fragment_values = {
                            "token": token,
                            "autoconnect": "1",
                            "resize": "scale",
                        }
                        if snapshot.novnc_manage_guest_password:
                            fragment_values["vnc_auth"] = "token"
                        fragment = urlencode(fragment_values)
                        scheme = "https" if snapshot.novnc_tls_cert else "http"
                        return {
                            "novnc_url": f"{scheme}://{url_host}:{port}/#{fragment}",
                            "novnc_port": port,
                            "novnc_pid": daemon_pid,
                            "novnc_runtime_dir": str(runtime_dir),
                            "vnc_target": f"{guest_ip}:{snapshot.novnc_guest_port}",
                        }

                    if daemon_pid:
                        await self._terminate_novnc_pid(daemon_pid, target_file)
                    log_text = log_path.read_text(encoding="utf-8", errors="replace")[-1000:]
                    last_error = log_text.strip() or (
                        f"websockify launcher exited with status {launcher_status} on port {port}"
                    )
                    if "Address already in use" not in last_error and "Errno 98" not in last_error:
                        raise RuntimeError(f"websockify failed: {last_error}")
        except BaseException:
            await asyncio.to_thread(shutil.rmtree, runtime_dir, True)
            raise

        await asyncio.to_thread(shutil.rmtree, runtime_dir, True)
        raise RuntimeError(
            f"no free noVNC port in {snapshot.novnc_port_min}-{snapshot.novnc_port_max}: "
            f"{last_error}"
        )

    async def _stop_novnc_bridge(self, sandbox: SandboxHandle) -> None:
        vmid = int(sandbox.metadata["vmid"])
        runtime_value = str(sandbox.metadata.get("novnc_runtime_dir") or "")
        if not runtime_value:
            return
        runtime_root = (
            Path(self._snapshot_for_sandbox(sandbox).novnc_runtime_root).expanduser().resolve()
        )
        runtime_dir = Path(runtime_value).expanduser().resolve()
        try:
            runtime_dir.relative_to(runtime_root)
        except ValueError:
            logger.warning("refusing to remove unexpected noVNC directory %s", runtime_dir)
            return
        pid = self._novnc_processes.pop(vmid, None)
        if pid is None and sandbox.metadata.get("novnc_pid"):
            pid = int(sandbox.metadata["novnc_pid"])
        if pid:
            await self._terminate_novnc_pid(pid, runtime_dir / "targets")
        await asyncio.to_thread(shutil.rmtree, runtime_dir, True)

    async def _stop_vm(
        self,
        snapshot: ProxmoxSnapshotConfig,
        vmid: int,
        *,
        ignore_missing: bool = False,
    ) -> None:
        node = quote(snapshot.node, safe="")
        try:
            status = await self._request(
                "GET",
                f"/nodes/{node}/qemu/{vmid}/status/current",
            )
        except ProxmoxApiError as exc:
            if ignore_missing and exc.status_code == 404:
                return
            raise
        if isinstance(status, dict) and status.get("status") == "stopped":
            return
        upid = await self._request(
            "POST",
            f"/nodes/{node}/qemu/{vmid}/status/stop",
            data={"timeout": 60},
        )
        await self._wait_task(
            snapshot.node,
            str(upid),
            timeout_s=snapshot.task_timeout_s,
            poll_interval_s=snapshot.poll_interval_s,
        )

    async def _delete_vm(
        self,
        snapshot: ProxmoxSnapshotConfig,
        vmid: int,
        *,
        ignore_missing: bool = False,
    ) -> None:
        await self._stop_vm(snapshot, vmid, ignore_missing=ignore_missing)
        node = quote(snapshot.node, safe="")
        try:
            upid = await self._request(
                "DELETE",
                f"/nodes/{node}/qemu/{vmid}",
                # PVE rejects content in DELETE request bodies.  Destruction
                # options belong in the query string for this endpoint.
                params={"purge": 1, "destroy-unreferenced-disks": 1},
            )
        except ProxmoxApiError as exc:
            if ignore_missing and exc.status_code == 404:
                return
            raise
        await self._wait_task(
            snapshot.node,
            str(upid),
            timeout_s=snapshot.task_timeout_s,
            poll_interval_s=snapshot.poll_interval_s,
        )

    async def acquire(self, spec: SandboxSpec) -> SandboxHandle:
        snapshot = self._cfg.snapshots.get(spec.snapshot)
        if snapshot is None:
            raise KeyError(
                f"snapshot {spec.snapshot!r} not in proxmox provider config "
                f"(known: {sorted(self._cfg.snapshots)})"
            )
        if spec.gpu and not snapshot.allow_gpu:
            raise RuntimeError(
                "proxmox snapshot does not allow GPU tasks; enable allow_gpu only "
                "for a template with working PCI passthrough"
            )

        from ..images import get as get_image
        from .gcloud import wait_cua_ready

        image = get_image(snapshot.image)
        if spec.os and spec.os != image.os:
            logger.warning(
                "os mismatch for %s: task declares %r but image %r is %r",
                spec.snapshot,
                spec.os,
                snapshot.image,
                image.os,
            )
        await self._preflight(snapshot)
        vcpus, memory_gb = self._resolve_shape(snapshot, spec)
        name = _vm_name(snapshot.name_prefix, spec)
        clone_submitted = False
        vmid = 0
        sandbox: SandboxHandle | None = None

        try:
            async with self._allocation_lock:
                vmid = int(await self._request("GET", "/cluster/nextid"))
                clone_data: dict[str, Any] = {
                    "newid": vmid,
                    "name": name,
                    "description": f"Disposable ALE sandbox for {spec.task_id or 'unknown task'}",
                    "full": int(snapshot.full_clone),
                }
                if snapshot.full_clone and snapshot.storage:
                    clone_data["storage"] = snapshot.storage
                if snapshot.pool:
                    clone_data["pool"] = snapshot.pool
                upid = await self._request(
                    "POST",
                    f"/nodes/{quote(snapshot.node, safe='')}/qemu/{snapshot.template_vmid}/clone",
                    data=clone_data,
                )
                clone_submitted = True

            await self._wait_task(
                snapshot.node,
                str(upid),
                timeout_s=snapshot.task_timeout_s,
                poll_interval_s=snapshot.poll_interval_s,
            )
            node = quote(snapshot.node, safe="")
            await self._request(
                "POST",
                f"/nodes/{node}/qemu/{vmid}/config",
                data={
                    "cores": vcpus,
                    "memory": memory_gb * 1024,
                    "onboot": 0,
                    "tags": "ale;ale-run",
                },
            )
            upid = await self._request(
                "POST",
                f"/nodes/{node}/qemu/{vmid}/status/start",
            )
            await self._wait_task(
                snapshot.node,
                str(upid),
                timeout_s=snapshot.task_timeout_s,
                poll_interval_s=snapshot.poll_interval_s,
            )

            guest_ip = await self._resolve_guest_ip(snapshot, vmid)
            endpoint_host = f"[{guest_ip}]" if ":" in guest_ip else guest_ip
            cua_port = image.cua_server_port or _DEFAULT_CUA_PORT
            cua_url = f"http://{endpoint_host}:{cua_port}"
            ready = await wait_cua_ready(
                cua_url,
                image.os,
                snapshot.ready_timeout_s,
                snapshot.poll_interval_s,
            )
            if not ready:
                status = await self._request(
                    "GET",
                    f"/nodes/{node}/qemu/{vmid}/status/current",
                )
                raise RuntimeError(
                    f"CUA server at {cua_url} did not become ready; VM status={status}"
                )

            pve_console_url = (
                f"{self._cfg.api_url}/?console=kvm&novnc=1&vmid={vmid}"
                f"&node={quote(snapshot.node, safe='')}&resize=scale"
            )
            metadata = {
                "provider": "proxmox",
                "vmid": vmid,
                "vm_name": name,
                "node": snapshot.node,
                "template_vmid": snapshot.template_vmid,
                "guest_ip": guest_ip,
                "pve_console_url": pve_console_url,
                "image": image.name,
                "snapshot": spec.snapshot,
                "machine_type": spec.machine_type,
                "vcpus": vcpus,
                "memory_gb": memory_gb,
                "full_clone": snapshot.full_clone,
            }
            sandbox = SandboxHandle(
                id=f"proxmox-{vmid}",
                endpoint=cua_url,
                os=image.os,
                **image.sandbox_paths(),
                metadata=metadata,
            )
            if self._cfg.gcs_sa_key:
                gcs_key_path, gcs_user_project = await self._inject_gcs_credentials(
                    sandbox,
                    self._cfg.gcs_sa_key,
                )
                metadata["gcs_key_path"] = gcs_key_path
                metadata["gcs_user_project"] = gcs_user_project

            bridge_metadata = await self._start_novnc_bridge(
                snapshot,
                vmid,
                guest_ip,
                sandbox,
            )
            metadata.update(bridge_metadata)

            logger.info(
                "Proxmox sandbox %s ready: vmid=%d image=%s vcpus=%d memory=%dG cua=%s",
                name,
                vmid,
                snapshot.image,
                vcpus,
                memory_gb,
                cua_url,
            )
            if bridge_metadata:
                logger.info("Proxmox live desktop: %s", bridge_metadata["novnc_url"])
            return sandbox
        except BaseException:
            if sandbox is not None:
                try:
                    await self._stop_novnc_bridge(sandbox)
                except Exception:
                    logger.exception("failed to clean up Proxmox noVNC bridge for VM %d", vmid)
            if clone_submitted:
                try:
                    await self._delete_vm(snapshot, vmid, ignore_missing=True)
                except Exception:
                    logger.exception(
                        "failed to clean up Proxmox VM %d after acquire failure",
                        vmid,
                    )
            raise

    @staticmethod
    async def _inject_gcs_credentials(
        sandbox: SandboxHandle,
        host_key_path: str,
    ) -> tuple[str, str]:
        key = Path(host_key_path)
        if not key.is_file():
            raise FileNotFoundError(f"gcs_sa_key not found: {host_key_path}")
        data = await asyncio.to_thread(key.read_bytes)
        try:
            parsed = json.loads(data)
            project_id = parsed.get("project_id", "") if isinstance(parsed, dict) else ""
        except json.JSONDecodeError:
            project_id = ""

        if sandbox.is_linux:
            destination = "/tmp/agenthle/gcs-reader.json"
            commands = ["mkdir -p /tmp/agenthle", f"chmod 600 {destination}"]
        else:
            destination = r"C:\agenthle\gcs-reader.json"
            commands = [r"cmd /c if not exist C:\agenthle mkdir C:\agenthle"]

        result = await sandbox.run_command(commands[0])
        if result.returncode != 0:
            raise RuntimeError(
                "failed to prepare Proxmox GCS credential directory: "
                f"{(result.stderr or result.stdout or '')[:300]}"
            )
        await sandbox.write_file(destination, data)
        if len(commands) == 2:
            result = await sandbox.run_command(commands[1])
            if result.returncode != 0:
                raise RuntimeError(
                    "failed to protect Proxmox GCS credential: "
                    f"{(result.stderr or result.stdout or '')[:300]}"
                )
        return destination, str(project_id or "")

    def _snapshot_for_sandbox(self, sandbox: SandboxHandle) -> ProxmoxSnapshotConfig:
        snapshot_tag = str(sandbox.metadata.get("snapshot") or "")
        snapshot = self._cfg.snapshots.get(snapshot_tag)
        if snapshot is None:
            raise KeyError(f"sandbox has unknown Proxmox snapshot {snapshot_tag!r}")
        return snapshot

    async def release(
        self,
        sandbox: SandboxHandle,
        *,
        mode: ReleaseMode = "delete",
    ) -> None:
        if mode == "keep":
            logger.info("Proxmox sandbox %s kept alive", sandbox.id)
            return
        snapshot = self._snapshot_for_sandbox(sandbox)
        vmid = int(sandbox.metadata["vmid"])
        await self._stop_novnc_bridge(sandbox)
        if mode == "stop":
            logger.info("Stopping Proxmox sandbox %s", sandbox.id)
            await self._stop_vm(snapshot, vmid, ignore_missing=True)
            return
        if mode != "delete":
            raise ValueError(f"unknown release mode: {mode!r}")
        logger.info("Deleting Proxmox sandbox %s", sandbox.id)
        await self._delete_vm(snapshot, vmid, ignore_missing=True)

    async def heartbeat(self, sandbox: SandboxHandle) -> None:
        snapshot = self._snapshot_for_sandbox(sandbox)
        vmid = int(sandbox.metadata["vmid"])
        status = await self._request(
            "GET",
            f"/nodes/{quote(snapshot.node, safe='')}/qemu/{vmid}/status/current",
        )
        if not isinstance(status, dict) or status.get("status") != "running":
            raise RuntimeError(f"Proxmox sandbox {sandbox.id} is not running: {status}")

    async def cancel_external(self, sandbox: SandboxHandle) -> None:
        snapshot = self._snapshot_for_sandbox(sandbox)
        await self._stop_novnc_bridge(sandbox)
        await self._stop_vm(
            snapshot,
            int(sandbox.metadata["vmid"]),
            ignore_missing=True,
        )

    def open_session(self, sandbox: SandboxHandle) -> Any:
        from cua_bench.computers.remote import RemoteDesktopSession

        from .gcloud import _init_computer_skip_wait

        session = RemoteDesktopSession(
            api_url=sandbox.endpoint,
            vnc_url=str(sandbox.metadata.get("novnc_url") or ""),
            os_type=sandbox.os,
        )
        _init_computer_skip_wait(session)
        return session


__all__ = [
    "ProxmoxApiError",
    "ProxmoxProvider",
    "ProxmoxProviderConfig",
    "ProxmoxSnapshotConfig",
]
