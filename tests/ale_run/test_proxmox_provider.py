from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ale_run.base_interface import SandboxHandle, SandboxSpec
from ale_run.environments.providers.proxmox import (
    ProxmoxApiError,
    ProxmoxProvider,
    _ProxmoxApi,
    _usable_agent_ips,
)
from ale_run.orchestration.config_loader import load_experiment
from ale_run.orchestration.experiment_spec import ProviderSpec
from ale_run.orchestration.factory import build_provider


def _provider_config(**snapshot_overrides: Any) -> dict[str, Any]:
    return {
        "api_url": "https://pve.example:8006",
        "token_id": "ale@pve!ale-run",
        "token_secret": "test-secret",
        "verify_ssl": False,
        "snapshots": {
            "cpu-free-ubuntu": {
                "image": "ale-ubuntu22",
                "node": "seed3",
                "template_vmid": 9000,
                "guest_ip": "192.168.0.50",
                "poll_interval_s": 0.001,
                **snapshot_overrides,
            }
        },
    }


def test_loader_builds_proxmox_snapshot_map(tmp_path: Path) -> None:
    agent = tmp_path / "agent.yaml"
    agent.write_text("harness: dummy\nmodel: test\n", encoding="utf-8")
    environment = tmp_path / "environment.yaml"
    environment.write_text(
        """
snapshots:
  cpu-free-ubuntu:
    provider: proxmox
    image: ale-ubuntu22
    proxmox:
      api_url: https://pve.example:8006
      token_id: ale@pve!ale-run
      token_secret: test-secret
      verify_ssl: false
      node: seed3
      template_vmid: 9000
      guest_cidr: 192.168.0.0/24
      novnc_enabled: true
      novnc_bind_address: 0.0.0.0
      novnc_public_host: 192.168.0.14
      novnc_web_root: /usr/share/novnc
task_data_source: baked_in_sandbox
output_path: local
""",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        f"""
name: proxmox-test
agent: {agent}
environment: {environment}
tasks:
  - path: demo/hello
""",
        encoding="utf-8",
    )

    spec = load_experiment(experiment)

    assert spec.environment.snapshot_kind == {"cpu-free-ubuntu": "proxmox"}
    provider = spec.environment.provider_specs["proxmox"]
    assert provider.config == {
        "api_url": "https://pve.example:8006",
        "token_id": "ale@pve!ale-run",
        "token_secret": "test-secret",
        "verify_ssl": False,
        "snapshots": {
            "cpu-free-ubuntu": {
                "image": "ale-ubuntu22",
                "node": "seed3",
                "template_vmid": 9000,
                "guest_cidr": "192.168.0.0/24",
                "novnc_enabled": True,
                "novnc_bind_address": "0.0.0.0",
                "novnc_public_host": "192.168.0.14",
                "novnc_web_root": "/usr/share/novnc",
            }
        },
    }


@pytest.mark.parametrize("missing", ["api_url", "token_id", "token_secret"])
def test_loader_rejects_missing_proxmox_connection_field(
    tmp_path: Path,
    missing: str,
) -> None:
    values = {
        "api_url": "https://pve.example:8006",
        "token_id": "ale@pve!ale-run",
        "token_secret": "test-secret",
    }
    del values[missing]
    connection = "\n".join(f"      {key}: {value}" for key, value in values.items())
    agent = tmp_path / "agent.yaml"
    agent.write_text("harness: dummy\n", encoding="utf-8")
    environment = tmp_path / "environment.yaml"
    environment.write_text(
        f"""
snapshots:
  cpu-free-ubuntu:
    provider: proxmox
    image: ale-ubuntu22
    proxmox:
{connection}
      node: seed3
      template_vmid: 9000
""",
        encoding="utf-8",
    )
    experiment = tmp_path / "experiment.yaml"
    experiment.write_text(
        f"""
name: proxmox-test
agent: {agent}
environment: {environment}
tasks:
  - path: demo/hello
""",
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match=missing):
        load_experiment(experiment)


def test_factory_builds_proxmox_without_affecting_qemu() -> None:
    proxmox = build_provider(ProviderSpec(kind="proxmox", config=_provider_config()))

    assert isinstance(proxmox, ProxmoxProvider)
    assert proxmox.config.snapshots["cpu-free-ubuntu"].template_vmid == 9000


def test_provider_validates_token_and_template() -> None:
    bad_token = _provider_config()
    bad_token["token_id"] = "ale@pve"
    with pytest.raises(ValueError, match="user@realm!token-name"):
        ProxmoxProvider(bad_token)

    with pytest.raises(ValueError, match="template_vmid"):
        ProxmoxProvider(_provider_config(template_vmid=0))

    with pytest.raises(ValueError, match="noVNC port range"):
        ProxmoxProvider(_provider_config(novnc_port_min=7000, novnc_port_max=6999))

    with pytest.raises(ValueError, match="must be set together"):
        ProxmoxProvider(_provider_config(novnc_tls_cert="/tmp/cert.pem"))


def test_guest_agent_ip_selection_prefers_matching_ipv4() -> None:
    payload = {
        "result": [
            {
                "name": "lo",
                "ip-addresses": [
                    {"ip-address": "127.0.0.1"},
                    {"ip-address": "::1"},
                ],
            },
            {
                "name": "eth0",
                "ip-addresses": [
                    {"ip-address": "fe80::1"},
                    {"ip-address": "10.20.0.9"},
                    {"ip-address": "192.168.0.51"},
                ],
            },
        ]
    }

    assert _usable_agent_ips(payload, "192.168.0.0/24") == ["192.168.0.51"]
    assert _usable_agent_ips(payload, "") == ["10.20.0.9", "192.168.0.51"]


def test_api_client_uses_token_header_and_unwraps_data() -> None:
    api = _ProxmoxApi(
        api_url="https://pve.example:8006",
        token_id="ale@pve!ale-run",
        token_secret="test-secret",
        verify_ssl=False,
        timeout_s=12,
    )
    calls: list[dict[str, Any]] = []

    class Response:
        ok = True
        status_code = 200
        text = ""

        @staticmethod
        def json() -> dict[str, Any]:
            return {"data": {"version": "9.2"}}

    def fake_request(method: str, url: str, **kwargs: Any) -> Response:
        calls.append({"method": method, "url": url, **kwargs})
        return Response()

    api._session.request = fake_request  # type: ignore[method-assign]

    assert api.request("GET", "/version") == {"version": "9.2"}
    assert api._session.headers["Authorization"] == ("PVEAPIToken=ale@pve!ale-run=test-secret")
    assert calls == [
        {
            "method": "GET",
            "url": "https://pve.example:8006/api2/json/version",
            "params": None,
            "data": None,
            "timeout": 12,
            "verify": False,
        }
    ]


def test_api_error_does_not_include_token() -> None:
    api = _ProxmoxApi(
        api_url="https://pve.example:8006",
        token_id="ale@pve!ale-run",
        token_secret="never-print-this",
        verify_ssl=False,
        timeout_s=12,
    )

    class Response:
        ok = False
        status_code = 403
        text = "permission denied"

        @staticmethod
        def json() -> dict[str, Any]:
            return {"message": "permission denied"}

    api._session.request = lambda *args, **kwargs: Response()  # type: ignore[method-assign]

    with pytest.raises(ProxmoxApiError, match="permission denied") as exc_info:
        api.request("GET", "/nodes")
    assert "never-print-this" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_preflight_rejects_normal_vm_as_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ProxmoxProvider(_provider_config())

    async def fake_request(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        _ = (method, params, data)
        if path == "/version":
            return {}
        if path == "/nodes/seed3/qemu/9000/config":
            return {"template": 0}
        raise AssertionError(path)

    monkeypatch.setattr(provider, "_request", fake_request)

    with pytest.raises(RuntimeError, match="must be converted to a template"):
        await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu", task_id="demo/hello"))


@pytest.mark.asyncio
async def test_novnc_bridge_serves_viewer_routes_token_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    web_root = tmp_path / "novnc"
    (web_root / "core").mkdir(parents=True)
    (web_root / "vendor").mkdir()
    (web_root / "core" / "rfb.js").write_text("export default class RFB {}\n")
    (web_root / "vendor" / "codec.js").write_text("export const codec = true;\n")
    runtime_root = tmp_path / "runtime"
    tls_cert = tmp_path / "server.pem"
    tls_key = tmp_path / "server.key"
    tls_cert.write_text("test certificate\n")
    tls_key.write_text("test key\n")
    provider = ProxmoxProvider(
        _provider_config(
            novnc_enabled=True,
            novnc_bind_address="0.0.0.0",
            novnc_public_host="192.168.0.14",
            novnc_port_min=6085,
            novnc_port_max=6085,
            novnc_websockify="/usr/bin/websockify",
            novnc_web_root=str(web_root),
            novnc_runtime_root=str(runtime_root),
            novnc_tls_cert=str(tls_cert),
            novnc_tls_key=str(tls_key),
            novnc_manage_guest_password=True,
        )
    )
    snapshot = provider.config.snapshots["cpu-free-ubuntu"]
    command: list[str] = []
    terminated: list[int] = []

    class FakeProcess:
        pid = 4321

        async def wait(self) -> int:
            return 0

    class FakeWriter:
        def close(self) -> None:
            pass

        async def wait_closed(self) -> None:
            pass

    process = FakeProcess()
    guest_commands: list[str] = []

    async def run_guest_command(command: str) -> SimpleNamespace:
        guest_commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    guest = SimpleNamespace(is_linux=True, run_command=run_guest_command)

    async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> FakeProcess:
        _ = kwargs
        command.extend(str(arg) for arg in args)
        return process

    async def fake_open_connection(*args: Any, **kwargs: Any) -> tuple[object, FakeWriter]:
        _ = (args, kwargs)
        return object(), FakeWriter()

    async def fake_wait_vnc(*args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)

    async def fake_terminate(pid: int, target_file: Path) -> None:
        assert target_file.name == "targets"
        terminated.append(pid)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(asyncio, "open_connection", fake_open_connection)
    monkeypatch.setattr(provider, "_wait_vnc_ready", fake_wait_vnc)
    monkeypatch.setattr(provider, "_find_novnc_pid", lambda target_file: 4321)
    monkeypatch.setattr(provider, "_terminate_novnc_pid", fake_terminate)
    monkeypatch.setattr("secrets.token_urlsafe", lambda size: "test-token")

    metadata = await provider._start_novnc_bridge(
        snapshot,
        1200,
        "192.168.0.50",
        guest,  # type: ignore[arg-type]
    )

    runtime_dir = Path(metadata["novnc_runtime_dir"])
    assert metadata["novnc_url"] == (
        "https://192.168.0.14:6085/#token=test-token&autoconnect=1&resize=scale&vnc_auth=token"
    )
    assert len(guest_commands) == 1
    assert "x11vnc -storepasswd test-tok" in guest_commands[0]
    assert "pkill -KILL -x x11vnc" in guest_commands[0]
    assert metadata["vnc_target"] == "192.168.0.50:5900"
    viewer = (runtime_dir / "index.html").read_text(encoding="utf-8")
    assert viewer.startswith("<!doctype html>")
    assert "rfb.sendCredentials" in viewer
    assert (runtime_dir / "core" / "rfb.js").is_file()
    assert (runtime_dir / "vendor" / "codec.js").is_file()
    assert (runtime_dir / "targets").read_text(encoding="utf-8") == (
        "test-token: 192.168.0.50:5900\n"
    )
    assert command[:2] == ["/usr/bin/websockify", "--daemon"]
    assert command[command.index("--web") + 1] == str(runtime_dir)
    assert command[command.index("--cert") + 1] == str(tls_cert)
    assert command[command.index("--key") + 1] == str(tls_key)
    assert "--ssl-only" in command
    assert command[-1] == "0.0.0.0:6085"

    sandbox = SandboxHandle(
        id="proxmox-1200",
        endpoint="http://192.168.0.50:5000",
        os="linux",
        work_dir_base="",
        task_data_root="",
        node="",
        python="",
        mcp_server_dir="",
        metadata={
            "vmid": 1200,
            "snapshot": "cpu-free-ubuntu",
            **metadata,
        },
    )
    await provider._stop_novnc_bridge(sandbox)

    assert terminated == [4321]
    assert not runtime_dir.exists()


@pytest.mark.asyncio
async def test_acquire_clones_template_starts_vm_and_returns_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ProxmoxProvider(_provider_config())
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    async def fake_request(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        _ = params
        calls.append((method, path, data))
        if path == "/version":
            return {"version": "9.2"}
        if path == "/nodes/seed3/qemu/9000/config":
            return {"template": 1}
        if path == "/cluster/nextid":
            return 1200
        if path.endswith("/clone"):
            return "UPID:seed3:clone"
        if path.endswith("/status/start"):
            return "UPID:seed3:start"
        if "/tasks/" in path:
            return {"status": "stopped", "exitstatus": "OK"}
        if path.endswith("/config") and method == "POST":
            return None
        raise AssertionError(f"unexpected request: {method} {path} {data}")

    async def ready(
        cua_url: str,
        os_type: str,
        timeout: float,
        poll_interval: float,
    ) -> bool:
        assert cua_url == "http://192.168.0.50:5000"
        assert os_type == "linux"
        assert timeout == 900
        assert poll_interval == 0.001
        return True

    monkeypatch.setattr(provider, "_request", fake_request)
    monkeypatch.setattr(
        "ale_run.environments.providers.gcloud.wait_cua_ready",
        ready,
    )

    async def start_novnc(*args: Any, **kwargs: Any) -> dict[str, Any]:
        _ = (args, kwargs)
        return {
            "novnc_url": "http://192.168.0.14:6080/#token=test-token",
            "novnc_port": 6080,
            "novnc_pid": 4321,
            "novnc_runtime_dir": "/tmp/ale-proxmox-novnc/vm-1200-test",
            "vnc_target": "192.168.0.50:5900",
        }

    monkeypatch.setattr(provider, "_start_novnc_bridge", start_novnc)

    sandbox = await provider.acquire(
        SandboxSpec(
            snapshot="cpu-free-ubuntu",
            os="linux",
            task_id="demo/hello",
            vcpus=6,
            memory_gb=12,
        )
    )

    assert sandbox.id == "proxmox-1200"
    assert sandbox.endpoint == "http://192.168.0.50:5000"
    assert sandbox.metadata["provider"] == "proxmox"
    assert sandbox.metadata["vmid"] == 1200
    assert sandbox.metadata["template_vmid"] == 9000
    assert sandbox.metadata["vcpus"] == 6
    assert sandbox.metadata["memory_gb"] == 12
    assert sandbox.metadata["novnc_url"] == ("http://192.168.0.14:6080/#token=test-token")
    assert "vmid=1200" in sandbox.metadata["pve_console_url"]
    assert sandbox.metadata["vnc_target"] == "192.168.0.50:5900"
    clone_call = next(call for call in calls if call[1].endswith("/clone"))
    assert clone_call[2] is not None
    assert clone_call[2]["newid"] == 1200
    assert clone_call[2]["full"] == 0
    config_call = next(
        call for call in calls if call[:2] == ("POST", "/nodes/seed3/qemu/1200/config")
    )
    assert config_call[2] == {
        "cores": 6,
        "memory": 12288,
        "onboot": 0,
        "tags": "ale;ale-run",
    }


@pytest.mark.asyncio
async def test_acquire_discovers_guest_ip_with_qemu_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ProxmoxProvider(_provider_config(guest_ip="", guest_cidr="10.0.0.0/24"))

    async def fake_request(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        _ = (method, params, data)
        if path == "/version":
            return {}
        if path == "/nodes/seed3/qemu/9000/config":
            return {"template": 1}
        if path == "/cluster/nextid":
            return 1201
        if path.endswith("/clone"):
            return "UPID:seed3:clone"
        if path.endswith("/status/start"):
            return "UPID:seed3:start"
        if "/tasks/" in path:
            return {"status": "stopped", "exitstatus": "OK"}
        if path.endswith("/config"):
            return None
        if path.endswith("/agent/network-get-interfaces"):
            return {
                "result": [
                    {"ip-addresses": [{"ip-address": "10.0.0.77"}]},
                ]
            }
        raise AssertionError(path)

    async def ready(*args: Any, **kwargs: Any) -> bool:
        return True

    monkeypatch.setattr(provider, "_request", fake_request)
    monkeypatch.setattr(
        "ale_run.environments.providers.gcloud.wait_cua_ready",
        ready,
    )

    sandbox = await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu", task_id="demo/hello"))

    assert sandbox.endpoint == "http://10.0.0.77:5000"
    assert sandbox.metadata["guest_ip"] == "10.0.0.77"


@pytest.mark.asyncio
async def test_acquire_failure_deletes_submitted_clone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = ProxmoxProvider(_provider_config())
    calls: list[tuple[str, str]] = []

    async def fake_request(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        _ = (params, data)
        calls.append((method, path))
        if path == "/version":
            return {}
        if path == "/nodes/seed3/qemu/9000/config":
            return {"template": 1}
        if path == "/cluster/nextid":
            return 1202
        if path.endswith("/clone"):
            return "UPID:seed3:clone"
        if path.endswith("/config"):
            return None
        if path.endswith("/status/start"):
            return "UPID:seed3:start"
        if path.endswith("/status/current"):
            return {"status": "running"}
        if path.endswith("/status/stop"):
            return "UPID:seed3:stop"
        if method == "DELETE":
            return "UPID:seed3:delete"
        if "/tasks/" in path:
            return {"status": "stopped", "exitstatus": "OK"}
        raise AssertionError(path)

    async def not_ready(*args: Any, **kwargs: Any) -> bool:
        return False

    monkeypatch.setattr(provider, "_request", fake_request)
    monkeypatch.setattr(
        "ale_run.environments.providers.gcloud.wait_cua_ready",
        not_ready,
    )

    with pytest.raises(RuntimeError, match="did not become ready"):
        await provider.acquire(SandboxSpec(snapshot="cpu-free-ubuntu", task_id="demo/hello"))

    assert ("POST", "/nodes/seed3/qemu/1202/status/stop") in calls
    assert ("DELETE", "/nodes/seed3/qemu/1202") in calls


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["stop", "delete"])
async def test_release_stops_or_deletes_vm(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ProxmoxProvider(_provider_config())
    calls: list[tuple[str, str]] = []
    delete_options: list[tuple[dict[str, Any] | None, dict[str, Any] | None]] = []
    bridge_stops: list[int] = []

    async def fake_request(
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        calls.append((method, path))
        if path.endswith("/status/current"):
            return {"status": "running"}
        if path.endswith("/status/stop"):
            return "UPID:seed3:stop"
        if method == "DELETE":
            delete_options.append((params, data))
            return "UPID:seed3:delete"
        if "/tasks/" in path:
            return {"status": "stopped", "exitstatus": "OK"}
        raise AssertionError(path)

    monkeypatch.setattr(provider, "_request", fake_request)

    async def stop_bridge(sandbox: SandboxHandle) -> None:
        bridge_stops.append(int(sandbox.metadata["vmid"]))

    monkeypatch.setattr(provider, "_stop_novnc_bridge", stop_bridge)
    sandbox = SandboxHandle(
        id="proxmox-1200",
        endpoint="http://192.168.0.50:5000",
        os="linux",
        work_dir_base="",
        task_data_root="",
        node="",
        python="",
        mcp_server_dir="",
        metadata={
            "provider": "proxmox",
            "vmid": 1200,
            "snapshot": "cpu-free-ubuntu",
        },
    )

    await provider.release(sandbox, mode=mode)  # type: ignore[arg-type]

    assert ("POST", "/nodes/seed3/qemu/1200/status/stop") in calls
    assert bridge_stops == [1200]
    assert (("DELETE", "/nodes/seed3/qemu/1200") in calls) is (mode == "delete")
    if mode == "delete":
        assert delete_options == [({"purge": 1, "destroy-unreferenced-disks": 1}, None)]


@pytest.mark.asyncio
async def test_release_keep_does_not_call_api(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = ProxmoxProvider(_provider_config())

    async def fail_request(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("keep must not call Proxmox")

    async def fail_bridge(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("keep must not stop noVNC")

    monkeypatch.setattr(provider, "_request", fail_request)
    monkeypatch.setattr(provider, "_stop_novnc_bridge", fail_bridge)
    sandbox = SimpleNamespace(id="proxmox-1200")

    await provider.release(sandbox, mode="keep")  # type: ignore[arg-type]
