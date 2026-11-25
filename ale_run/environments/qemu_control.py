"""CUA-independent QEMU probes and token-scoped shared control files."""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_LIMIT = 128 * 1024


async def docker_query(*args: str, timeout: float = 5) -> dict[str, Any]:
    """Run a bounded Docker query, killing its client on timeout/cancellation."""
    proc = await asyncio.create_subprocess_exec(
        "docker",
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def read(stream):
        data = bytearray()
        while len(data) <= _LIMIT:
            chunk = await stream.read(min(16384, _LIMIT + 1 - len(data)))
            if not chunk:
                return data.decode(errors="replace")
            data.extend(chunk)
        raise ValueError("Docker query output exceeded limit")

    try:
        async with asyncio.timeout(timeout):
            stdout, stderr = await asyncio.gather(read(proc.stdout), read(proc.stderr))
            await proc.wait()
            return {"returncode": proc.returncode, "stdout": stdout, "stderr": stderr}
    finally:
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


async def query_qmp(container: str, *, timeout: float = 5) -> dict[str, Any]:
    """Query QMP inside the runner network namespace; never send VM mutations."""
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "exec",
        "-i",
        container,
        "nc",
        "-w",
        "4",
        "127.0.0.1",
        "7200",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        limit=65536,
    )
    try:
        async with asyncio.timeout(timeout):
            greeting = json.loads(await proc.stdout.readline())
            if not isinstance(greeting, dict) or "QMP" not in greeting:
                raise ValueError("missing QMP greeting")
            result = {}
            for command in ("qmp_capabilities", "query-status", "query-kvm"):
                proc.stdin.write((json.dumps({"execute": command, "id": command}) + "\n").encode())
                await proc.stdin.drain()
                for _ in range(64):
                    response = json.loads(await proc.stdout.readline())
                    if response.get("id") != command:
                        continue
                    if "error" in response:
                        raise RuntimeError(f"QMP {command}: {response['error']}")
                    result[command] = response["return"]
                    break
                else:
                    raise ValueError("too many unmatched QMP messages")
            return {"status": result["query-status"], "kvm": result["query-kvm"]}
    finally:
        proc.stdin.close()
        if proc.returncode is None:
            proc.kill()
        await proc.wait()


async def inspect_health(sandbox) -> dict[str, Any]:
    """Report hypervisor state separately from unknown guest/solver health."""
    report: dict[str, Any] = {"guest": "unknown", "observed_at": time.time()}
    name = sandbox.metadata.get("container_name") or sandbox.id

    async def inspect():
        # Do not collect Config.Env, mount contents, or arbitrary inspect fields.
        raw = await docker_query("inspect", "--format", "{{json .State}}", name)
        if raw["returncode"]:
            return {"error": raw["stderr"][:1000]}
        state = json.loads(raw["stdout"])
        return {
            key: state.get(key)
            for key in (
                "Status",
                "Running",
                "Paused",
                "OOMKilled",
                "Dead",
                "Pid",
                "ExitCode",
                "StartedAt",
                "FinishedAt",
            )
        }

    values = await asyncio.gather(inspect(), query_qmp(name), return_exceptions=True)
    for key, value in zip(("container", "qmp"), values):
        report[key] = (
            {"error": f"{type(value).__name__}: {value}"}
            if isinstance(value, BaseException)
            else value
        )
    return report


def read_control(directory: Path, name: str, token: str) -> dict | None:
    """Read a size-bounded, token-matched control record from the host share."""
    try:
        with (directory / name).open("rb") as stream:
            raw = stream.read(_LIMIT + 1)
        if len(raw) > _LIMIT:
            return None
        record = json.loads(raw)
        if isinstance(record, dict) and record.get("run_token") == token:
            return record
    except (OSError, ValueError):
        pass
    return None


@dataclass
class HeartbeatWatch:
    """Freshness uses host monotonic observation of sequence advances, not guest time."""

    sequence: int | None = None
    advanced_at: float | None = None
    sampled_at: float | None = None

    def fresh(self, report: dict | None, *, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if not report or not isinstance(report.get("state"), str):
            return False
        if report["state"] not in {"starting", "running", "draining"}:
            return False
        sequence = report.get("sequence")
        if type(sequence) is not int or sequence < 0:
            return False
        if self.sequence is not None and sequence < self.sequence:
            return False
        gap = now - self.sampled_at if self.sampled_at is not None else None
        if gap is None or gap > 45:
            self.advanced_at = None
        elif self.sequence is not None and sequence > self.sequence:
            self.advanced_at = now
        self.sequence = sequence
        self.sampled_at = now
        return self.advanced_at is not None and now - self.advanced_at <= 45

    async def check(self, directory: Path, token: str) -> bool:
        report = read_control(directory, "heartbeat.json", token)
        if self.fresh(report):
            return True
        if report is None:
            return False
        await asyncio.sleep(6)
        return self.fresh(read_control(directory, "heartbeat.json", token))


async def prepare_control(sandbox, token: str) -> tuple[Path, str] | None:
    """Mount a per-run control directory before launch and verify bidirectional I/O."""
    metadata = sandbox.metadata or {}
    if metadata.get("provider") != "qemu":
        return None
    if not re.fullmatch(r"[a-z0-9-]+", token):
        raise ValueError("invalid control token")
    host = Path(metadata["exchange_host_dir"]) / "control" / token
    host.mkdir(parents=True, exist_ok=False)
    (host / "host-probe").write_text(token)
    if sandbox.is_linux:
        mount = f"/tmp/ale-control-{token}"
        guest = f"{mount}/control/{token}"
        command = (
            f"set -eu; mkdir -p {shlex.quote(mount)}; "
            f"sudo -n mount -t cifs {shlex.quote(metadata['exchange_guest_share'])} "
            f"{shlex.quote(mount)} -o guest,vers=3.0,cache=none,actimeo=0,soft,"
            "uid=$(id -u),gid=$(id -g),file_mode=0600,dir_mode=0700; "
            f'test "$(cat {shlex.quote(guest + "/host-probe")})" = {shlex.quote(token)}; '
            f"printf %s {shlex.quote(token)} > {shlex.quote(guest + '/guest-probe')}"
        )
    else:
        guest = metadata["exchange_guest_share"].rstrip("\\") + f"\\control\\{token}"
        command = (
            'powershell -NoProfile -Command "'
            f"if ((Get-Content -Raw -LiteralPath '{guest}\\host-probe') -ne '{token}') "
            "{ exit 1 }; "
            f"[IO.File]::WriteAllText('{guest}\\guest-probe','{token}')\""
        )
    async with asyncio.timeout(45):
        result = await sandbox.run_command(command, timeout=30)
    if result.returncode != 0 or (host / "guest-probe").read_text() != token:
        raise RuntimeError("QEMU independent shared-control handshake failed")
    metadata["control_host_dir"] = str(host)
    return host, guest


async def collect_diagnostics(sandbox, destination: Path) -> dict[str, Any]:
    """Persist bounded independent evidence before VM deletion, including missing probes."""
    destination.mkdir(parents=True, exist_ok=True)
    report = await inspect_health(sandbox)
    name = sandbox.metadata.get("container_name") or sandbox.id
    for key, args in (
        # 只记录网络地址，不读取可能包含密钥的容器环境变量。
        ("runner_network", ("inspect", "--format", "{{json .NetworkSettings.Networks}}", name)),
        ("runner_log", ("logs", "--tail", "120", name)),
        ("qemu_log", ("exec", name, "tail", "-c", "32768", "/run/shm/qemu.log")),
    ):
        try:
            report[key] = await docker_query(*args)
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            report[key] = {"error": f"{type(exc).__name__}: {exc}"}
    directory = sandbox.metadata.get("control_host_dir")
    report["control"] = {}
    if directory:
        directory = Path(directory)
        for filename in ("heartbeat.json", "stopped.json", "diagnostics.json"):
            report["control"][filename] = read_control(directory, filename, directory.name)
    else:
        report["control"]["error"] = "independent guest control was not initialized"
    report["probe_history"] = sandbox.metadata.get("health_probe_history", [])
    target = destination / "qemu.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(target)
    return {"path": str(target), "guest": report["guest"]}


def main() -> None:
    """Read-only health/diagnostic CLI for an existing runner container."""
    import argparse
    from types import SimpleNamespace

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("container")
    parser.add_argument("--output", type=Path, help="Persist diagnostics in this directory")
    parser.add_argument("--control-dir", type=Path, help="Host-side token-scoped control directory")
    arguments = parser.parse_args()
    metadata = {"container_name": arguments.container}
    if arguments.control_dir is not None:
        metadata["control_host_dir"] = str(arguments.control_dir)
    sandbox = SimpleNamespace(id=arguments.container, metadata=metadata)
    operation = (
        collect_diagnostics(sandbox, arguments.output)
        if arguments.output is not None
        else inspect_health(sandbox)
    )
    print(json.dumps(asyncio.run(operation), indent=2))


if __name__ == "__main__":
    main()
