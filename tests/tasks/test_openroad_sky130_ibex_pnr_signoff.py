from __future__ import annotations

from types import SimpleNamespace

import pytest

from tasks.engineering.openroad_sky130_ibex_pnr_signoff import main


def task_metadata() -> dict[str, str]:
    return {
        "variant_name": "base",
        "reference_dir": "/task/reference",
        "reference_frozen_hashes": "/task/reference/frozen_hashes.json",
        "reference_metrics": "/task/reference/reference_metrics.json",
        "reference_starter_zip": "/task/reference/starter_project.zip",
        "remote_output_dir": "/task/output",
        "output_dir_name": "output",
    }


class FakeInterface:
    async def create_dir(self, _path: str) -> None:
        return None


class FakeSession:
    def __init__(self, *, missing: set[str] | None = None):
        self.interface = FakeInterface()
        self.missing = missing or set()

    async def file_exists(self, path: str) -> bool:
        return path not in self.missing

    async def directory_exists(self, _path: str) -> bool:
        return False

    async def run_command(self, _command: str, *, check: bool = False) -> dict[str, object]:
        del check
        return {"return_code": 0, "stdout": "/tmp/eval-run\n", "stderr": ""}

    async def write_file(self, _path: str, _content: str) -> None:
        return None


class FinishedVerifierSession:
    def __init__(self):
        self.commands: list[str] = []
        self.written_files: dict[str, str] = {}

    async def write_file(self, path: str, content: str) -> None:
        self.written_files[path] = content

    async def run_command(self, command: str, *, check: bool = False) -> dict[str, object]:
        del check
        self.commands.append(command)
        if "systemd-run" in command:
            return {"return_code": 0, "stdout": "4321\n", "stderr": ""}
        if "test -f" in command:
            return {"return_code": 0, "stdout": "DONE\n", "stderr": ""}
        if "/verifier.rc" in command:
            return {"return_code": 0, "stdout": "0\n", "stderr": ""}
        if "/verifier.stdout" in command:
            return {"return_code": 0, "stdout": '{"normalized_score": 1.0}\n', "stderr": ""}
        return {"return_code": 0, "stdout": "", "stderr": ""}


@pytest.mark.asyncio
async def test_missing_submission_remains_a_zero_score():
    metadata = task_metadata()
    session = FakeSession(missing={metadata["remote_output_dir"]})

    assert await main.evaluate(SimpleNamespace(metadata=metadata), session) == [0.0]


@pytest.mark.asyncio
async def test_missing_benchmark_reference_is_an_infrastructure_failure():
    metadata = task_metadata()
    session = FakeSession(missing={metadata["reference_metrics"]})

    with pytest.raises(RuntimeError, match="missing evaluator reference paths"):
        await main.evaluate(SimpleNamespace(metadata=metadata), session)


@pytest.mark.asyncio
async def test_verifier_transport_failure_is_not_converted_to_zero(monkeypatch):
    async def fail(*_args, **_kwargs):
        raise RuntimeError("Request failed")

    monkeypatch.setattr(main, "_run_verifier_background", fail)

    with pytest.raises(RuntimeError, match="Request failed"):
        await main.evaluate(
            SimpleNamespace(metadata=task_metadata()),
            FakeSession(),
        )


@pytest.mark.asyncio
async def test_malformed_verifier_output_is_not_converted_to_zero(monkeypatch):
    async def malformed(*_args, **_kwargs):
        return "not-json", "verifier stderr"

    monkeypatch.setattr(main, "_run_verifier_background", malformed)

    with pytest.raises(RuntimeError, match="invalid JSON output"):
        await main.evaluate(
            SimpleNamespace(metadata=task_metadata()),
            FakeSession(),
        )


@pytest.mark.asyncio
async def test_verifier_launch_fully_detaches_from_cua_command(monkeypatch):
    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(main.asyncio, "sleep", no_sleep)
    session = FinishedVerifierSession()

    stdout, stderr = await main._run_verifier_background(
        session,
        ["python", "/tmp/verify_submission.py"],
        tag="base",
        run_dir="/tmp/eval-run",
    )

    assert stdout == '{"normalized_score": 1.0}\n'
    assert stderr == ""
    assert "systemd-run" in session.commands[0]
    assert "--collect" in session.commands[0]
    assert "--uid=user" in session.commands[0]
    assert "verifier.pid" in session.commands[0]
    assert "nohup" not in session.commands[0]
    assert "$(" not in session.commands[0]
    assert "$?" not in session.commands[0]
    launcher = session.written_files["/tmp/eval-run/run_verifier.sh"]
    assert '"$$"' in launcher
    assert "rc=$?" in launcher
    assert '"$rc"' in launcher
