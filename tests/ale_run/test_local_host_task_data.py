from __future__ import annotations

import base64
import io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from ale_run.base_interface import TaskDataSpec
from ale_run.environments.task_data import local_host


class _Sandbox:
    task_data_root = "/task-data"

    def __init__(
        self,
        *,
        provider: str = "proxmox",
        is_linux: bool = True,
        existing_paths: set[str] | None = None,
    ) -> None:
        self.id = "sandbox-123"
        self.is_linux = is_linux
        self.metadata = {"provider": provider}
        self.existing_paths = existing_paths or set()
        self.commands: list[str] = []
        self.created: list[str] = []
        self.removed: list[list[str]] = []
        self.uploads: dict[str, bytes] = {}

    async def mkdir(self, path: str) -> None:
        self.created.append(path)

    async def rm(self, paths: list[str]) -> None:
        self.removed.append(paths)

    async def write_file(self, path: str, content: bytes) -> None:
        self.uploads[path] = content

    async def exists(self, path: str) -> bool:
        return path in self.existing_paths

    async def run_command(
        self,
        command: str,
        *,
        timeout: float = 60,
    ) -> SimpleNamespace:
        _ = timeout
        self.commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    async def list_dir(self, path: str) -> list[dict[str, str]]:
        _ = path
        return [{"relpath": "results.json"}]


def _task_data() -> TaskDataSpec:
    return TaskDataSpec(
        requires_task_data=True,
        domain_name="demo",
        task_name="hello",
        variant_name="base",
    )


def _write_task_data(root: Path) -> Path:
    task = root / "demo" / "hello" / "base"
    (task / "input").mkdir(parents=True)
    (task / "input" / "question.txt").write_text("question", encoding="utf-8")
    (task / "software").mkdir()
    (task / "software" / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    (task / "reference").mkdir()
    (task / "reference" / "answer.txt").write_text("answer", encoding="utf-8")
    return task


def _uploaded_archive(sandbox: _Sandbox) -> tarfile.TarFile:
    chunks = [
        content
        for path, content in sorted(sandbox.uploads.items())
        if "/part-" in path.replace("\\", "/")
    ]
    return tarfile.open(fileobj=io.BytesIO(b"".join(chunks)), mode="r:gz")


@pytest.mark.asyncio
async def test_proxmox_stages_only_input_and_software_before_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_task_data(tmp_path)
    monkeypatch.setattr(local_host, "_PROXMOX_UPLOAD_CHUNK_BYTES", 16)
    sandbox = _Sandbox()

    report = await local_host.stage_input(
        sandbox,
        _task_data(),
        source=f"local:{tmp_path}",
    )

    assert report == {"staged": ["input", "software"], "source": "local"}
    assert len(sandbox.uploads) > 1
    with _uploaded_archive(sandbox) as archive:
        names = set(archive.getnames())
    assert "input/question.txt" in names
    assert "software/run.sh" in names
    assert not any(name.startswith("reference") for name in names)
    assert any("tar -xzf" in command for command in sandbox.commands)
    assert any(
        paths
        == [
            "/task-data/demo/hello/base/input",
            "/task-data/demo/hello/base/software",
            "/task-data/demo/hello/base/output",
            "/task-data/demo/hello/base/reference",
        ]
        for paths in sandbox.removed
    )
    assert sandbox.removed[-1][0].startswith("/task-data/demo/hello/base/.ale-local-stage-")


@pytest.mark.asyncio
async def test_static_cua_stages_fresh_data_and_hides_stale_reference(
    tmp_path: Path,
) -> None:
    _write_task_data(tmp_path)
    sandbox = _Sandbox(provider="static")

    report = await local_host.stage_input(
        sandbox,
        _task_data(),
        source=f"local:{tmp_path}",
    )

    assert report == {"staged": ["input", "software"], "source": "local"}
    assert sandbox.removed[0] == [
        "/task-data/demo/hello/base/input",
        "/task-data/demo/hello/base/software",
        "/task-data/demo/hello/base/output",
        "/task-data/demo/hello/base/reference",
    ]
    with _uploaded_archive(sandbox) as archive:
        names = set(archive.getnames())
    assert "input/question.txt" in names
    assert "software/run.sh" in names
    assert not any(name.startswith("reference") for name in names)


@pytest.mark.asyncio
async def test_proxmox_stages_reference_only_after_agent(tmp_path: Path) -> None:
    _write_task_data(tmp_path)
    sandbox = _Sandbox()

    report = await local_host.stage_reference(
        sandbox,
        _task_data(),
        source=f"local:{tmp_path}",
    )

    assert report == {"staged": ["reference"], "source": "local"}
    with _uploaded_archive(sandbox) as archive:
        names = set(archive.getnames())
    assert "reference/answer.txt" in names
    assert not any(name.startswith("input") for name in names)
    assert sandbox.removed[0] == ["/task-data/demo/hello/base/reference"]


@pytest.mark.asyncio
async def test_local_reference_takes_priority_over_baked_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_task_data(tmp_path)
    monkeypatch.delenv("ALE_REFERENCE_ARCHIVE_PASSWORD", raising=False)
    sandbox = _Sandbox(
        existing_paths={"/task-data/demo/hello/base/reference.7z"},
    )

    report = await local_host.stage_reference(
        sandbox,
        _task_data(),
        source=f"local:{tmp_path}",
    )

    assert report == {"staged": ["reference"], "source": "local"}
    assert not any("7z x" in command for command in sandbox.commands)


@pytest.mark.asyncio
async def test_missing_local_reference_falls_back_to_baked_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _write_task_data(tmp_path)
    (task / "reference" / "answer.txt").unlink()
    (task / "reference").rmdir()
    monkeypatch.setenv("ALE_REFERENCE_ARCHIVE_PASSWORD", "host-only-password")
    sandbox = _Sandbox(
        existing_paths={"/task-data/demo/hello/base/reference.7z"},
    )

    report = await local_host.stage_reference(
        sandbox,
        _task_data(),
        source=f"local:{tmp_path}",
    )

    assert report == {
        "staged": ["reference"],
        "source": "baked_in_sandbox",
        "decrypted_from": "reference.7z",
    }
    assert any("7z x" in command for command in sandbox.commands)
    assert any("host-only-password" in command for command in sandbox.commands)


@pytest.mark.asyncio
async def test_missing_local_and_baked_reference_skips_cleanly(tmp_path: Path) -> None:
    task = _write_task_data(tmp_path)
    (task / "reference" / "answer.txt").unlink()
    (task / "reference").rmdir()
    sandbox = _Sandbox()

    report = await local_host.stage_reference(
        sandbox,
        _task_data(),
        source=f"local:{tmp_path}",
    )

    assert report == {"skipped": True, "reason": "no_reference_7z"}


@pytest.mark.asyncio
async def test_proxmox_windows_reassembles_chunks_in_name_order(
    tmp_path: Path,
) -> None:
    _write_task_data(tmp_path)
    sandbox = _Sandbox(is_linux=False)

    await local_host.stage_input(
        sandbox,
        _task_data(),
        source=f"local:{tmp_path}",
    )

    encoded = sandbox.commands[0].rsplit(" ", 1)[-1]
    script = base64.b64decode(encoded).decode("utf-16-le")
    assert "Sort-Object Name" in script
    assert "tar.exe -xzf" in script


@pytest.mark.asyncio
async def test_local_source_preserves_docker_cp_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _write_task_data(tmp_path)
    sandbox = _Sandbox(provider="docker")
    copies: list[tuple[str, str, str]] = []

    async def fake_docker_cp(src: str, container: str, dst: str) -> None:
        copies.append((src, container, dst))

    monkeypatch.setattr(local_host, "_docker_cp", fake_docker_cp)

    report = await local_host.stage_input(
        sandbox,
        _task_data(),
        source=f"local:{tmp_path}",
    )

    assert report == {"staged": ["input", "software"], "source": "local"}
    assert copies == [
        (
            f"{task / 'input'}/.",
            "sandbox-123",
            "/task-data/demo/hello/base/input",
        ),
        (
            f"{task / 'software'}/.",
            "sandbox-123",
            "/task-data/demo/hello/base/software",
        ),
    ]
    assert sandbox.uploads == {}


@pytest.mark.asyncio
async def test_local_source_still_rejects_qemu(tmp_path: Path) -> None:
    _write_task_data(tmp_path)
    sandbox = _Sandbox(provider="qemu")

    with pytest.raises(NotImplementedError, match="Docker, Proxmox, and static"):
        await local_host.stage_input(
            sandbox,
            _task_data(),
            source=f"local:{tmp_path}",
        )
