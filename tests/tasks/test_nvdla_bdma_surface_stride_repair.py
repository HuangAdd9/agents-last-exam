from __future__ import annotations

from types import SimpleNamespace

import pytest

from tasks.engineering.nvdla_bdma_surface_stride_repair import main


def task_metadata() -> dict[str, object]:
    return main.NvdlaBdmaConfig(REMOTE_ROOT_DIR="/task").to_metadata()


class SetupSession:
    def __init__(self, *, missing: set[str] | None = None):
        self.missing = missing or set()
        self.commands: list[str] = []

    async def file_exists(self, path: str) -> bool:
        return path not in self.missing

    async def run_command(self, command: str, *, check: bool = False) -> dict[str, object]:
        del check
        self.commands.append(command)
        return {"return_code": 0, "stdout": "", "stderr": ""}


class EvaluationSession:
    def __init__(self, *, submission_exists: bool = True):
        self.submission_exists = submission_exists
        self.commands: list[str] = []
        self.written_files: dict[str, str] = {}

    async def file_exists(self, path: str) -> bool:
        return path.endswith("grader.py")

    async def directory_exists(self, path: str) -> bool:
        metadata = task_metadata()
        if path == metadata["output_bdma_dir"]:
            return self.submission_exists
        return path.endswith("hidden_cases")

    async def run_command(self, command: str, *, check: bool = False) -> dict[str, object]:
        del check
        self.commands.append(command)
        if "test -f /dev/shm" in command:
            return {"return_code": 0, "stdout": "DONE\n", "stderr": ""}
        return {"return_code": 0, "stdout": "", "stderr": ""}

    async def write_file(self, path: str, content: str) -> None:
        self.written_files[path] = content

    async def read_file(self, path: str) -> str:
        if path.endswith("grader.stdout"):
            return '{"score": 0.75, "passed": false, "outcomes": {}}\n'
        if path.endswith("grader.stderr"):
            return ""
        raise AssertionError(f"unexpected read: {path}")


@pytest.mark.asyncio
async def test_setup_uses_native_container_without_solver_wrapper():
    metadata = task_metadata()
    session = SetupSession()

    await main.NvdlaBdmaSetup().setup(SimpleNamespace(metadata=metadata), session)

    command = session.commands[-1]
    assert "docker create" in command
    assert "--network none" in command
    assert main.TASK_CONTAINER in command
    assert "\\n" not in command
    assert "\ndocker create" in command
    assert "run_cases.sh" not in command
    assert "grader.py" not in command
    assert "hidden_cases" not in command


@pytest.mark.asyncio
async def test_setup_rejects_missing_public_input():
    metadata = task_metadata()
    missing = {
        f"{metadata['public_tests_dir']}/{main.PUBLIC_CASES[0]}/input.txn",
    }

    with pytest.raises(RuntimeError, match="task input is missing"):
        await main.NvdlaBdmaSetup().setup(
            SimpleNamespace(metadata=metadata),
            SetupSession(missing=missing),
        )


@pytest.mark.asyncio
async def test_missing_submission_scores_zero_after_container_cleanup():
    session = EvaluationSession(submission_exists=False)

    assert await main.evaluate(
        SimpleNamespace(metadata=task_metadata()),
        session,
    ) == [0.0]
    assert "docker rm -f" in session.commands[0]


@pytest.mark.asyncio
async def test_evaluation_runs_detached_clean_grader(monkeypatch):
    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(main.asyncio, "sleep", no_sleep)
    session = EvaluationSession()

    assert await main.evaluate(
        SimpleNamespace(metadata=task_metadata()),
        session,
    ) == [0.75]

    launch = next(command for command in session.commands if "systemd-run" in command)
    assert "--collect" in launch
    assert "--uid=user" in launch
    launcher = session.written_files[f"{main.EVAL_ROOT}/run_grader.sh"]
    assert "--keep-logs" in launcher
    assert "hidden_cases" in launcher
    assert "status=$?" in launcher
