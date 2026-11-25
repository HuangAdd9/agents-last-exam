from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import ale_run.rollout as rollout_module
from ale_run.cli import main
from ale_run.orchestration.experiment_spec import RunUnit, UnitResult
from ale_run.rollout import load_rollout

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "engineering/kicad_navswitch_library_integration_release_002"


def _write_environment(path: Path, *, provider: str = "static") -> None:
    if provider == "static":
        text = """\
provider: static
endpoint: http://127.0.0.1:5000
image: ale-win10
task_data_source: baked_in_sandbox
output_path: null
"""
    else:
        text = """\
provider: docker
image: ale-win10
task_data_source: baked_in_sandbox
output_path: null
"""
    path.write_text(text, encoding="utf-8")


def _write_rollout(
    path: Path,
    environment: Path,
    *,
    task_data_source: str | None = None,
    agent_extra: str = "",
) -> None:
    source_line = (
        f"  task_data_source: {task_data_source}\n" if task_data_source is not None else ""
    )
    path.write_text(
        f"""\
schema_version: 1
name: rollout-test
repo_root: {REPO_ROOT}
task:
  id: {TASK_ID}
  variant: 0
environment:
  profile: {environment}
{source_line}agent:
  harness: codex
  model: gpt-5.6-sol
  reasoning_effort: high
  api:
    provider: mindwaving
    base_url: https://gateway.example/v1
    api_key_env: ROLLOUT_TEST_KEY
{agent_extra}run:
  output_root: ./results
  cleanup_mode: delete
  max_attempts: 1
""",
        encoding="utf-8",
    )


def test_load_rollout_builds_one_native_unit_without_leaking_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "environment.yaml"
    rollout = tmp_path / "rollout.yaml"
    _write_environment(environment)
    _write_rollout(rollout, environment)
    monkeypatch.setenv("ROLLOUT_TEST_KEY", "very-secret-test-value")

    plan = load_rollout(rollout)

    assert plan.repo_root == REPO_ROOT
    assert plan.experiment.tasks[0].path == TASK_ID
    assert plan.experiment.tasks[0].variants == [0]
    assert plan.experiment.agents[0].class_ == "codex"
    assert plan.experiment.agents[0].config["model"] == "gpt-5.6-sol"
    assert plan.experiment.agents[0].config["reasoning_effort"] == "high"
    assert plan.experiment.agents[0].config["provider"] == "openrouter"
    assert plan.experiment.agents[0].config["api_key"] is None
    assert plan.auth_env == {"OPENROUTER_API_KEY": "very-secret-test-value"}
    assert "very-secret-test-value" not in json.dumps(plan.manifest)
    assert plan.manifest["agent"]["api"]["api_key_env"] == "ROLLOUT_TEST_KEY"
    assert plan.manifest["agent"]["api"]["provider"] == "mindwaving"
    assert "openrouter" not in json.dumps(plan.manifest)
    assert plan.experiment.artifacts.output_path == "local"
    assert plan.experiment.auto_resume is False


def test_load_rollout_routes_qizhi_through_responses_gateway_and_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "environment.yaml"
    rollout = tmp_path / "rollout.yaml"
    _write_environment(environment)
    _write_rollout(rollout, environment)
    rollout.write_text(
        rollout.read_text(encoding="utf-8")
        .replace("provider: mindwaving", "provider: qizhi")
        .replace(
            "    api_key_env: ROLLOUT_TEST_KEY\n",
            "    api_key_env: ROLLOUT_TEST_KEY\n"
            "    proxy_url: http://192.168.0.7:18443\n",
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ROLLOUT_TEST_KEY", "qizhi-secret-test-value")

    plan = load_rollout(rollout)

    config = plan.experiment.agents[0].config
    assert config["provider"] == "openrouter"
    assert config["base_url"] == "https://gateway.example/v1"
    assert config["proxy_url"] == "http://192.168.0.7:18443"
    assert plan.auth_env == {"OPENROUTER_API_KEY": "qizhi-secret-test-value"}
    assert plan.manifest["agent"]["api"] == {
        "provider": "qizhi",
        "base_url": "https://gateway.example/v1",
        "api_key_env": "ROLLOUT_TEST_KEY",
        "configured": True,
        "proxy_url": "http://192.168.0.7:18443",
    }
    assert "qizhi-secret-test-value" not in json.dumps(plan.manifest)


def test_load_rollout_accepts_mindwaving_bulk_as_a_distinct_gateway_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "environment.yaml"
    rollout = tmp_path / "rollout.yaml"
    _write_environment(environment)
    _write_rollout(rollout, environment)
    rollout.write_text(
        rollout.read_text(encoding="utf-8").replace(
            "provider: mindwaving", "provider: mindwaving_bulk"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ROLLOUT_TEST_KEY", "bulk-secret-test-value")

    plan = load_rollout(rollout)

    assert plan.experiment.agents[0].config["provider"] == "openrouter"
    assert plan.auth_env == {"OPENROUTER_API_KEY": "bulk-secret-test-value"}
    assert plan.manifest["agent"]["api"]["provider"] == "mindwaving_bulk"
    assert "bulk-secret-test-value" not in json.dumps(plan.manifest)


def test_load_rollout_preflights_local_task_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "environment.yaml"
    rollout = tmp_path / "rollout.yaml"
    task_data_root = tmp_path / "task-data"
    (task_data_root / TASK_ID / "base" / "input").mkdir(parents=True)
    _write_environment(environment, provider="docker")
    _write_rollout(rollout, environment, task_data_source=f"local:{task_data_root}")
    monkeypatch.setenv("ROLLOUT_TEST_KEY", "test-value")

    plan = load_rollout(rollout)

    assert plan.experiment.artifacts.task_data_source == f"local:{task_data_root}"
    assert plan.manifest["environment"]["provider"] == "docker"


def test_load_rollout_allows_local_task_data_for_static_cua(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "environment.yaml"
    rollout = tmp_path / "rollout.yaml"
    task_data_root = tmp_path / "task-data"
    (task_data_root / TASK_ID / "base" / "input").mkdir(parents=True)
    _write_environment(environment, provider="static")
    _write_rollout(rollout, environment, task_data_source=f"local:{task_data_root}")
    monkeypatch.setenv("ROLLOUT_TEST_KEY", "test-value")

    plan = load_rollout(rollout)

    assert plan.experiment.artifacts.task_data_source == f"local:{task_data_root}"
    assert plan.manifest["environment"]["provider"] == "static"


def test_load_rollout_rejects_missing_api_key_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "environment.yaml"
    rollout = tmp_path / "rollout.yaml"
    _write_environment(environment)
    _write_rollout(rollout, environment)
    monkeypatch.delenv("ROLLOUT_TEST_KEY", raising=False)

    with pytest.raises(KeyError, match="ROLLOUT_TEST_KEY"):
        load_rollout(rollout)


def test_load_rollout_supports_named_chatgpt_account_without_leaking_auth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "environment.yaml"
    rollout = tmp_path / "rollout.yaml"
    account_dir = tmp_path / ".codex-accounts" / "lidihan"
    account_dir.mkdir(parents=True)
    auth_content = json.dumps(
        {"auth_mode": "chatgpt", "tokens": {"access_token": "private-token"}}
    )
    (account_dir / "auth.json").write_text(auth_content, encoding="utf-8")
    _write_environment(environment)
    rollout.write_text(
        f"""\
schema_version: 1
name: chatgpt-account-test
repo_root: {REPO_ROOT}
task:
  id: {TASK_ID}
environment:
  profile: {environment}
agent:
  harness: codex
  model: gpt-5.6-sol
  reasoning_effort: high
  api:
    provider: chatgpt
    account: lidihan
run:
  output_root: ./results
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(rollout_module.Path, "home", classmethod(lambda cls: tmp_path))

    plan = load_rollout(rollout)

    assert plan.experiment.agents[0].config["provider"] == "chatgpt"
    assert plan.auth_env == {"CODEX_AUTH_JSON": auth_content}
    assert plan.manifest["agent"]["api"] == {
        "provider": "chatgpt",
        "account": "lidihan",
        "configured": True,
    }
    assert "private-token" not in json.dumps(plan.manifest)


def test_load_rollout_rejects_literal_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "environment.yaml"
    rollout = tmp_path / "rollout.yaml"
    _write_environment(environment)
    _write_rollout(
        rollout,
        environment,
        agent_extra="  config:\n    api_key: do-not-store-this\n",
    )
    monkeypatch.setenv("ROLLOUT_TEST_KEY", "test-value")

    with pytest.raises(ValueError, match="api_key_env"):
        load_rollout(rollout)


def test_load_rollout_rejects_task_path_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "environment.yaml"
    rollout = tmp_path / "rollout.yaml"
    _write_environment(environment)
    _write_rollout(rollout, environment)
    rollout.write_text(
        rollout.read_text(encoding="utf-8").replace(TASK_ID, "engineering/../secret"),
        encoding="utf-8",
    )
    monkeypatch.setenv("ROLLOUT_TEST_KEY", "test-value")

    with pytest.raises(ValueError, match="relative ALE path"):
        load_rollout(rollout)


def test_rollout_validate_cli_prints_secret_free_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    environment = tmp_path / "environment.yaml"
    rollout = tmp_path / "rollout.yaml"
    _write_environment(environment)
    _write_rollout(rollout, environment)
    monkeypatch.setenv("ROLLOUT_TEST_KEY", "cli-secret-value")

    exit_code = main(["rollout", "validate", str(rollout), "--json"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert json.loads(output)["task"]["id"] == TASK_ID
    assert "cli-secret-value" not in output


async def test_run_rollout_restores_process_state_and_writes_wrapper_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = tmp_path / "environment.yaml"
    rollout = tmp_path / "rollout.yaml"
    run_dir = tmp_path / "native-run"
    _write_environment(environment)
    _write_rollout(rollout, environment)
    monkeypatch.setenv("ROLLOUT_TEST_KEY", "run-secret-value")
    monkeypatch.setenv("OPENROUTER_API_KEY", "original-value")
    plan = load_rollout(rollout)
    original_cwd = Path.cwd()

    class FakeRunner:
        def __init__(self, spec):
            self.spec = spec

        def enumerate_units(self):
            return [
                RunUnit(
                    agent_id=self.spec.agents[0].id,
                    agent_spec=self.spec.agents[0],
                    task_path=self.spec.tasks[0].path,
                    variant_index=self.spec.tasks[0].variants[0],
                )
            ]

        async def run(self, units, *, max_attempts):
            assert Path.cwd() == REPO_ROOT
            assert os.environ["OPENROUTER_API_KEY"] == "run-secret-value"
            assert max_attempts == 1
            run_dir.mkdir()
            (run_dir / "run.json").write_text(
                json.dumps({"run_id": "native-run-id"}), encoding="utf-8"
            )
            return [
                UnitResult(
                    unit=units[0],
                    status="completed",
                    score=0.75,
                    eval_status="completed",
                    duration_s=12.5,
                    run_dir=run_dir,
                )
            ]

    monkeypatch.setattr(rollout_module, "Runner", FakeRunner)

    result, summary = await rollout_module.run_rollout(plan)

    assert result.status == "completed"
    assert summary["run_id"] == "native-run-id"
    assert Path.cwd() == original_cwd
    assert os.environ["OPENROUTER_API_KEY"] == "original-value"
    assert (run_dir / "rollout_manifest.json").is_file()
    assert json.loads((run_dir / "rollout_summary.json").read_text())["score"] == 0.75
    assert "run-secret-value" not in (run_dir / "rollout_manifest.json").read_text()
