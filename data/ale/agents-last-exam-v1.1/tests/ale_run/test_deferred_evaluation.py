from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from ale_run.base_interface import AgentRunResult, GatherReport
from ale_run.orchestration import lifecycle
from ale_run.orchestration.config_loader import load_experiment
from ale_run.orchestration.experiment_spec import AgentSpec, ArtifactsSpec, RunUnit
from ale_run.orchestration.runner import Runner
from ale_run.rollout import load_rollout


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,agent_status", [("inline", "completed"), ("deferred", "completed"),
                                             ("deferred", "failed"), ("deferred", "timeout")])
async def test_lifecycle_deferred_preserves_setup_output_trajectory_and_cleanup(tmp_path, monkeypatch, mode, agent_status):
    @dataclass
    class Config:
        model: str = "test"
        name: str = "test-agent"

    class Deployer:
        default_executor = "local"
        parse_count = 0

        @staticmethod
        def parse_artifacts(**kwargs):
            Deployer.parse_count += 1
            kwargs["builder"].add_step(source="agent", message="done")

    task_meta = {"description": "solve", "snapshot_name": "test", "os_type": "linux"}
    loader = SimpleNamespace(load=lambda _: task_meta)
    monkeypatch.setattr(lifecycle, "TaskLoader", lambda *_: loader)
    monkeypatch.setattr(lifecycle, "resolve_agent", lambda _: (Deployer, Config))
    monkeypatch.setattr(lifecycle, "build_config", lambda *_: Config())
    env = SimpleNamespace(sandbox=SimpleNamespace(id="test", metadata={}, os="linux"), session=object(),
                          current_phase=None, reset_async=AsyncMock(), reset_session=AsyncMock(),
                          close_async=AsyncMock())
    env.set_phase = lambda p: setattr(env, "current_phase", p)
    monkeypatch.setattr(lifecycle, "ALEEnv", lambda **_: env)
    driver = SimpleNamespace(setup=AsyncMock(), evaluate=AsyncMock(return_value={"score": 0.75}), close=AsyncMock())
    monkeypatch.setattr(lifecycle, "TaskDriver", lambda **_: driver)
    executor = SimpleNamespace(work_dir="/tmp/test", run_deployer=AsyncMock(return_value=AgentRunResult(
        status=agent_status, error=None if agent_status == "completed" else "agent failed")),
        stop_deployer=AsyncMock(),
        gather_dir=AsyncMock(return_value=GatherReport(transport="local")))
    monkeypatch.setattr(lifecycle, "_build_executor", lambda **_: executor)
    async def check_trace_before_output(**kwargs):
        run_dir = kwargs['run_dir']
        trace = json.loads((run_dir / 'trajectory.json').read_text())
        record = json.loads((run_dir / 'run.json').read_text())
        assert trace['final_metrics']['status'] == agent_status
        assert record['collection_status'] == 'pending'
        assert record['score'] is None
    inputs, outputs, references = AsyncMock(), AsyncMock(side_effect=check_trace_before_output), AsyncMock()
    monkeypatch.setattr(lifecycle, "_stage_task_data", inputs)
    monkeypatch.setattr(lifecycle, "pull_agent_output", outputs)
    monkeypatch.setattr(lifecycle, "stage_reference", references)
    agent = AgentSpec(id="test", class_="dummy", config={"model": "test"})
    result = await lifecycle.run_one_unit(
        unit=RunUnit(agent_id="test", agent_spec=agent, task_path="demo/test", variant_index=0),
        router=SimpleNamespace(provider_for=lambda _: object()), output_root=tmp_path,
        artifacts=ArtifactsSpec(output_path="local"), evaluation_mode=mode,
    )
    inputs.assert_awaited_once()
    driver.setup.assert_awaited_once()
    executor.run_deployer.assert_awaited_once()
    executor.stop_deployer.assert_awaited_once()
    outputs.assert_awaited_once()
    assert Deployer.parse_count == 1
    driver.close.assert_awaited_once()
    env.close_async.assert_awaited_once_with(mode="delete")
    record = json.loads((result.run_dir / "run.json").read_text())
    trajectory = json.loads((result.run_dir / "trajectory.json").read_text())
    assert record["status"] == agent_status
    assert record["evaluation_mode"] == mode
    assert trajectory["steps"]
    if mode == "deferred":
        references.assert_not_awaited()
        driver.evaluate.assert_not_awaited()
        assert result.score is None and result.eval_status == "deferred"
        assert record["evaluation_status"] == record["evaluation"]["status"] == "deferred"
    else:
        references.assert_awaited_once()
        driver.evaluate.assert_awaited_once()
        assert result.score == 0.75 and result.eval_status == "success"


@pytest.mark.parametrize("mode", ["inline", "deferred", "invalid"])
def test_rollout_deferred_mode_validation(tmp_path, monkeypatch, mode):
    import ale_run.rollout as module

    repo = tmp_path / "repo"
    task = repo / "tasks/demo/test"
    task.mkdir(parents=True)
    (repo / "ale_run").mkdir()
    (task / "main.py").write_text("")
    (task / "task_card.json").write_text("{}")
    env = tmp_path / "environment.yaml"
    env.write_text("provider: static\nendpoint: http://localhost:5000\nimage: ale-ubuntu22-docker\n")
    spec = tmp_path / "rollout.yaml"
    spec.write_text(yaml.safe_dump({"name": "test", "repo_root": str(repo), "task": {"id": "demo/test"},
                                   "environment": {"profile": str(env)}, "agent": {"harness": "dummy"},
                                   "run": {"evaluation_mode": mode}}))
    monkeypatch.setattr(module, "TaskLoader", lambda *_: SimpleNamespace(load=lambda _: {
        "snapshot_name": "test", "task_data": SimpleNamespace(variant_name="base", requires_task_data=False)}))
    monkeypatch.setattr(module, "_build_agent", lambda *_args, **_kwargs: (AgentSpec(id="test", class_="dummy"), {}, {}))
    if mode == "invalid":
        with pytest.raises(ValueError, match="evaluation_mode"):
            load_rollout(spec)
    else:
        plan = load_rollout(spec)
        assert plan.experiment.evaluation_mode == plan.manifest["run"]["evaluation_mode"] == mode


@pytest.mark.parametrize("mode", [None, "deferred", "invalid"])
def test_experiment_evaluation_mode(tmp_path, mode):
    (tmp_path / "agent.yaml").write_text("harness: dummy\nmodel: test\n")
    (tmp_path / "environment.yaml").write_text("provider: static\nendpoint: http://localhost:5000\n")
    spec = {"name": "test", "agent": "agent.yaml", "environment": "environment.yaml", "tasks": [{"path": "demo/test"}]}
    if mode is not None:
        spec["evaluation_mode"] = mode
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(spec))
    if mode == "invalid":
        with pytest.raises(ValueError, match="evaluation_mode"):
            load_experiment(path)
    else:
        assert load_experiment(path).evaluation_mode == (mode or "inline")


@pytest.mark.asyncio
async def test_runner_forwards_evaluation_mode(tmp_path, monkeypatch):
    from ale_run.orchestration.experiment_spec import (
        EnvironmentSpec,
        ExperimentSpec,
        OutputSpec,
        TaskSpec,
        UnitResult,
    )

    fake = AsyncMock(side_effect=lambda **kw: UnitResult(unit=kw["unit"], status="completed", eval_status="deferred"))
    monkeypatch.setattr(lifecycle, "run_one_unit", fake)
    spec = ExperimentSpec(name="test", output=OutputSpec(root=str(tmp_path)), environment=EnvironmentSpec(),
                          agents=[AgentSpec(id="test", class_="dummy")], tasks=[TaskSpec(path="demo/test")],
                          evaluation_mode="deferred")
    await Runner(spec).run()
    assert fake.call_args.kwargs["evaluation_mode"] == "deferred"
