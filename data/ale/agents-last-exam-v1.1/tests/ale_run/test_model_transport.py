"""Managed provider startup failures must not provision or grade a candidate."""

import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from ale_run import cli, model_transport, rollout


@pytest.fixture
def managed(tmp_path):
    (tmp_path / "deployment.json").write_text(json.dumps({
        "service": "kimi-responses.service", "model": "Kimi-K3",
        "base_url": "http://127.0.0.1:8788/v1",
        "upstream": "http://upstream.invalid:5050/v1",
    }))
    return {"model_catalog_path": str(tmp_path / "models.json"),
            "base_url": "http://192.168.0.14:8788/v1", "model": "Kimi-K3",
            "api_key": "test-secret-must-not-appear"}


def test_upstream_timeout_preserves_gateway_success(managed, tmp_path):
    with patch.object(model_transport.socket, "create_connection") as connect:
        connect.side_effect = [connect.return_value, TimeoutError()]
        report = model_transport.check_managed_transport(managed, tmp_path)
    assert report["status"] == "unreachable"
    assert report["checks"][0]["connected"]
    assert not report["checks"][1]["connected"]
    assert "test-secret" not in json.dumps(report)


def test_healthy_transport_closes_connections(managed, tmp_path):
    with patch.object(model_transport.socket, "create_connection") as connect:
        report = model_transport.check_managed_transport(managed, tmp_path)
    assert report["status"] == "reachable"
    assert connect.return_value.__exit__.call_count == 2


def test_unmanaged_model_is_skipped(managed, tmp_path):
    managed["model"] = "another-model"
    with patch.object(model_transport.socket, "create_connection") as connect:
        assert model_transport.check_managed_transport(managed, tmp_path)["status"] == "not_applicable"
    connect.assert_not_called()


def test_actual_rollout_entry_fails_before_runner(managed, tmp_path, capsys):
    plan = SimpleNamespace(
        source_path=tmp_path / "rollout.yaml",
        experiment=SimpleNamespace(agents=[SimpleNamespace(config=managed)], name="test"),
        manifest={"run": {"output_root": str(tmp_path / "out")}},
    )
    args = argparse.Namespace(spec_path=plan.source_path, rollout_cmd="run",
                              dry_run=False, json_output=True)
    with patch.object(rollout, "load_rollout", return_value=plan), \
         patch.object(model_transport.socket, "create_connection", side_effect=TimeoutError()), \
         patch.object(rollout, "Runner") as runner:
        assert asyncio.run(cli._cmd_rollout(args)) == 2
    runner.assert_not_called()
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "not_executed"
    assert result["score"] is None
    reports = list((tmp_path / "out/test").glob("transport-preflight-*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["score"] is None
    assert not list(tmp_path.rglob("run.json"))


def test_validation_does_not_probe_or_run():
    args = argparse.Namespace(spec_path=Path("rollout.yaml"), rollout_cmd="validate",
                              dry_run=False, json_output=True)
    with patch.object(rollout, "load_rollout", return_value=SimpleNamespace(manifest={})), \
         patch.object(model_transport.socket, "create_connection") as connect, \
         patch.object(rollout, "run_rollout", new_callable=AsyncMock) as run:
        assert asyncio.run(cli._cmd_rollout(args)) == 0
    connect.assert_not_called()
    run.assert_not_called()
