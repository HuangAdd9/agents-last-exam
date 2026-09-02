from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from ale_run.agents.codex.config import CodexConfig
from ale_run.agents.codex.deployer import CodexDeployer
from ale_run.agents.codex.telemetry import (
    CodexOtelCollector,
    recover_telemetry_artifacts,
)


def _record(name: str, timestamp: str, **attributes: object) -> dict:
    items = [
        {"key": "event.name", "value": {"stringValue": name}},
        {"key": "event.timestamp", "value": {"stringValue": timestamp}},
    ]
    for key, value in attributes.items():
        if isinstance(value, bool):
            encoded = {"boolValue": value}
        elif isinstance(value, int):
            encoded = {"intValue": str(value)}
        else:
            encoded = {"stringValue": str(value)}
        items.append({"key": key, "value": encoded})
    return {
        "observedTimeUnixNano": "123",
        "severityNumber": 9,
        "severityText": "INFO",
        "attributes": items,
    }


def test_collector_persists_and_summarizes_otel_logs(tmp_path: Path) -> None:
    collector = CodexOtelCollector(tmp_path)
    collector.start()
    payload = {
        "resourceLogs": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": "codex_exec"},
                        }
                    ]
                },
                "scopeLogs": [
                    {
                        "scope": {"name": "codex_otel.log_only"},
                        "logRecords": [
                            _record(
                                "codex.api_request",
                                "2026-07-02T10:00:00.500Z",
                                duration_ms=500,
                                endpoint="/responses",
                                attempt=0,
                            ),
                            _record(
                                "codex.sse_event",
                                "2026-07-02T10:00:02.000Z",
                                **{"event.kind": "response.completed"},
                            ),
                            _record(
                                "codex.tool_result",
                                "2026-07-02T10:00:03.000Z",
                                tool_name="exec_command",
                                call_id="call_123",
                                duration_ms=250,
                                success=True,
                                arguments='{"cmd":"echo test"}',
                                output="test",
                            ),
                        ],
                    }
                ],
            }
        ]
    }
    request = urllib.request.Request(
        collector.endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 200
    collector.stop()

    raw_records = [
        json.loads(line) for line in collector.raw_path.read_text().splitlines()
    ]
    assert raw_records[0]["record_type"] == "otlp_request_chunk"
    assert raw_records[0]["chunk_count"] == 1

    events = [json.loads(line) for line in collector.events_path.read_text().splitlines()]
    assert len(events) == 3
    assert events[2]["attributes"]["arguments"] == '{"cmd":"echo test"}'
    assert events[2]["attributes"]["output"] == "test"

    summary = json.loads(collector.summary_path.read_text())
    assert summary["event_count"] == 3
    assert summary["api_calls"][0]["transport_duration_ms"] == 500
    assert summary["api_calls"][0]["response_duration_ms"] == 2000
    assert summary["tool_executions"][0]["duration_ms"] == 250
    assert summary["tool_executions"][0]["arguments"] == '{"cmd":"echo test"}'
    assert summary["tool_executions"][0]["output"] == "test"


def test_large_raw_request_is_chunked_and_recovers_derived_files(tmp_path: Path) -> None:
    large_output = "x" * (2 * 1024 * 1024)
    payload = {
        "resourceLogs": [
            {
                "scopeLogs": [
                    {
                        "logRecords": [
                            _record(
                                "codex.tool_result",
                                "2026-07-02T10:00:03.000Z",
                                tool_name="exec_command",
                                call_id="call_large",
                                duration_ms=250,
                                success=True,
                                output=large_output,
                            )
                        ]
                    }
                ]
            }
        ]
    }
    collector = CodexOtelCollector(tmp_path)
    collector.start()
    request = urllib.request.Request(
        collector.endpoint,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        assert response.status == 200
    collector.stop()

    raw_lines = collector.raw_path.read_bytes().splitlines()
    assert len(raw_lines) > 1
    assert max(map(len, raw_lines)) < 1024 * 1024

    collector.events_path.unlink()
    collector.summary_path.unlink()
    assert recover_telemetry_artifacts(tmp_path) == 1

    recovered = json.loads(collector.events_path.read_text().strip())
    assert recovered["attributes"]["output"] == large_output
    summary = json.loads(collector.summary_path.read_text())
    assert summary["event_count"] == 1
    assert summary["tool_executions"][0]["output"] == large_output


def test_codex_incrementally_mirrors_raw_telemetry_wal() -> None:
    assert "otel_requests.jsonl" in CodexDeployer.hot_artifacts
    assert "telemetry.jsonl" not in CodexDeployer.hot_artifacts


def test_codex_rate_limit_resume_is_enabled_globally() -> None:
    config = CodexConfig()
    assert config.rate_limit_resume_retries == 4
    assert config.rate_limit_backoff_base_s == 30.0
    assert config.rate_limit_backoff_max_s == 240.0


def test_codex_session_capture_copies_matching_thread(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    session_dir = codex_home / "sessions" / "2026" / "09" / "01"
    session_dir.mkdir(parents=True)
    thread_id = "01test-thread"
    session = session_dir / f"rollout-2026-09-01T00-00-00-{thread_id}.jsonl"
    session.write_text(
        '{"type":"response_item","payload":{"type":"reasoning",'
        '"encrypted_content":"opaque"}}\n',
        encoding="utf-8",
    )
    # An older/unrelated session must not be copied.
    (session_dir / "rollout-old-other-thread.jsonl").write_text("old\n")

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"type": "thread.started", "thread_id": thread_id}) + "\n",
        encoding="utf-8",
    )
    deployer = CodexDeployer(SimpleNamespace(config=None))
    deployer._codex_home = str(codex_home)
    deployer._capture_codex_session(tmp_path / "work", transcript)

    captured = tmp_path / "work" / "codex_session"
    assert (captured / "2026/09/01" / session.name).read_text() == session.read_text()
    assert not (captured / "2026/09/01" / "rollout-old-other-thread.jsonl").exists()
    manifest = json.loads((captured / "manifest.json").read_text())
    assert manifest["status"] == "captured"
    assert manifest["thread_id"] == thread_id


def test_codex_session_capture_matches_thread_in_metadata(tmp_path: Path) -> None:
    codex_home = tmp_path / "codex-home"
    session_dir = codex_home / "sessions" / "2026" / "09" / "01"
    session_dir.mkdir(parents=True)
    thread_id = "01metadata-thread"
    session = session_dir / "rollout-without-id.jsonl"
    session.write_text(
        json.dumps(
            {
                "type": "session_meta",
                "payload": {"id": thread_id},
            }
        )
        + "\n"
        + '{"type":"response_item","payload":{"type":"reasoning",'
        '"encrypted_content":"opaque-from-metadata-match"}}\n',
        encoding="utf-8",
    )

    transcript = tmp_path / "transcript.jsonl"
    transcript.write_text(
        json.dumps({"type": "thread.started", "thread_id": thread_id}) + "\n",
        encoding="utf-8",
    )
    deployer = CodexDeployer(SimpleNamespace(config=None))
    deployer._codex_home = str(codex_home)
    deployer._capture_codex_session(tmp_path / "work", transcript)

    captured = tmp_path / "work" / "codex_session" / "2026/09/01" / session.name
    assert captured.read_text(encoding="utf-8") == session.read_text(encoding="utf-8")
    manifest = json.loads(
        (tmp_path / "work/codex_session/manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "captured"
    assert manifest["files"] == ["2026/09/01/rollout-without-id.jsonl"]


@pytest.mark.asyncio
async def test_codex_resumes_same_session_after_rate_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work_dir = tmp_path / "work"
    (work_dir / ".git").mkdir(parents=True)
    thread_id = "01rate-limit-thread"
    invocations: list[tuple[list[str], str]] = []

    class FakeProcess:
        def __init__(self, argv, *, stdin, stdout, stderr, **kwargs) -> None:
            del stderr, kwargs
            invocation = len(invocations)
            invocations.append((list(argv), stdin.read().decode("utf-8")))
            self.pid = 1000 + invocation
            if invocation == 0:
                stdout.write(
                    (
                        json.dumps({"type": "thread.started", "thread_id": thread_id})
                        + "\n"
                        + json.dumps(
                            {
                                "type": "error",
                                "message": "exceeded retry limit, last status: "
                                "429 Too Many Requests",
                            }
                        )
                        + "\n"
                    ).encode()
                )
                self.returncode = 1
            else:
                stdout.write(b'{"type":"turn.completed","usage":{}}\n')
                self.returncode = 0
            stdout.flush()

        def poll(self) -> int:
            return self.returncode

    monkeypatch.setattr("ale_run.agents.codex.deployer.subprocess.Popen", FakeProcess)
    config = CodexConfig(
        provider="openrouter",
        otel_enabled=False,
        capture_session=False,
        rate_limit_resume_retries=2,
        rate_limit_backoff_base_s=0.001,
        rate_limit_backoff_max_s=0.001,
    )
    executor = SimpleNamespace(
        config=config,
        work_dir=str(work_dir),
        env={"OPENROUTER_API_KEY": "test-key"},
    )
    deployer = CodexDeployer(executor)
    deployer._codex_path = "/usr/bin/codex"

    result = await deployer.launch("complete the task")

    assert result.status == "completed"
    assert result.exit_code == 0
    assert len(invocations) == 2
    assert invocations[0][0][:2] == ["/usr/bin/codex", "exec"]
    assert invocations[0][1] == "complete the task"
    assert invocations[1][0][:3] == ["/usr/bin/codex", "exec", "resume"]
    assert invocations[1][0][-2:] == [thread_id, "-"]
    assert "Continue the original task" in invocations[1][1]


def test_recovery_ignores_incomplete_trailing_request(tmp_path: Path) -> None:
    collector = CodexOtelCollector(tmp_path)
    complete_payload = {
        "resourceLogs": [
            {
                "scopeLogs": [
                    {
                        "logRecords": [
                            _record(
                                "codex.turn_ttft",
                                "2026-07-02T10:00:00.000Z",
                                duration_ms=123,
                            )
                        ]
                    }
                ]
            }
        ]
    }
    collector._store_request({}, json.dumps(complete_payload).encode())
    collector._store_request({}, b'{"resourceLogs":"' + b"x" * (2 * 1024 * 1024))

    raw_lines = collector.raw_path.read_text().splitlines()
    collector.raw_path.write_text("\n".join(raw_lines[:-1]) + "\n")

    assert recover_telemetry_artifacts(tmp_path) == 1
    summary = json.loads(collector.summary_path.read_text())
    assert summary["event_count"] == 1
    assert summary["events_by_name"] == {"codex.turn_ttft": 1}
