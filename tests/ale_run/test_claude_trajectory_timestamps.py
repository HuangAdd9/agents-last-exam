from datetime import datetime

from ale_run.agents.claude_code.deployer import ClaudeCodeDeployer
from ale_run.base_interface import TrajectoryBuilder


def test_replay_retains_message_and_tool_result_times():
    builder = TrajectoryBuilder(agent_name="claude-code", task_path="test", variant_index=0)
    events = [
        {
            "type": "assistant",
            "timestamp": "2026-11-23T21:18:11.123Z",
            "message": {
                "content": [{"type": "tool_use", "id": "call1", "name": "Read", "input": {}}]
            },
        },
        {
            "type": "user",
            "timestamp": "2026-11-23T23:18:11.123Z",
            "message": {
                "content": [{"type": "tool_result", "tool_use_id": "call1", "content": "done"}]
            },
        },
    ]
    for event in events:
        ClaudeCodeDeployer._consume_event(event, builder)
    steps = builder.trajectory.steps
    assert [s.timestamp for s in steps] == [e["timestamp"] for e in events]
    assert steps[1].observation.results[0].tool_call_id == steps[0].tool_calls[0].id
    assert (
        datetime.fromisoformat(steps[1].timestamp) - datetime.fromisoformat(steps[0].timestamp)
    ).total_seconds() == 7200
    ClaudeCodeDeployer._consume_event({"type": "system", "timestamp": "later"}, builder)
    assert len(steps) == 2
    assert steps[1].timestamp == events[1]["timestamp"]


def test_missing_timestamp_keeps_builder_default():
    builder = TrajectoryBuilder(agent_name="claude-code", task_path="test", variant_index=0)
    ClaudeCodeDeployer._consume_event({"type": "assistant", "message": {"content": []}}, builder)
    assert datetime.fromisoformat(builder.trajectory.steps[0].timestamp).tzinfo is not None
