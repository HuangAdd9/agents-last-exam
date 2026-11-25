import pytest

from ale_run.agents.claude_code.config import ClaudeCodeConfig
from ale_run.agents.claude_code.deployer import ClaudeCodeDeployer


@pytest.mark.parametrize(
    ("disabled", "extras", "expected"),
    [
        (False, (), None),
        (True, (), {"WebSearch", "WebFetch"}),
        (True, ("Write", "WebSearch"), {"Write", "WebSearch", "WebFetch"}),
        (False, ("Write", "Edit"), {"Write", "Edit"}),
    ],
)
def test_web_tool_policy_preserves_explicit_exclusions(disabled, extras, expected):
    cfg = ClaudeCodeConfig(disable_web_search=disabled, disabled_tools=extras)
    argv = ClaudeCodeDeployer._build_argv(
        claude_path="claude", cfg=cfg, mcp_config="cua.json",
    )
    if expected is None:
        assert "--disallowedTools" not in argv
    else:
        assert argv.count("--disallowedTools") == 1
        excluded = argv[argv.index("--disallowedTools") + 1].split(",")
        assert set(excluded) == expected
        assert len(excluded) == len(expected)
    assert cfg.disabled_tools == extras
