"""GLM token budgets must reach the CLI inside every sandbox backend."""
from types import SimpleNamespace

import pytest

from ale_run.agents.claude_code.config import ClaudeCodeConfig
from ale_run.agents.claude_code.deployer import ClaudeCodeDeployer
from ale_run.orchestration.lifecycle import _collect_env_passthrough


CONTEXT = "CLAUDE_CODE_MAX_CONTEXT_TOKENS"
OUTPUT = "CLAUDE_CODE_MAX_OUTPUT_TOKENS"


@pytest.fixture
def deployer(monkeypatch: pytest.MonkeyPatch) -> ClaudeCodeDeployer:
    monkeypatch.delenv(CONTEXT, raising=False)
    monkeypatch.delenv(OUTPUT, raising=False)
    agent = object.__new__(ClaudeCodeDeployer)
    agent.executor = SimpleNamespace(env={})
    return agent


@pytest.mark.parametrize("model", ["glm-5.3", "zai/glm-5.3"])
def test_glm_defaults(deployer: ClaudeCodeDeployer, model: str) -> None:
    cfg = ClaudeCodeConfig(model=model, provider="direct", api_key="test-only")
    env = deployer._build_env(cfg)
    assert env[CONTEXT] == "1000000"
    assert env[OUTPUT] == "64000"


def test_explicit_limits_survive_sandbox_passthrough(
    deployer: ClaudeCodeDeployer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(CONTEXT, "500000")
    monkeypatch.setenv(OUTPUT, "48000")
    deployer.executor.env = _collect_env_passthrough()
    # Simulate the separate sandbox process, which has no host env.
    monkeypatch.delenv(CONTEXT)
    monkeypatch.delenv(OUTPUT)
    cfg = ClaudeCodeConfig(model="glm-5.3", provider="direct", api_key="test-only")
    env = deployer._build_env(cfg)
    assert env[CONTEXT] == "500000"
    assert env[OUTPUT] == "48000"


def test_other_models_unchanged(deployer: ClaudeCodeDeployer) -> None:
    cfg = ClaudeCodeConfig(model="Kimi-K3", provider="direct", api_key="test-only")
    env = deployer._build_env(cfg)
    assert CONTEXT not in env
    assert OUTPUT not in env


def test_config_output_limit_takes_precedence(deployer: ClaudeCodeDeployer) -> None:
    deployer.executor.env[OUTPUT] = "48000"
    cfg = ClaudeCodeConfig(
        model="glm-5.3", provider="direct", api_key="test-only", max_output_tokens=32000,
    )
    assert deployer._build_env(cfg)[OUTPUT] == "32000"
