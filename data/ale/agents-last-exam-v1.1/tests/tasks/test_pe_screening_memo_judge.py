from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tasks.business_finance.pe_screening_memo_1.scripts import score_screening_memo as scorer


def test_client_loads_evaluator_credentials_and_endpoint(monkeypatch):
    monkeypatch.setattr(scorer, "_client", None)
    monkeypatch.setattr(scorer, "load_eval_env", MagicMock())
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://judge.example/v1")
    constructor = MagicMock()
    monkeypatch.setattr(scorer, "OpenAI", constructor)

    assert scorer._get_client() is constructor.return_value
    assert scorer._get_client() is constructor.return_value
    scorer.load_eval_env.assert_called_once_with()
    constructor.assert_called_once_with(
        api_key="test-only-key",
        base_url="https://judge.example/v1",
        timeout=30,
        max_retries=3,
    )


@pytest.mark.parametrize("answer, expected", [("YES", True), ("no", False)])
def test_judge_uses_gpt54_parameters(monkeypatch, answer, expected):
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
    )
    monkeypatch.setattr(scorer, "_client", client)

    assert scorer._llm_judge("memo", "question") is expected
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "gpt-5.4"
    assert request["reasoning_effort"] == "none"
    assert request["max_completion_tokens"] == 32
    assert "max_tokens" not in request
    assert request["temperature"] == 0
    assert request["messages"][0]["content"] == scorer.SYSTEM_PROMPT


@pytest.mark.parametrize("answer", [None, "", "  "])
def test_empty_judge_response_is_not_a_negative_verdict(monkeypatch, answer):
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=answer))]
    )
    monkeypatch.setattr(scorer, "_client", client)

    with pytest.raises(RuntimeError, match="empty response"):
        scorer._llm_judge("memo", "question")
