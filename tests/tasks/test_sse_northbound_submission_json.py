import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tasks.business_finance.sse_northbound_programmatic_trading_01 import main


@pytest.mark.parametrize("submission", [b'{"evidence": "unescaped "quote""}', b"{}"])
async def test_bad_submission_is_zero_but_valid_submission_reaches_scorer(monkeypatch, submission):
    metadata = {"variant_name": "base"}
    metadata.update({name: name for name in (
        "output_file", "answer_key_file", "evaluator_documents_file", "document_manifest_file",
    )})
    files = dict.fromkeys(metadata.values(), b"{}")
    files["output_file"] = submission
    session = SimpleNamespace(
        file_exists=AsyncMock(return_value=True),
        directory_exists=AsyncMock(return_value=False),
        read_bytes=AsyncMock(side_effect=lambda path: files[path]),
    )
    scorer = MagicMock(return_value={"score": 1.0})
    monkeypatch.setattr(main, "score_submission", scorer)

    scores = await main.evaluate(SimpleNamespace(metadata=metadata), session)
    assert scores == ([1.0] if submission == b"{}" else [0.0])
    assert scorer.call_count == int(submission == b"{}")


async def test_invalid_reference_remains_an_evaluation_error(monkeypatch):
    metadata = {"variant_name": "base"}
    metadata.update({name: name for name in (
        "output_file", "answer_key_file", "evaluator_documents_file", "document_manifest_file",
    )})
    files = dict.fromkeys(metadata.values(), b"{}")
    files["answer_key_file"] = b"invalid reference JSON"
    session = SimpleNamespace(
        file_exists=AsyncMock(return_value=True),
        directory_exists=AsyncMock(return_value=False),
        read_bytes=AsyncMock(side_effect=lambda path: files[path]),
    )
    monkeypatch.setattr(main, "score_submission", MagicMock())

    with pytest.raises(json.JSONDecodeError):
        await main.evaluate(SimpleNamespace(metadata=metadata), session)
