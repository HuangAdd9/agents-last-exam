from __future__ import annotations

from types import SimpleNamespace

import pytest

from tasks.engineering.Analog_Active import eval as official_eval
from tasks.engineering.Analog_Active import eval_corrected

CONFIG = SimpleNamespace()


@pytest.mark.parametrize(
    "directive",
    [
        ".tran 1m startup",
        ".tran 1ms startup",
        ".tran 0.001 startup",
        ".tran 1e-3 startup",
        ".tran 0 1m 0 startup",
        ".tran 10u 1m 0 1u startup",
        "TEXT -160 736 Left 2 !.tran 0 1m 0 startup",
    ],
)
def test_corrected_parser_accepts_valid_one_millisecond_forms(directive: str) -> None:
    assert eval_corrected.check_tran_directive(directive, CONFIG)


@pytest.mark.parametrize(
    "directive",
    [
        ".tran 1m",
        ".tran 2m startup",
        ".tran 0 2m 0 startup",
        ".tran startup",
        ".tran invalid 1m startup",
    ],
)
def test_corrected_parser_rejects_wrong_or_incomplete_directives(directive: str) -> None:
    assert not eval_corrected.check_tran_directive(directive, CONFIG)


def test_official_parser_behavior_is_preserved() -> None:
    assert official_eval.check_tran_directive(".tran 1m startup", CONFIG)
    assert not official_eval.check_tran_directive(".tran 0 1m 0 startup", CONFIG)
    assert eval_corrected.check_tran_directive(".tran 0 1m 0 startup", CONFIG)
