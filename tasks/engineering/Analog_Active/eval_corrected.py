"""Audited Analog Active evaluator with LTspice-aware transient parsing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tasks.engineering.Analog_Active import eval as official_eval

_TIME_VALUE_RE = re.compile(
    r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*([munpf]?s?)?$",
    re.IGNORECASE,
)
_TIME_MULTIPLIERS = {
    "": 1.0,
    "s": 1.0,
    "m": 1e-3,
    "ms": 1e-3,
    "u": 1e-6,
    "us": 1e-6,
    "n": 1e-9,
    "ns": 1e-9,
    "p": 1e-12,
    "ps": 1e-12,
    "f": 1e-15,
    "fs": 1e-15,
}


def _parse_time_seconds(value: str) -> float | None:
    match = _TIME_VALUE_RE.fullmatch(value)
    if not match:
        return None
    return float(match.group(1)) * _TIME_MULTIPLIERS[(match.group(2) or "").lower()]


def check_tran_directive(asc_text: str, config: Any) -> bool:
    """Accept a 1 ms startup analysis in either compact or multi-argument form."""

    del config
    for raw_line in asc_text.splitlines():
        line = raw_line.strip()
        if not line.upper().startswith(("TEXT", ".TRAN")):
            continue

        match = re.search(r"\.tran\s+(.+)", line, re.IGNORECASE)
        if not match:
            continue

        tokens = match.group(1).split()
        lowered = [token.lower() for token in tokens]
        if "startup" not in lowered:
            continue

        startup_index = lowered.index("startup")
        time_values = [_parse_time_seconds(token) for token in tokens[:startup_index]]
        if not time_values or any(value is None for value in time_values):
            continue

        # LTspice accepts both `.tran Tstop startup` and the SPICE form
        # `.tran Tstep Tstop [Tstart [dTmax]] startup`.
        stop_time = time_values[0] if len(time_values) == 1 else time_values[1]
        if abs(stop_time - 1e-3) < 1e-4:
            return True

    return False


def evaluate_output(output_dir: str | Path, config: Any) -> dict[str, Any]:
    """Score a pulled Analog Active output directory with the corrected parser."""

    output = Path(output_dir)
    asc_path = output / config.CIRCUIT_FILE
    raw_path = output / config.RAW_FILE

    asc_text = asc_path.read_bytes().decode("latin-1") if asc_path.is_file() else ""
    raw_bytes = (
        raw_path.read_bytes() if raw_path.is_file() and raw_path.stat().st_size > 100 else None
    )

    checkpoints: dict[str, dict[str, Any]] = {
        "voltage_source": {
            "weight": 0.10,
            "passed": bool(asc_text and official_eval.check_voltage_source_value(asc_text, config)),
        },
        "feedback_resistor": {
            "weight": 0.10,
            "passed": bool(
                asc_text and official_eval.check_feedback_resistor_value(asc_text, config)
            ),
        },
        "load_resistor": {
            "weight": 0.10,
            "passed": bool(asc_text and official_eval.check_rload_value(asc_text, config)),
        },
        "capacitor_placement": {
            "weight": 0.10,
            "passed": bool(asc_text and official_eval.check_capacitor_placement(asc_text)),
        },
        "transient_and_raw": {
            "weight": 0.10,
            "passed": bool(asc_text and check_tran_directive(asc_text, config) and raw_bytes),
        },
    }

    measurements = {
        "output_voltage": {"value": None, "pass": False},
        "output_ripple": {"value_mv": None, "pass": False},
        "load_current": {"value": None, "pass": False},
    }
    if raw_bytes:
        measurements = official_eval.analyze_raw_output(raw_bytes, config)

    voltage_passed = bool(measurements["output_voltage"]["pass"])
    checkpoints.update(
        {
            "output_voltage": {"weight": 0.20, "passed": voltage_passed},
            "output_ripple": {
                "weight": 0.15,
                "passed": voltage_passed and bool(measurements["output_ripple"]["pass"]),
            },
            "load_current": {
                "weight": 0.15,
                "passed": voltage_passed and bool(measurements["load_current"]["pass"]),
            },
        }
    )

    score = sum(checkpoint["weight"] for checkpoint in checkpoints.values() if checkpoint["passed"])
    return {
        "evaluator": "analog_active_corrected",
        "score": round(score, 10),
        "checkpoints": checkpoints,
        "measurements": measurements,
    }
