"""Command-line entry point for the audited Analog Active evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tasks.engineering.Analog_Active.eval_corrected import evaluate_output
from tasks.engineering.Analog_Active.main import VARIANTS, TaskConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--variant", choices=VARIANTS, default="Analog_Active_v1")
    parser.add_argument("--result-path", type=Path)
    args = parser.parse_args()

    result = evaluate_output(args.output_dir, TaskConfig(VARIANT_NAME=args.variant))
    result_text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.result_path:
        args.result_path.write_text(result_text, encoding="utf-8")
    print(result_text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
