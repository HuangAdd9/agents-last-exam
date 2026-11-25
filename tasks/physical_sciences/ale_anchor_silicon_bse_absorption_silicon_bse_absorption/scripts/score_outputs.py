#!/usr/bin/env python3
"""CLI wrapper around the independent QE/BerkeleyGW workflow verifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oracle import score_run


def main() -> int:
    parser = argparse.ArgumentParser()
    for name in ("output", "work", "case", "structure", "pseudo", "expected", "expected_work"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    result = score_run(
        args.output, args.work, args.case, args.structure, args.pseudo,
        args.expected, args.expected_work,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
