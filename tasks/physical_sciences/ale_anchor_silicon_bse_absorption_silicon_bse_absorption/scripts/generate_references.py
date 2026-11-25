#!/usr/bin/env python3
"""Generate the evaluator-local visible reference with the real workflow.

Hidden inputs and outputs are intentionally created only inside evaluate().
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
WORKFLOW_TIMEOUT_SECONDS = 3600


def run(case: Path, structure: Path, pseudo: Path, work: Path, output: Path) -> None:
    subprocess.run([
        "/usr/bin/python3", str(PACKAGE / "scripts/oracle_workflow.py"),
        "--case", str(case), "--structure", str(structure), "--pseudo", str(pseudo),
        "--work-dir", str(work), "--output-dir", str(output),
    ], check=True, timeout=WORKFLOW_TIMEOUT_SECONDS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    args = parser.parse_args()
    target = args.output_root.resolve()
    if target.exists():
        raise FileExistsError(f"target already exists: {target}")
    target.mkdir(parents=True)
    run(
        PACKAGE / "input/workflow_case.json",
        PACKAGE / "input/silicon.vasp",
        PACKAGE / "input/Si.UPF",
        target / "visible_evidence",
        target / "visible_reference",
    )
    (target / "visible_reference/workflow.py").unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
