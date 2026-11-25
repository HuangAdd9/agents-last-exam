#!/usr/bin/env python3
"""Consolidated deterministic checks for the silicon QE/BerkeleyGW package."""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import oracle

PACKAGE = Path(__file__).resolve().parents[1]
PUBLIC_REFERENCE_FILES = {"README.md"}


def validate_public_reference(reference: Path) -> None:
    """Require the tracked placeholder without exposing grading material."""
    assert reference.is_dir(), "missing_dir:reference"
    entries = {path.name for path in reference.iterdir()}
    assert entries == PUBLIC_REFERENCE_FILES, f"unsafe_reference_entries:{sorted(entries)}"
    readme = reference / "README.md"
    assert readme.is_file() and not readme.is_symlink(), "invalid_reference_placeholder"
    text = readme.read_text(encoding="utf-8").lower()
    assert "intentionally free of scientific outputs" in text
    assert "evaluator-private" in text


def exercise_public_reference_rejections() -> None:
    """Adversarially prove that answer files and private assets are rejected."""
    def expect_rejection(reference: Path, label: str) -> None:
        try:
            validate_public_reference(reference)
        except AssertionError:
            return
        raise AssertionError(f"public reference gate accepted {label}")

    with tempfile.TemporaryDirectory(prefix="ale_si_bse_reference_gate_") as temporary:
        reference = Path(temporary) / "reference"
        expect_rejection(reference, "a missing directory")
        reference.mkdir()
        shutil.copy2(PACKAGE / "reference/README.md", reference / "README.md")
        validate_public_reference(reference)

        scientific_output = reference / "absorption_eh.dat"
        scientific_output.write_text("forbidden grading output\n", encoding="utf-8")
        expect_rejection(reference, scientific_output.name)
        scientific_output.unlink()

        private_assets = reference / "evaluator_private"
        private_assets.mkdir()
        (private_assets / "oracle.py").write_text("# forbidden evaluator asset\n", encoding="utf-8")
        expect_rejection(reference, private_assets.name)
        shutil.rmtree(private_assets)


def main() -> int:
    card = json.loads((PACKAGE / "task_card.json").read_text(encoding="utf-8"))
    runtime = json.loads((PACKAGE / "input/runtime_manifest.json").read_text(encoding="utf-8"))
    assert card["vm"]["snapshot"] == "cpu-free-ubuntu"
    assert card["vm"]["machineType"] == "c4-standard-16"
    assert card["vm"]["vcpus"] == 16
    assert card["requiredSystemPackages"] == ["qe-bgw-6.7.0-4.0"]
    assert runtime["runtime_image"] == "ale/base-physical-qe-bgw:6.7.0-4.0-20260920"
    assert runtime["requiredSystemPackages"] == ["qe-bgw-6.7.0-4.0"]
    assert runtime["network"] == "disabled"
    case = json.loads((PACKAGE / "input/workflow_case.json").read_text(encoding="utf-8"))
    assert case["kgrid"] == 5 and case["bands"] >= 40
    assert case["epsilon_cutoff_ry"] >= 10.0
    assert case["screening_bands"] == 27 and case["sigma_bands"] == 23

    for path in sorted(PACKAGE.rglob("*.py")):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for path in sorted(PACKAGE.rglob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))
    for name in ("problem_spec.md", "workflow_case.json", "silicon.vasp", "Si.UPF", "runtime_manifest.json"):
        assert (PACKAGE / "input" / name).is_file()
    validate_public_reference(PACKAGE / "reference")
    exercise_public_reference_rejections()
    private_reference = PACKAGE / "evaluator_private/visible_reference"
    private_evidence = PACKAGE / "evaluator_private/visible_evidence"
    assert private_reference.is_dir() and private_evidence.is_dir()
    assert not (private_reference / "workflow.py").exists()
    assert {path.name for path in private_reference.iterdir()} == {
        *oracle.SCIENTIFIC_OUTPUTS, "run_manifest.json"
    }

    combined = "\n".join(
        path.read_text(encoding="utf-8", errors="replace").lower()
        for path in [PACKAGE / "main.py", PACKAGE / "input/problem_spec.md", PACKAGE / "scripts/oracle_workflow.py"]
    )
    assert "numpy.linalg.eigh" not in combined and "effective bse" not in combined
    assert "degeneracy_check_override" not in (PACKAGE / "scripts/oracle_workflow.py").read_text(encoding="utf-8")
    for executable in ("pw.x", "epsilon.cplx.x", "sigma.cplx.x", "kernel.cplx.x", "absorption.cplx.x", "inteqp.cplx.x"):
        assert executable in combined

    with tempfile.TemporaryDirectory(prefix="ale_si_bse_validate_") as temporary:
        root = Path(temporary)
        visible_output, visible_work = root / "visible_output", root / "visible_work"
        shutil.copytree(private_reference, visible_output)
        shutil.copy2(PACKAGE / "scripts/oracle_workflow.py", visible_output / "workflow.py")
        shutil.copytree(private_evidence, visible_work)
        visible = oracle.score_run(
            visible_output, visible_work,
            PACKAGE / "input/workflow_case.json", PACKAGE / "input/silicon.vasp",
            PACKAGE / "input/Si.UPF", private_reference, private_evidence,
        )
        assert visible["score"] == 1.0 and not visible["failures"], visible

    subprocess.run([sys.executable, "-B", "scripts/test_adversarial.py"], cwd=PACKAGE, check=True, timeout=180)
    print("package validation passed: benchmark-scale native stages, private oracle isolation, physical canonicalization, and adversarial coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
