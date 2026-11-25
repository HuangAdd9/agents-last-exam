"""Canonical snapshot-validation entry point for the adversarial suite."""
from pathlib import Path
import runpy


runpy.run_path(
    str(Path(__file__).with_name("validate_local.py")),
    run_name="__main__",
)
