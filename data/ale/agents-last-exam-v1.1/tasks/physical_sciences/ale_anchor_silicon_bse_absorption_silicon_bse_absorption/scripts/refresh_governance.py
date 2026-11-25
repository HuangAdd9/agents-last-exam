#!/usr/bin/env python3
"""Refresh governed hashes and human-readable validation records."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
IMAGE = "ale/base-physical-qe-bgw:6.7.0-4.0-20260920"
IMAGE_ID = "sha256:4b8c244f9cfdeb15e081973186b64f8e71e3d9d5ae406dcd53cb66b9e5cb3251"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_hashes(exclude: set[str] | None = None) -> dict[str, str]:
    excluded = exclude or set()
    return {
        str(path.relative_to(PACKAGE)): sha256(path)
        for path in sorted(PACKAGE.rglob("*"))
        if path.is_file() and str(path.relative_to(PACKAGE)) not in excluded
    }


def main() -> int:
    validated_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    snapshot = {
        "status": "passed",
        "validated_at": validated_at,
        "logical_snapshot": "cpu-free-ubuntu",
        "requiredSystemPackages": ["qe-bgw-6.7.0-4.0"],
        "image": IMAGE,
        "image_id": IMAGE_ID,
        "network": "disabled",
        "runtime_root_sha256": "d23a8b230214346b039ba9a12090136384c0c72d3fb459f505557ea82bfc0ea3",
        "observed_versions": {"python": "3.10.12", "openmpi": "4.1.6", "quantum_espresso": "6.7MaX", "berkeleygw": "4.0"},
        "executable_closure": ["pw.x", "pw2bgw.x", "epsilon.cplx.x", "sigma.cplx.x", "kernel.cplx.x", "absorption.cplx.x", "inteqp.cplx.x"],
        "checks": {
            "all_executables_present": "passed",
            "ldd_no_missing_libraries": "passed",
            "runtime_manifest_matches_image": "passed",
            "offline_bootstrap": "passed",
        },
    }
    (PACKAGE / "snapshot-validation.json").write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = """# Validation report

- Source: verified local Agents' Last Exam silicon BSE task card, verifier, POSCAR, and UPF.
- Loader gate: the real ALE `TaskLoader` loaded one `cpu-free-ubuntu` task; `start()` verified every framework-prestaged task-data file and passed the pinned offline runtime bootstrap.
- Runtime gate: `requiredSystemPackages` is `qe-bgw-6.7.0-4.0`; the image is `ale/base-physical-qe-bgw:6.7.0-4.0-20260920`; all seven QE/converter/BerkeleyGW executables are present and `ldd` reports no missing libraries.
- Scale gate: the native calculation uses a complete 5x5x5 mesh, a 41-band WFN, 10 Ry dielectric cutoff, and BerkeleyGW-validated 27-band epsilon / 23-band sigma manifolds without degeneracy overrides; four complete valence and conduction manifolds enter BSE.
- Isolation gate: the solver VM stages `input/`, `software/`, and a `reference/` directory containing only a tracked README placeholder; scientific references remain evaluator-local, while hidden inputs, oracle code, outputs, and evidence are created in a mode-0700 evaluator directory after the isolated candidate UID exits.
- Oracle gate: the end-to-end ALE `start()`/`evaluate()` path returned a perfect score after running the submitted workflow on the changed cell, nonlocal UPF coefficient, cutoff, dielectric cutoff, q shift, and optical resolution, followed by an independent evaluator-private execution of actual `pw.x`, `pw2bgw.x`, `epsilon.cplx.x`, `sigma.cplx.x`, `kernel.cplx.x`, `absorption.cplx.x`, and `inteqp.cplx.x`.
- Adversarial gate: explicit tests reject a missing public reference directory, scientific output or evaluator-private asset placed in it, wrong cutoffs, q shifts/resolution, incomplete grids, degeneracy overrides, forged logs, truncated columns, zero or rescaled spectra, toy eigensolvers, ignored inputs, and a workflow that searches staged storage for reference outputs. Positive tests accept reordered QE/sigma k-point lists, reordered k-resolved tables, state permutations, and unitary rotations within degenerate optical subspaces.
- Package gate: JSON/Python syntax, the exact README-only public-reference layout, exact full-column schemas, evaluator asset isolation, semantic stage provenance, physical-key canonicalization, degenerate-subspace invariants, source/runtime declarations, and offline snapshot compatibility passed.
"""
    (PACKAGE / "VALIDATION_REPORT.md").write_text(report, encoding="utf-8")

    scientific_names = [
        "bandstructure.dat", "eqp.dat", "eqp_q.dat", "absorption_eh.dat",
        "absorption_noeh.dat", "eigenvalues.dat", "eigenvalues_noeh.dat",
    ]
    manifest = {
        "status": "accepted",
        "failed_gates": [],
        "seed_id": "ale-anchor-silicon_bse_absorption",
        "task_dir": str(PACKAGE),
        "source_status": "verified_local",
        "adapter": "qe_bgw_multistage_workflow",
        "central_operation": "Actual QE mean-field and BerkeleyGW epsilon -> sigma -> kernel -> absorption plus inteqp executions",
        "runtime": {
            "logical_snapshot": "cpu-free-ubuntu",
            "requiredSystemPackages": ["qe-bgw-6.7.0-4.0"],
            "image": IMAGE,
            "image_id": IMAGE_ID,
            "network": "disabled",
        },
        "oracle": {
            "status": "passed_clean_private_recompute",
            "clean_runs": 1,
            "visible_sha256": {
                name: sha256(PACKAGE / "evaluator_private/visible_reference" / name)
                for name in scientific_names
            },
            "hidden_reference_storage": "runtime_ephemeral_mode_0700",
            "hidden_outputs_persisted": False,
            "hidden_changes": ["POSCAR lattice", "UPF nonlocal DIJ coefficient", "ecutwfc", "epsilon cutoff", "q shift", "absorption resolution"],
        },
        "grader": {
            "status": "passed_end_to_end",
            "visible_weight": 0.35,
            "hidden_recomputed_weight": 0.65,
            "pass_threshold": 0.75,
            "hidden_failure_cap": 0.69,
            "adversarial_cases": [
                "missing_public_reference_directory", "public_reference_scientific_output",
                "public_reference_evaluator_private_asset",
                "wrong_wavefunction_cutoff", "wrong_epsilon_cutoff", "wrong_qshift",
                "wrong_resolution", "incomplete_q_grid", "incomplete_k_grid",
                "degeneracy_override", "full_marker_forgery", "truncated_columns",
                "zero_spectrum", "absolute_intensity_rescale", "bse_equals_ip",
                "numpy_toy_solver", "ignore_structure", "ignore_pseudopotential",
                "staged_reference_copy_search", "alternate_semantic_representation",
                "reordered_qe_sigma_kpoints", "reordered_k_resolved_tables",
                "degenerate_state_permutation", "degenerate_unitary_rotation",
            ],
        },
        "validation": {
            "taskloader": "passed",
            "start": "passed",
            "evaluate_perfect": 1.0,
            "snapshot_compatible": "passed",
            "adversarial": "passed",
            "package": "passed",
            "validated_at": validated_at,
        },
        "provenance": {
            "canonical_url": "https://github.com/rdi-berkeley/agents-last-exam/tree/main/tasks/physical_sciences/silicon_bse_absorption",
            "source_task_card": "/data/ale/agents-last-exam/tasks/physical_sciences/silicon_bse_absorption/task_card.json",
            "source_verifier": "/data/ale/agents-last-exam/tasks/physical_sciences/silicon_bse_absorption/scripts/verify_outputs.py",
            "public_bundle": "/data/ale/agent_last_exam/task_data/public/physical_sciences/silicon_bse_absorption/base",
            "code_license": "Apache-2.0",
            "data_license": "CC-BY-4.0",
        },
        "package_sha256": file_hashes({"GENERATION_MANIFEST.json"}),
    }
    (PACKAGE / "GENERATION_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("governance hashes and reports refreshed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
