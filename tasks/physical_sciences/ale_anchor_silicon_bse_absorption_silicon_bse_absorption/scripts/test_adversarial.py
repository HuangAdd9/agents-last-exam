#!/usr/bin/env python3
"""Adversarial and positive-equivalence checks for the scientific verifier."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import oracle

PACKAGE = Path(__file__).resolve().parents[1]
VISIBLE_REFERENCE = PACKAGE / "evaluator_private/visible_reference"
VISIBLE_EVIDENCE = PACKAGE / "evaluator_private/visible_evidence"


def clone(root: Path, name: str) -> tuple[Path, Path]:
    output, work = root / f"{name}_output", root / f"{name}_work"
    shutil.copytree(VISIBLE_REFERENCE, output)
    shutil.copy2(PACKAGE / "scripts/oracle_workflow.py", output / "workflow.py")
    shutil.copytree(VISIBLE_EVIDENCE, work)
    return output, work


def load_manifest(output: Path) -> dict:
    return json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))


def save_manifest(output: Path, manifest: dict) -> None:
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_output_hash(output: Path, name: str) -> None:
    manifest = load_manifest(output)
    manifest["outputs"][name] = oracle.sha256(output / name)
    save_manifest(output, manifest)


def update_stage_hash(output: Path, work: Path, stage: str, kind: str = "deck") -> None:
    manifest = load_manifest(output)
    record = next(item for item in manifest["stages"] if item["stage"] == stage)
    record[f"{kind}_sha256"] = oracle.sha256(work / record[kind])
    save_manifest(output, manifest)


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise AssertionError(f"expected one occurrence of {old!r} in {path}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def failures(output: Path, work: Path, case: Path, structure: Path, pseudo: Path) -> list[str]:
    return oracle.validate_run(output, work, case, structure, pseudo, VISIBLE_EVIDENCE)


def reverse_qe_kpoints(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    heading = next(index for index, line in enumerate(lines) if line.lower().startswith("k_points"))
    count = int(lines[heading + 1].split()[0])
    start = heading + 2
    lines[start:start + count] = reversed(lines[start:start + count])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reverse_bgw_block(path: Path, block: str) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    begin = lines.index(f"begin {block}")
    end = lines.index("end", begin)
    lines[begin + 1:end] = reversed(lines[begin + 1:end])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reverse_numeric_rows(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    comments = [line for line in lines if not line.strip() or line.lstrip().startswith(("#", "!"))]
    numeric = [line for line in lines if line.strip() and not line.lstrip().startswith(("#", "!"))]
    path.write_text("\n".join([*comments, *reversed(numeric)]) + "\n", encoding="utf-8")


def reverse_eqp_blocks(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    blocks = [lines[index:index + 9] for index in range(0, len(lines), 9)]
    if not blocks or any(len(block) != 9 for block in blocks):
        raise AssertionError(f"unexpected eqp block layout: {path}")
    path.write_text("\n".join(line for block in reversed(blocks) for line in block) + "\n", encoding="utf-8")


def rotate_complex_rows(path: Path, first: int, second: int, real_column: int, imaginary_column: int, strength_column: int) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    numeric_indices = [
        index for index, line in enumerate(lines)
        if line.strip() and not line.lstrip().startswith(("#", "!"))
    ]
    left = [float(value) for value in lines[numeric_indices[first]].split()]
    right = [float(value) for value in lines[numeric_indices[second]].split()]
    cosine, sine = 0.8, 0.6
    left_value = complex(left[real_column], left[imaginary_column])
    right_value = complex(right[real_column], right[imaginary_column])
    rotated = (cosine * left_value + sine * right_value, -sine * left_value + cosine * right_value)
    for row, value in ((left, rotated[0]), (right, rotated[1])):
        row[real_column] = value.real
        row[imaginary_column] = value.imag
        row[strength_column] = abs(value) ** 2
    lines[numeric_indices[first]] = " ".join(f"{value:.12g}" for value in left)
    lines[numeric_indices[second]] = " ".join(f"{value:.12g}" for value in right)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    case = PACKAGE / "input/workflow_case.json"
    structure = PACKAGE / "input/silicon.vasp"
    pseudo = PACKAGE / "input/Si.UPF"
    expected = VISIBLE_REFERENCE
    expected_work = VISIBLE_EVIDENCE

    with tempfile.TemporaryDirectory(prefix="ale_si_bse_adversarial_") as temporary:
        root = Path(temporary)
        baseline_output, baseline_work = clone(root, "baseline")
        baseline = oracle.score_run(
            baseline_output, baseline_work, case, structure, pseudo, expected, expected_work
        )
        assert baseline["score"] == 1.0 and not baseline["failures"], baseline

        output, work = clone(root, "skip_gw")
        manifest = load_manifest(output)
        manifest["stages"] = [item for item in manifest["stages"] if item["stage"] not in {"sigma", "kernel"}]
        save_manifest(output, manifest)
        assert failures(output, work, case, structure, pseudo)

        output, work = clone(root, "wrong_cutoff")
        replace_once(work / "scf.in", "ecutwfc=10.00000000", "ecutwfc = 9.0")
        update_stage_hash(output, work, "pw_scf")
        assert "deck-protocol:pw_scf" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "wrong_wfn_cutoff")
        replace_once(work / "wfn.in", "ecutwfc=10.00000000", "ecutwfc = 9.0")
        update_stage_hash(output, work, "pw_wfn")
        assert "deck-protocol:pw_wfn" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "wrong_epsilon_cutoff")
        replace_once(work / "epsilon.inp", "epsilon_cutoff 10.00000000", "epsilon_cutoff = 2.0")
        update_stage_hash(output, work, "epsilon")
        assert "deck-protocol:epsilon" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "wrong_screening_bands")
        replace_once(work / "epsilon.inp", "number_bands 27", "number_bands = 26")
        update_stage_hash(output, work, "epsilon")
        assert "deck-protocol:epsilon" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "wrong_qshift")
        wfnq = work / "wfnq.in"
        text = wfnq.read_text(encoding="utf-8")
        assert "0.002000000000" in text
        wfnq.write_text(text.replace("0.002000000000", "0.004000000000", 1), encoding="utf-8")
        update_stage_hash(output, work, "pw_wfnq")
        assert "deck-protocol:pw_wfnq" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "wrong_converter_qshift")
        replace_once(work / "wfnq.pp.in", "wfng_dk3=0.002", "wfng_dk3 = 0.004")
        update_stage_hash(output, work, "pw2bgw_wfnq")
        assert "deck-protocol:pw2bgw_wfnq" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "wrong_epsilon_qshift")
        replace_once(work / "epsilon.inp", "0.002000000000", "0.005000000000")
        update_stage_hash(output, work, "epsilon")
        assert "deck-protocol:epsilon" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "wrong_resolution")
        replace_once(work / "absorption.inp", "energy_resolution 0.15000000", "energy_resolution = 0.2")
        update_stage_hash(output, work, "absorption")
        assert "deck-protocol:absorption" in failures(output, work, case, structure, pseudo)

        for missing_flag in ("use_momentum", "lorentzian_broadening", "eqp_co_corrections"):
            output, work = clone(root, "missing_" + missing_flag)
            replace_once(work / "absorption.inp", missing_flag + "\n", "")
            update_stage_hash(output, work, "absorption")
            assert "deck-protocol:absorption" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "incomplete_q_grid")
        lines = (work / "epsilon.inp").read_text(encoding="utf-8").splitlines()
        del lines[-2]
        (work / "epsilon.inp").write_text("\n".join(lines) + "\n", encoding="utf-8")
        update_stage_hash(output, work, "epsilon")
        assert "deck-protocol:epsilon" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "incomplete_k_grid")
        lines = (work / "sigma.inp").read_text(encoding="utf-8").splitlines()
        del lines[-2]
        (work / "sigma.inp").write_text("\n".join(lines) + "\n", encoding="utf-8")
        update_stage_hash(output, work, "sigma")
        assert "deck-protocol:sigma" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "override")
        with (work / "epsilon.inp").open("a", encoding="utf-8") as handle:
            handle.write("degeneracy_check_override\n")
        update_stage_hash(output, work, "epsilon")
        assert "deck-protocol:epsilon" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "full_marker_forgery")
        forged = [
            "BerkeleyGW version 4.0",
            "BSE/Absorption code, Complex version",
            "Running MPI version (parallel)",
            "Compilation flags:",
            "Number of valence bands: 4",
            "Number of cond. bands: 4",
            "Number of k-points in the coarse k-grid: 125",
            "Broadening (eV): .150",
            "Using the momentum operator",
            "Started reading wavefunctions (WFN_fi)",
            "Reading quasiparticle energy corrections from eqp_co.dat",
            "Assembling spin-singlet kernel",
            "Diagonalization done",
            "- TOTAL : 1.0 s",
            "Job Done",
        ]
        forged.extend(f"plausible marker line {index}" for index in range(70))
        (work / "absorption.log").write_text("\n".join(forged) + "\n", encoding="utf-8")
        update_stage_hash(output, work, "absorption", "log")
        assert "stage-incomplete:absorption" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "no_eh")
        shutil.copy2(output / "absorption_noeh.dat", output / "absorption_eh.dat")
        update_output_hash(output, "absorption_eh.dat")
        assert "electron-hole-spectrum-equals-independent" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "toy_solver")
        with (output / "workflow.py").open("a", encoding="utf-8") as handle:
            handle.write("\n# forbidden substitute: np.linalg.eigh(model_hamiltonian)\n")
        assert "substitute-eigensolver" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "truncated_columns")
        lines = (output / "eigenvalues.dat").read_text(encoding="utf-8").splitlines()
        row = next(index for index, line in enumerate(lines) if line.strip() and not line.lstrip().startswith("#"))
        lines[row] = " ".join(lines[row].split()[:2])
        (output / "eigenvalues.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
        update_output_hash(output, "eigenvalues.dat")
        assert "table-schema:eigenvalues.dat" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "corrupt_dipole_component")
        lines = (output / "eigenvalues_noeh.dat").read_text(encoding="utf-8").splitlines()
        row = next(index for index, line in enumerate(lines) if line.strip() and not line.lstrip().startswith("#"))
        values = lines[row].split()
        values[8] = f"{float(values[8]) + 1.0:.12g}"
        lines[row] = " ".join(values)
        (output / "eigenvalues_noeh.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
        update_output_hash(output, "eigenvalues_noeh.dat")
        assert "dipole-schema:eigenvalues_noeh.dat" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "zero_spectrum")
        lines = (output / "absorption_eh.dat").read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if line.strip() and not line.lstrip().startswith("#"):
                values = line.split()
                values[1] = "0.0"
                lines[index] = " ".join(values)
        (output / "absorption_eh.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
        update_output_hash(output, "absorption_eh.dat")
        assert "zero-spectrum:absorption_eh.dat" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "absolute_scale")
        lines = (output / "absorption_eh.dat").read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if line.strip() and not line.lstrip().startswith("#"):
                values = line.split()
                values[1] = f"{float(values[1]) * 0.5:.12g}"
                lines[index] = " ".join(values)
        (output / "absorption_eh.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
        update_output_hash(output, "absorption_eh.dat")
        scaled = oracle.score_run(output, work, case, structure, pseudo, expected, expected_work)
        assert scaled["components"]["absorption_eh.dat"] < 0.8
        assert "numeric-mismatch:absorption_eh.dat" in scaled["failures"]

        output, work = clone(root, "ignored_table_column")
        lines = (output / "absorption_noeh.dat").read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            if line.strip() and not line.lstrip().startswith("#"):
                values = line.split()
                values[3] = f"{float(values[3]) + 5.0:.12g}"
                lines[index] = " ".join(values)
        (output / "absorption_noeh.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
        update_output_hash(output, "absorption_noeh.dat")
        changed_column = oracle.score_run(output, work, case, structure, pseudo, expected, expected_work)
        assert changed_column["components"]["absorption_noeh.dat"] < 0.8
        assert "numeric-mismatch:absorption_noeh.dat" in changed_column["failures"]

        output, work = clone(root, "reused_stage_evidence")
        manifest = load_manifest(output)
        scf = next(item for item in manifest["stages"] if item["stage"] == "pw_scf")
        wfn = next(item for item in manifest["stages"] if item["stage"] == "pw_wfn")
        wfn["log"] = scf["log"]
        wfn["log_sha256"] = scf["log_sha256"]
        save_manifest(output, manifest)
        assert "stage-evidence-not-unique" in failures(output, work, case, structure, pseudo)

        output, work = clone(root, "alternate_representation")
        epsilon = work / "epsilon.inp"
        text = epsilon.read_text(encoding="utf-8")
        text = text.replace("epsilon_cutoff 10.00000000", "  epsilon_cutoff = 1.0d1  ")
        text = text.replace("number_bands 27", "number_bands = 27")
        lines = text.splitlines()
        begin = lines.index("begin qpoints")
        end = lines.index("end", begin)
        small_q = lines.pop(begin + 1)
        lines.insert(end - 1, small_q)
        text = "\n".join(lines) + "\n"
        epsilon.write_text(text, encoding="utf-8")
        replace_once(work / "scf.in", "Si.UPF", "silicon_nc.UPF")
        update_stage_hash(output, work, "epsilon")
        update_stage_hash(output, work, "pw_scf")
        manifest = load_manifest(output)
        record = next(item for item in manifest["stages"] if item["stage"] == "pw_scf")
        old_log = work / record["log"]
        new_log = work / "ground_state_native.log"
        old_log.rename(new_log)
        record["log"] = new_log.name
        record["log_sha256"] = oracle.sha256(new_log)
        save_manifest(output, manifest)
        alternate = oracle.score_run(output, work, case, structure, pseudo, expected, expected_work)
        assert alternate["score"] == 1.0 and not alternate["failures"], alternate

        output, work = clone(root, "physical_equivalents")
        for stage, deck_name in (
            ("pw_scf", "scf.in"), ("pw_wfn", "wfn.in"), ("pw_wfnq", "wfnq.in")
        ):
            reverse_qe_kpoints(work / deck_name)
            update_stage_hash(output, work, stage)
        reverse_bgw_block(work / "sigma.inp", "kpoints")
        update_stage_hash(output, work, "sigma")
        reverse_numeric_rows(output / "bandstructure.dat")
        reverse_eqp_blocks(output / "eqp.dat")
        reverse_eqp_blocks(output / "eqp_q.dat")
        rotate_complex_rows(output / "eigenvalues.dat", 0, 1, 2, 3, 1)
        reverse_numeric_rows(output / "eigenvalues.dat")
        # Rows 0 and 4 have the same k point, valence energy, and degenerate
        # conduction energy.  A unitary rotation redistributes oscillator
        # strength while preserving the transition-subspace invariant.
        rotate_complex_rows(output / "eigenvalues_noeh.dat", 0, 4, 8, 9, 7)
        reverse_numeric_rows(output / "eigenvalues_noeh.dat")
        for name in (
            "bandstructure.dat", "eqp.dat", "eqp_q.dat",
            "eigenvalues.dat", "eigenvalues_noeh.dat",
        ):
            update_output_hash(output, name)
        equivalent = oracle.score_run(
            output, work, case, structure, pseudo, expected, expected_work
        )
        assert equivalent["score"] > 0.999999 and not equivalent["failures"], equivalent

        hidden_structure = root / "silicon_hidden.vasp"
        hidden_structure.write_bytes(structure.read_bytes().replace(b"2.715500000", b"2.688000000"))
        hidden_pseudo = root / "Si_hidden.UPF"
        hidden_pseudo.write_bytes(pseudo.read_bytes().replace(b"6.749115800389129E-001", b"7.749115800389129E-001", 1))
        output, work = clone(root, "ignore_inputs")
        manifest = load_manifest(output)
        manifest["input_sha256"]["structure"] = oracle.sha256(hidden_structure)
        manifest["input_sha256"]["pseudo"] = oracle.sha256(hidden_pseudo)
        save_manifest(output, manifest)
        hidden_failures = oracle.validate_run(output, work, case, hidden_structure, hidden_pseudo, expected_work)
        assert "structure-not-consumed" in hidden_failures or "deck-protocol:pw_scf" in hidden_failures

    print("adversarial checks passed: isolation-sensitive provenance, complete grids, full schemas, reordered k points, and degenerate-subspace invariants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
