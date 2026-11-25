#!/usr/bin/env python3
"""Independent semantic verifier for the silicon QE/BerkeleyGW workflow."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

OUTPUTS = (
    "bandstructure.dat", "eqp.dat", "eqp_q.dat", "absorption_eh.dat",
    "absorption_noeh.dat", "eigenvalues.dat", "eigenvalues_noeh.dat",
    "workflow.py", "run_manifest.json",
)
SCIENTIFIC_OUTPUTS = OUTPUTS[:-2]
STAGES = {
    "pw_scf": "pw.x", "pw_wfn": "pw.x", "pw2bgw_wfn": "pw2bgw.x",
    "pw_wfnq": "pw.x", "pw2bgw_wfnq": "pw2bgw.x", "epsilon": "epsilon.cplx.x",
    "sigma": "sigma.cplx.x", "kernel": "kernel.cplx.x",
    "absorption": "absorption.cplx.x", "inteqp": "inteqp.cplx.x",
}
TABLE_COLUMNS = {
    "bandstructure.dat": 8,
    "eqp.dat": 4,
    "eqp_q.dat": 4,
    "absorption_eh.dat": 4,
    "absorption_noeh.dat": 4,
    "eigenvalues.dat": 4,
    "eigenvalues_noeh.dat": 10,
}
RUNTIME_IMAGE = "ale/base-physical-qe-bgw:6.7.0-4.0-20260920"
ANGSTROM_TO_BOHR = 1.8897261254578281
FLOAT_TOL = 2.0e-7
DEGENERATE_TOL_EV = 2.0e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite_float(token: str) -> float:
    value = float(token.replace("d", "e").replace("D", "E"))
    if not math.isfinite(value):
        raise ValueError("non-finite number")
    return value


def numeric_rows(path: Path, exact_columns: int | None = None) -> list[list[float]]:
    """Read a native numeric table, rejecting rather than skipping malformed rows."""
    rows: list[list[float]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        stripped = raw.strip()
        if not stripped or stripped.startswith(("#", "!")):
            continue
        try:
            row = [_finite_float(token) for token in stripped.replace(",", " ").split()]
        except ValueError as exc:
            raise ValueError(f"nonnumeric row {line_number}") from exc
        if exact_columns is not None and len(row) != exact_columns:
            raise ValueError(f"row {line_number} has {len(row)} columns, expected {exact_columns}")
        rows.append(row)
    return rows


def _poscar(path: Path, scale: float) -> tuple[list[list[float]], list[list[float]], float]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 10:
        raise ValueError("truncated POSCAR")
    factor = _finite_float(lines[1]) * scale
    lattice = [[factor * _finite_float(value) for value in lines[row].split()[:3]] for row in range(2, 5)]
    counts = [int(value) for value in lines[6].split()]
    if sum(counts) != 2 or not lines[7].lower().startswith("d"):
        raise ValueError("expected two direct-coordinate atoms")
    atoms = [[_finite_float(value) for value in lines[row].split()[:3]] for row in range(8, 10)]
    a, b, c = lattice
    cross = (b[1] * c[2] - b[2] * c[1], b[2] * c[0] - b[0] * c[2], b[0] * c[1] - b[1] * c[0])
    volume = abs(sum(a[index] * cross[index] for index in range(3)))
    return lattice, atoms, volume


def _grid_points(grid: int, shift_z: float = 0.0) -> list[tuple[float, float, float]]:
    return [(i / grid, j / grid, k / grid + shift_z) for i in range(grid) for j in range(grid) for k in range(grid)]


def _close(left: float, right: float, tolerance: float = FLOAT_TOL) -> bool:
    return abs(left - right) <= tolerance * max(1.0, abs(left), abs(right))


def _same_vectors(left: list[list[float]], right: list[list[float]]) -> bool:
    return len(left) == len(right) and all(
        len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b)) for a, b in zip(left, right)
    )


def _point_counter(points: list[tuple[float, float, float]]) -> Counter:
    return Counter(tuple(round(value, 7) for value in point) for point in points)


def _strip_deck_lines(text: str) -> list[str]:
    lines = []
    for raw in text.splitlines():
        clean = raw.split("!", 1)[0].split("#", 1)[0].strip()
        if clean:
            lines.append(clean)
    return lines


def _qe_value(text: str, key: str) -> str | None:
    match = re.search(rf"(?im)\b{re.escape(key)}\s*=\s*([^,\n/]+)", text)
    return match.group(1).strip().strip("'\"") if match else None


def _qe_card(lines: list[str], heading: str, count: int) -> list[list[str]]:
    for index, line in enumerate(lines):
        if line.lower().startswith(heading.lower()):
            rows = lines[index + 1:index + 1 + count]
            if len(rows) != count:
                raise ValueError(f"truncated {heading}")
            return [row.split() for row in rows]
    raise ValueError(f"missing {heading}")


def _qe_kpoints(lines: list[str]) -> list[tuple[float, float, float]]:
    for index, line in enumerate(lines):
        if line.lower().startswith("k_points"):
            if "crystal" not in line.lower():
                raise ValueError("K_POINTS must use crystal coordinates")
            count = int(lines[index + 1].split()[0])
            rows = lines[index + 2:index + 2 + count]
            if len(rows) != count:
                raise ValueError("truncated K_POINTS")
            return [tuple(_finite_float(value) for value in row.split()[:3]) for row in rows]
    raise ValueError("missing K_POINTS")


def _parse_qe_deck(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = _strip_deck_lines(text)
    cell = [[_finite_float(value) for value in row[:3]] for row in _qe_card(lines, "cell_parameters", 3)]
    atom_rows = _qe_card(lines, "atomic_positions", 2)
    species = _qe_card(lines, "atomic_species", 1)[0]
    return {
        "calculation": (_qe_value(text, "calculation") or "").lower(),
        "ecutwfc": _finite_float(_qe_value(text, "ecutwfc") or "nan"),
        "nbnd": int(_finite_float(_qe_value(text, "nbnd") or "nan")),
        "nosym": (_qe_value(text, "nosym") or "").lower() in {".true.", "true", "t"},
        "noinv": (_qe_value(text, "noinv") or "").lower() in {".true.", "true", "t"},
        "cell": cell,
        "atoms": [[_finite_float(value) for value in row[1:4]] for row in atom_rows],
        "pseudo_name": Path(species[2]).name if len(species) >= 3 else "",
        "kpoints": _qe_kpoints(lines),
    }


def _parse_bgw_deck(path: Path) -> tuple[dict[str, list[str]], set[str], dict[str, list[list[float]]]]:
    lines = _strip_deck_lines(path.read_text(encoding="utf-8", errors="replace"))
    values: dict[str, list[str]] = {}
    flags: set[str] = set()
    blocks: dict[str, list[list[float]]] = {}
    index = 0
    while index < len(lines):
        tokens = lines[index].replace("=", " ").replace(",", " ").split()
        key = tokens[0].lower()
        if key == "begin":
            if len(tokens) != 2:
                raise ValueError("malformed begin block")
            name = tokens[1].lower()
            index += 1
            rows: list[list[float]] = []
            while index < len(lines) and lines[index].split()[0].lower() != "end":
                rows.append([_finite_float(token) for token in lines[index].replace(",", " ").split()])
                index += 1
            if index >= len(lines):
                raise ValueError(f"unterminated {name} block")
            blocks[name] = rows
        elif len(tokens) == 1:
            flags.add(key)
        else:
            values[key] = tokens[1:]
        index += 1
    return values, flags, blocks


def _bgw_number(values: dict[str, list[str]], key: str) -> float:
    if key not in values or len(values[key]) != 1:
        raise ValueError(f"missing scalar {key}")
    return _finite_float(values[key][0])


def _validate_qe_protocol(stage: str, deck: Path, case: dict, structure: Path) -> list[str]:
    try:
        parsed = _parse_qe_deck(deck)
        lattice, atoms, _ = _poscar(structure, float(case["lattice_scale"]))
        shift = float(case["qshift"]) if stage == "pw_wfnq" else 0.0
        expected_calculation = "scf" if stage == "pw_scf" else "bands"
        valid = (
            parsed["calculation"] == expected_calculation
            and _close(parsed["ecutwfc"], float(case["ecutwfc_ry"]))
            and parsed["nbnd"] == int(case["bands"])
            and parsed["nosym"] and parsed["noinv"]
            and parsed["pseudo_name"].lower().endswith(".upf")
            and _same_vectors(parsed["cell"], lattice)
            and _same_vectors(parsed["atoms"], atoms)
            and _point_counter(parsed["kpoints"]) == _point_counter(_grid_points(int(case["kgrid"]), shift))
        )
    except Exception:
        valid = False
    return [] if valid else ["deck-protocol:" + stage]


def _validate_converter(stage: str, deck: Path, case: dict) -> list[str]:
    text = deck.read_text(encoding="utf-8", errors="replace")
    grid = int(case["kgrid"])
    shift = float(case["qshift"]) if stage == "pw2bgw_wfnq" else 0.0
    required = {"wfng_flag": ".true.", "wfng_nk1": str(grid), "wfng_nk2": str(grid), "wfng_nk3": str(grid)}
    valid = all((_qe_value(text, key) or "").lower() == value for key, value in required.items())
    try:
        valid = valid and all(_close(_finite_float(_qe_value(text, key) or "nan"), value) for key, value in (
            ("wfng_dk1", 0.0), ("wfng_dk2", 0.0), ("wfng_dk3", shift)
        ))
    except ValueError:
        valid = False
    if stage == "pw2bgw_wfn":
        valid = valid and all((_qe_value(text, key) or "").lower() == ".true." for key in ("rhog_flag", "vxc_flag"))
    return [] if valid else ["deck-protocol:" + stage]


def _validate_bgw_protocol(stage: str, deck: Path, case: dict) -> list[str]:
    try:
        values, flags, blocks = _parse_bgw_deck(deck)
        grid = int(case["kgrid"])
        valid = "degeneracy_check_override" not in flags
        if stage == "epsilon":
            rows = blocks.get("qpoints", [])
            grid_q = [point for point in _grid_points(grid) if point != (0.0, 0.0, 0.0)]
            valid = valid and _close(_bgw_number(values, "epsilon_cutoff"), float(case["epsilon_cutoff_ry"]))
            valid = valid and int(_bgw_number(values, "number_bands")) == int(case["screening_bands"])
            valid = valid and len(rows) == grid ** 3 and all(len(row) == 5 for row in rows)
            head_rows = [row for row in rows if _close(row[4], 1.0)]
            body_rows = [row for row in rows if _close(row[4], 0.0)]
            valid = valid and len(head_rows) == 1 and len(body_rows) == grid ** 3 - 1
            valid = valid and _point_counter([tuple(row[:3]) for row in head_rows]) == _point_counter([
                (0.0, 0.0, float(case["qshift"]))
            ])
            valid = valid and _point_counter([tuple(row[:3]) for row in body_rows]) == _point_counter(grid_q)
        elif stage == "sigma":
            rows = blocks.get("kpoints", [])
            valid = valid and {"screening_semiconductor", "no_symmetries_q_grid"} <= flags
            valid = valid and int(_bgw_number(values, "number_bands")) == int(case["sigma_bands"])
            valid = valid and int(_bgw_number(values, "number_sigma_pools")) == 8
            valid = valid and int(_bgw_number(values, "band_index_min")) == 1
            valid = valid and int(_bgw_number(values, "band_index_max")) == 8
            valid = valid and len(rows) == grid ** 3 and all(len(row) == 4 for row in rows)
            valid = valid and _point_counter([tuple(row[:3]) for row in rows]) == _point_counter(_grid_points(grid))
        elif stage == "kernel":
            valid = valid and {"no_symmetries_coarse_grid", "screening_semiconductor"} <= flags
            valid = valid and int(_bgw_number(values, "number_val_bands")) == 4
            valid = valid and int(_bgw_number(values, "number_cond_bands")) == 4
        elif stage in {"absorption", "inteqp"}:
            required = {"no_symmetries_coarse_grid", "no_symmetries_fine_grid", "use_momentum"}
            if stage == "absorption":
                required |= {"no_symmetries_shifted_grid", "screening_semiconductor", "lorentzian_broadening", "eqp_co_corrections"}
                valid = valid and _close(_bgw_number(values, "energy_resolution"), float(case["energy_resolution_ev"]))
            valid = valid and required <= flags
            for suffix in ("coarse", "fine"):
                valid = valid and int(_bgw_number(values, "number_val_bands_" + suffix)) == 4
                valid = valid and int(_bgw_number(values, "number_cond_bands_" + suffix)) == 4
        else:
            valid = False
    except Exception:
        valid = False
    return [] if valid else ["deck-protocol:" + stage]


def _extract(pattern: str, text: str) -> float | None:
    match = re.search(pattern, text, re.IGNORECASE)
    return _finite_float(match.group(1)) if match else None


def _native_log(stage: str, text: str, deck_name: str, case: dict) -> bool:
    lowered = text.lower()
    if (
        "error:" in lowered
        or "aborted" in lowered
        or "breaks degenerate subspace" in lowered
        or len(text.splitlines()) < 40
    ):
        return False
    if stage.startswith("pw_") or stage.startswith("pw2bgw"):
        program = "pwscf" if not stage.startswith("pw2bgw") else "pw2bgw"
        valid = (
            re.search(rf"Program\s+{program}\s+v\.6\.7MaX", text, re.IGNORECASE) is not None
            and re.search(
                rf"\b{program}\s*:\s*(?:[0-9]+m)?[0-9.]+s CPU",
                text,
                re.IGNORECASE,
            ) is not None
            and "job done" in lowered
        )
        if stage.startswith("pw2bgw"):
            valid = valid and "done write_wfng" in lowered
            if stage == "pw2bgw_wfn":
                valid = valid and "done write_rhog" in lowered and "done write_vxc_g" in lowered
        else:
            valid = valid and re.search(rf"Reading input from\s+{re.escape(deck_name)}\b", text, re.IGNORECASE) is not None
        return valid
    title = {
        "epsilon": "Epsilon code, Complex version", "sigma": "Sigma code, Complex version",
        "kernel": "BSE/Kernel code, Complex version", "absorption": "BSE/Absorption code, Complex version",
        "inteqp": "BSE/IntEqp code, Complex version",
    }[stage]
    valid = (
        "berkeleygw version 4.0" in lowered
        and title.lower() in lowered
        and "running mpi version" in lowered
        and re.search(r"(?:-\s+)?TOTAL(?:\s*:|\s{2,})", text) is not None
    )
    if stage != "epsilon":
        valid = valid and "compilation flags:" in lowered
    if stage != "kernel":
        valid = valid and "job done" in lowered
    numeric_patterns = {
        "epsilon": (r"Total number of bands:\s*(\d+)", int(case["screening_bands"]), r"Number of q-points:\s*(\d+)", int(case["kgrid"]) ** 3),
        "sigma": (r"Total number of bands in the calculation:\s*(\d+)", int(case["sigma_bands"]), r"Number of bands to compute diagonal self-energy matrix elements:\s*(\d+)", 8),
        "kernel": (r"Number of valence bands:\s*(\d+)", 4, r"Number of conduction bands:\s*(\d+)", 4),
        "absorption": (r"Number of valence bands:\s*(\d+)", 4, r"Number of cond\. bands:\s*(\d+)", 4),
        "inteqp": (r"Number of valence bands\s*=\s*(\d+)", 4, r"Number of cond\. bands\s*=\s*(\d+)", 4),
    }
    first_pattern, first_expected, second_pattern, second_expected = numeric_patterns[stage]
    first = _extract(first_pattern, text)
    second = _extract(second_pattern, text)
    valid = valid and first == first_expected and second == second_expected

    # Completion banners alone are forgeable.  Require independent, ordered
    # milestones emitted while each native executable reads its binary inputs
    # and performs the stage's central operation.
    signatures = {
        "epsilon": (
            "computing the static inverse dielectric matrix",
            "independent matrix elements of chi",
            "writing dielectric matrix to file",
        ),
        "sigma": (
            "computing ch as a partial sum over empty bands",
            "reading exchange-correlation matrix elements from vxc.dat file",
            "reading header of wfn_inner",
            "calculation parameters:",
        ),
        "kernel": (
            "calculating the kernel only for",
            "started reading wavefunctions (wfn_co)",
            "epsilon matrix for q->0 read from eps0mat",
            "started calculation of matrix elements",
        ),
        "absorption": (
            "using the momentum operator",
            "started reading wavefunctions (wfn_fi)",
            "reading quasiparticle energy corrections from eqp_co.dat",
            "assembling spin-singlet kernel",
            "diagonalization done",
        ),
        "inteqp": (
            "started reading wavefunctions (wfn_fi)",
            "reading quasiparticle energy corrections from eqp_co.dat",
            "started wavefunction interpolation",
            "finished wavefunction interpolation",
        ),
    }
    positions = [lowered.find(marker) for marker in signatures[stage]]
    valid = valid and all(position >= 0 for position in positions)
    valid = valid and positions == sorted(positions)

    grid_count = int(case["kgrid"]) ** 3
    if stage == "epsilon":
        cutoff = _extract(r"Screened Coulomb cutoff:\s*([0-9.eEdD+-]+)\s*Ry", text)
        valid = valid and cutoff is not None and _close(cutoff, float(case["epsilon_cutoff_ry"]), 1.0e-5)
        # A real epsilon run emits one completed matrix build for each stored
        # q point (the q->0 head is handled separately).  This trace makes a
        # pasted banner plus a list of expected phrases insufficient.
        valid = valid and lowered.count("finished writing matrix") >= grid_count - 1
        valid = valid and lowered.count("finished building polarizability matrix") >= grid_count - 2
    elif stage == "sigma":
        log_kpoints = _extract(r"Number of k-points in WFN_inner:\s*(\d+)", text)
        valid = valid and log_kpoints == grid_count
        valid = valid and lowered.count("finished calculating sigma") == grid_count
    elif stage == "kernel":
        states = _extract(r"Started reading wavefunctions \(WFN_co\) with\s*(\d+)\s*state", text)
        records = _extract(r"Started writing out BSE matrices with\s*(\d+)\s*records", text)
        valid = valid and states == grid_count * 8
        valid = valid and records == grid_count * 4 * 4 * 4
        valid = valid and lowered.count("finished writing out bse matrices") == 1
    elif stage == "absorption":
        log_kpoints = _extract(r"Number of k-points in the coarse k-grid:\s*(\d+)", text)
        broadening = _extract(r"Broadening \(eV\):\s*([0-9.eEdD+-]+)", text)
        eigenvalues = _extract(r"Number of eigenvalues to be computed:\s*(\d+)", text)
        written = _extract(r"Writing\s*(\d+)\s*eigenvectors", text)
        valid = valid and log_kpoints == grid_count
        valid = valid and broadening is not None and _close(
            broadening, float(case["energy_resolution_ev"]), 5.0e-4
        )
        valid = valid and eigenvalues == grid_count * 16 and written == grid_count * 16
        valid = valid and lowered.count("finished reading wavefunctions (wfn_fi)") == 2
    elif stage == "inteqp":
        log_kpoints = _extract(r"Number of k-points in the coarse k-grid:\s*(\d+)", text)
        valid = valid and log_kpoints == grid_count
        valid = valid and lowered.count("finished reading wavefunctions (wfn_fi)") == 2
    return valid


def _manifest_stages(manifest: dict) -> tuple[dict[str, dict], list[str]]:
    stages = manifest.get("stages")
    if not isinstance(stages, list) or len(stages) != len(STAGES):
        return {}, ["stage-count"]
    records = [record for record in stages if isinstance(record, dict)]
    by_stage = {record.get("stage"): record for record in records}
    failures = [] if len(records) == len(STAGES) and set(by_stage) == set(STAGES) else ["stage-set"]
    decks = [record.get("deck") for record in records]
    logs = [record.get("log") for record in records]
    if len(set(decks)) != len(STAGES) or len(set(logs)) != len(STAGES):
        failures.append("stage-evidence-not-unique")
    return by_stage, failures


def _expected_scf_md5(expected_work: Path | None) -> str | None:
    if expected_work is None or not expected_work.is_dir():
        return None
    matches = []
    for path in expected_work.iterdir():
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if (
            "Program PWSCF" not in text
            or "unit-cell volume" not in text
            or "End of self-consistent calculation" not in text
        ):
            continue
        match = re.search(r"MD5 check sum:\s*([0-9a-f]+)", text, re.IGNORECASE)
        if match:
            matches.append(match.group(1).lower())
    return matches[0] if len(matches) == 1 else None


def _validate_table_schemas(output: Path, case: dict) -> tuple[dict[str, list[list[float]]], list[str]]:
    tables: dict[str, list[list[float]]] = {}
    failures: list[str] = []
    for name, columns in TABLE_COLUMNS.items():
        try:
            tables[name] = numeric_rows(output / name, columns)
        except Exception:
            failures.append("table-schema:" + name)
    if failures:
        return tables, failures
    grid_count = int(case["kgrid"]) ** 3
    expected_rows = {
        "bandstructure.dat": grid_count * 8,
        "eigenvalues.dat": grid_count * 16,
        "eigenvalues_noeh.dat": grid_count * 16,
    }
    for name, count in expected_rows.items():
        if len(tables[name]) != count:
            failures.append("table-row-count:" + name)
    for name in ("absorption_eh.dat", "absorption_noeh.dat"):
        rows = tables[name]
        if len(rows) < 100 or any(rows[index][0] <= rows[index - 1][0] for index in range(1, len(rows))):
            failures.append("spectrum-grid:" + name)
        if not rows or max(abs(row[1]) for row in rows) <= 1.0e-12:
            failures.append("zero-spectrum:" + name)
    for name, block_size in (("eqp.dat", 8), ("eqp_q.dat", 8)):
        rows = tables[name]
        index = 0
        blocks = 0
        valid = True
        while index < len(rows):
            header = rows[index]
            count = int(round(header[3]))
            if not _close(header[3], float(count)) or count != block_size or index + count >= len(rows):
                valid = False
                break
            for row in rows[index + 1:index + count + 1]:
                if not _close(row[0], 1.0) or not _close(row[1], round(row[1])):
                    valid = False
            blocks += 1
            index += count + 1
        if not valid or index != len(rows) or blocks != grid_count:
            failures.append("table-block-schema:" + name)
    for name, amplitude_column, real_column, imaginary_column in (
        ("eigenvalues.dat", 1, 2, 3),
        ("eigenvalues_noeh.dat", 7, 8, 9),
    ):
        if any(
            not _close(
                row[amplitude_column],
                row[real_column] ** 2 + row[imaginary_column] ** 2,
                2.0e-6,
            )
            for row in tables[name]
        ):
            failures.append("dipole-schema:" + name)
    eh, noeh = tables["absorption_eh.dat"], tables["absorption_noeh.dat"]
    if len(eh) != len(noeh) or any(not _close(a[0], b[0]) for a, b in zip(eh, noeh)):
        failures.append("spectrum-grid-mismatch")
    elif all(all(_close(x, y, 1.0e-10) for x, y in zip(a, b)) for a, b in zip(eh, noeh)):
        failures.append("electron-hole-spectrum-equals-independent")
    return tables, failures


def validate_run(
    output: Path, work: Path, case_path: Path, structure: Path, pseudo: Path,
    expected_work: Path | None = None,
) -> list[str]:
    failures: list[str] = []
    try:
        if not output.is_dir() or {path.name for path in output.iterdir()} != set(OUTPUTS):
            return ["output-file-set"]
        if any(not path.is_file() or path.stat().st_size == 0 or path.stat().st_size > 16 * 1024 * 1024 for path in output.iterdir()):
            failures.append("output-size-or-type")
        manifest = _json(output / "run_manifest.json")
        case = _json(case_path)
    except Exception:
        return failures + ["manifest-or-case-invalid"]
    required_case = {"bands", "screening_bands", "sigma_bands", "ecutwfc_ry", "energy_resolution_ev", "epsilon_cutoff_ry", "kgrid", "lattice_scale", "qshift"}
    if set(case) != required_case:
        failures.append("case-schema")
    if manifest.get("schema_version") != 2 or manifest.get("runtime_image") != RUNTIME_IMAGE:
        failures.append("runtime-contract")
    if manifest.get("case") != case:
        failures.append("case-not-consumed")
    expected_inputs = {"case": sha256(case_path), "structure": sha256(structure), "pseudo": sha256(pseudo)}
    if manifest.get("input_sha256") != expected_inputs:
        failures.append("input-hashes")
    outputs = manifest.get("outputs", {})
    if not isinstance(outputs, dict) or set(outputs) != set(SCIENTIFIC_OUTPUTS):
        failures.append("output-hash-set")
    else:
        for name in SCIENTIFIC_OUTPUTS:
            if outputs.get(name) != sha256(output / name):
                failures.append("output-hash:" + name)

    by_stage, stage_failures = _manifest_stages(manifest)
    failures.extend(stage_failures)
    for stage, executable in STAGES.items():
        record = by_stage.get(stage, {})
        if record.get("executable") != executable or record.get("return_code") != 0:
            failures.append("stage-executable-or-return:" + stage)
        deck_name, log_name = record.get("deck"), record.get("log")
        if not isinstance(deck_name, str) or not isinstance(log_name, str) or Path(deck_name).name != deck_name or Path(log_name).name != log_name:
            failures.append("stage-path:" + stage)
            continue
        deck, log = work / deck_name, work / log_name
        if not deck.is_file() or record.get("deck_sha256") != (sha256(deck) if deck.is_file() else None):
            failures.append("deck-evidence:" + stage)
        else:
            if stage in {"pw_scf", "pw_wfn", "pw_wfnq"}:
                failures.extend(_validate_qe_protocol(stage, deck, case, structure))
            elif stage.startswith("pw2bgw"):
                failures.extend(_validate_converter(stage, deck, case))
            else:
                failures.extend(_validate_bgw_protocol(stage, deck, case))
        if not log.is_file() or record.get("log_sha256") != (sha256(log) if log.is_file() else None):
            failures.append("log-evidence:" + stage)
        elif not _native_log(stage, log.read_text(encoding="utf-8", errors="replace"), deck_name, case):
            failures.append("stage-incomplete:" + stage)

    workflow = (output / "workflow.py").read_text(encoding="utf-8", errors="replace").lower()
    if "numpy.linalg.eigh" in workflow or "np.linalg.eigh" in workflow:
        failures.append("substitute-eigensolver")
    for executable in set(STAGES.values()):
        if executable not in workflow:
            failures.append("workflow-omits:" + executable)

    scf_record = by_stage.get("pw_scf", {})
    scf_name = scf_record.get("log")
    scf_log = work / scf_name if isinstance(scf_name, str) else Path("missing")
    if not scf_log.is_file():
        failures.append("scf-log-missing")
    else:
        text = scf_log.read_text(encoding="utf-8", errors="replace")
        reported_volume = _extract(r"unit-cell volume\s*=\s*([0-9.eEdD+-]+)", text)
        reported_cutoff = _extract(r"kinetic-energy cutoff\s*=\s*([0-9.eEdD+-]+)", text)
        reported_kpoints = _extract(r"number of k points\s*=\s*([0-9.eEdD+-]+)", text)
        try:
            _, _, volume_angstrom = _poscar(structure, float(case["lattice_scale"]))
            expected_volume = volume_angstrom * ANGSTROM_TO_BOHR ** 3
        except Exception:
            expected_volume = math.nan
        if reported_volume is None or not _close(reported_volume, expected_volume, 2.0e-3):
            failures.append("structure-not-consumed")
        if reported_cutoff is None or not _close(reported_cutoff, float(case["ecutwfc_ry"]), 1.0e-6):
            failures.append("cutoff-not-consumed")
        if reported_kpoints is None or int(round(reported_kpoints)) != int(case["kgrid"]) ** 3:
            failures.append("kgrid-not-consumed")
        candidate_md5 = re.search(r"MD5 check sum:\s*([0-9a-f]+)", text, re.IGNORECASE)
        expected_md5 = _expected_scf_md5(expected_work)
        if expected_work is not None and expected_md5 is None:
            failures.append("expected-pseudopotential-provenance")
        elif expected_md5 is not None and (not candidate_md5 or candidate_md5.group(1).lower() != expected_md5):
            failures.append("pseudopotential-not-consumed")

    _, table_failures = _validate_table_schemas(output, case)
    failures.extend(table_failures)
    return list(dict.fromkeys(failures))


def _table_similarity(candidate: list[list[float]], expected: list[list[float]]) -> float:
    if len(candidate) != len(expected) or not candidate:
        return 0.0
    columns = len(expected[0])
    if any(len(row) != columns for row in candidate + expected):
        return 0.0
    column_scores = []
    for column in range(columns):
        reference = [row[column] for row in expected]
        values = [row[column] for row in candidate]
        scale = max(max(abs(value) for value in reference), 1.0)
        rmse = math.sqrt(sum((left - right) ** 2 for left, right in zip(values, reference)) / len(reference))
        column_scores.append(max(0.0, 1.0 - rmse / (0.01 * scale + 1.0e-7)))
    return sum(column_scores) / len(column_scores)


def _physical_key(values: list[float]) -> tuple[float, ...]:
    """Stable key for coordinates or energies printed with finite precision."""
    return tuple(round(value, 7) for value in values)


def _canonical_eqp(rows: list[list[float]]) -> list[list[float]]:
    """Flatten native k-point blocks and sort by k point, spin, then band."""
    flattened: list[list[float]] = []
    index = 0
    while index < len(rows):
        header = rows[index]
        count = int(round(header[3]))
        if count < 1 or index + count >= len(rows):
            raise ValueError("malformed eqp block")
        for state in rows[index + 1:index + count + 1]:
            flattened.append([header[0], header[1], header[2], *state])
        index += count + 1
    if index != len(rows):
        raise ValueError("trailing eqp rows")
    return sorted(
        flattened,
        key=lambda row: (*_physical_key(row[:3]), round(row[3]), round(row[4])),
    )


def _same_degenerate_energy(left: float, right: float) -> bool:
    return abs(left - right) <= DEGENERATE_TOL_EV * max(1.0, abs(left), abs(right))


def _canonical_bse_eigenvalues(rows: list[list[float]]) -> list[list[float]]:
    """Represent each degenerate exciton subspace by spectral invariants.

    Within a degenerate manifold the eigenvectors and their complex optical
    amplitudes may undergo an arbitrary unitary rotation.  Multiplicity,
    energy centroid, and total oscillator strength are invariant.
    """
    ordered = sorted(rows, key=lambda row: row[0])
    groups: list[list[list[float]]] = []
    for row in ordered:
        if not groups or not _same_degenerate_energy(row[0], groups[-1][0][0]):
            groups.append([row])
        else:
            groups[-1].append(row)
    return [
        [float(len(group)), sum(row[0] for row in group) / len(group), sum(row[1] for row in group)]
        for group in groups
    ]


def _canonical_independent_eigenvalues(rows: list[list[float]]) -> list[list[float]]:
    """Discard enumeration labels and reduce degenerate transitions invariantly."""
    energy_columns = (4, 5, 6)
    ordered = sorted(rows, key=lambda row: _physical_key([row[column] for column in energy_columns]))
    groups: list[list[list[float]]] = []
    for row in ordered:
        energies = [row[column] for column in energy_columns]
        if not groups:
            groups.append([row])
            continue
        anchor = groups[-1][0]
        anchor_energies = [anchor[column] for column in energy_columns]
        if all(_same_degenerate_energy(left, right) for left, right in zip(energies, anchor_energies)):
            groups[-1].append(row)
        else:
            groups.append([row])
    invariants = []
    for group in groups:
        invariants.append([
            float(len(group)),
            *(sum(row[column] for row in group) / len(group) for column in energy_columns),
            sum(row[7] for row in group),
        ])
    return sorted(invariants, key=lambda row: _physical_key(row[1:4]))


def _canonical_table(name: str, rows: list[list[float]]) -> list[list[float]]:
    if name == "bandstructure.dat":
        return sorted(
            rows,
            key=lambda row: (*_physical_key(row[2:5]), round(row[0]), round(row[1])),
        )
    if name in {"eqp.dat", "eqp_q.dat"}:
        return _canonical_eqp(rows)
    if name in {"absorption_eh.dat", "absorption_noeh.dat"}:
        return sorted(rows, key=lambda row: round(row[0], 9))
    if name == "eigenvalues.dat":
        return _canonical_bse_eigenvalues(rows)
    if name == "eigenvalues_noeh.dat":
        return _canonical_independent_eigenvalues(rows)
    raise ValueError(f"no canonicalizer for {name}")


def compare_outputs(candidate: Path, expected: Path) -> tuple[float, dict[str, float]]:
    components: dict[str, float] = {}
    for name, columns in TABLE_COLUMNS.items():
        try:
            components[name] = _table_similarity(
                _canonical_table(name, numeric_rows(candidate / name, columns)),
                _canonical_table(name, numeric_rows(expected / name, columns)),
            )
        except Exception:
            components[name] = 0.0
    return sum(components.values()) / len(components), components


def score_run(
    output: Path, work: Path, case: Path, structure: Path, pseudo: Path,
    expected: Path, expected_work: Path | None = None,
) -> dict:
    failures = validate_run(output, work, case, structure, pseudo, expected_work)
    quality, components = compare_outputs(output, expected) if not failures else (0.0, {})
    if not failures:
        for name, component in components.items():
            if component < 0.90:
                failures.append("numeric-mismatch:" + name)
        if failures:
            quality = 0.0
    return {"score": quality, "failures": failures, "components": components}
