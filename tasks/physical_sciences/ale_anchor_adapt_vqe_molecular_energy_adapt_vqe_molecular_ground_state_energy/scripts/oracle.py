"""Independent fixed-number-sector oracle (author/evaluator host only)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

CASES = {
    "tier1": ("h2_hamiltonian.json", "H2", "VQE"),
    "tier2": ("lih_hamiltonian.json", "LiH", "ADAPT-VQE"),
    "tier3": ("beh2_hamiltonian.json", "BeH2", "ADAPT-VQE"),
}


def _load_hamiltonian(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    n = payload.get("n_qubits")
    n_electrons = payload.get("n_electrons")
    terms = payload.get("pauli_terms")
    required = {"molecule", "bond_length", "bond_unit", "basis", "n_qubits", "n_electrons", "active_space", "n_pauli_terms", "pauli_terms"}
    if set(payload) != required or not isinstance(n, int) or isinstance(n, bool) or n <= 0 or not isinstance(n_electrons, int) or isinstance(n_electrons, bool) or n_electrons <= 0 or n_electrons >= n or not isinstance(terms, dict) or not isinstance(payload.get("n_pauli_terms"), int) or payload["n_pauli_terms"] <= 0 or len(terms) != payload.get("n_pauli_terms"):
        raise ValueError(f"invalid Hamiltonian metadata: {path.name}")
    if not isinstance(payload.get("bond_length"), (int, float)) or isinstance(payload["bond_length"], bool) or not math.isfinite(float(payload["bond_length"])) or payload["bond_length"] <= 0 or payload.get("bond_unit") not in {"angstrom", "bohr"} or not isinstance(payload.get("basis"), str) or not payload["basis"]:
        raise ValueError(f"invalid molecular schema: {path.name}")
    for word, coeff in terms.items():
        if not isinstance(word, str) or len(word) != n or set(word) - set("IXYZ") or not isinstance(coeff, (int, float)) or isinstance(coeff, bool) or not math.isfinite(coeff):
            raise ValueError(f"invalid Pauli term in {path.name}")
    return payload


def _validated_config(path: Path, payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema_version", "field_model", "site_z_coefficients_ha", "identity_shift_ha", "electron_sector_required", "hartree_to_ev"}
    if set(config) != required or config["schema_version"] != 2:
        raise ValueError("field configuration schema mismatch")
    if config["field_model"] != "site_resolved_z" or config["electron_sector_required"] is not True:
        raise ValueError("field model or sector flag mismatch")
    if set(config["site_z_coefficients_ha"]) != set(payloads):
        raise ValueError("field molecule keys mismatch")
    for key in ("identity_shift_ha", "hartree_to_ev"):
        value = config[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite numeric")
    if float(config["hartree_to_ev"]) <= 0.0:
        raise ValueError("hartree_to_ev must be positive")
    for molecule, payload in payloads.items():
        values = config["site_z_coefficients_ha"].get(molecule)
        if not isinstance(values, list) or len(values) != int(payload["n_qubits"]):
            raise ValueError(f"bad field vector for {molecule}")
        if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values):
            raise ValueError(f"nonfinite field for {molecule}")
    return config


def _sector_matrix(payload: dict[str, Any], config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Build directly in the number sector by bit actions, not Kronecker products."""
    n, n_electrons = int(payload["n_qubits"]), int(payload["n_electrons"])
    sector = np.array([state for state in range(1 << n) if state.bit_count() == n_electrons], dtype=np.int64)
    lookup = {int(state): index for index, state in enumerate(sector)}
    matrix = np.zeros((len(sector), len(sector)), dtype=np.complex128)
    terms = dict(payload["pauli_terms"])
    molecule = str(payload["molecule"])
    terms["I" * n] = terms.get("I" * n, 0.0) + float(config["identity_shift_ha"])
    for q, value in enumerate(config["site_z_coefficients_ha"][molecule]):
        word = "I" * q + "Z" + "I" * (n - q - 1)
        terms[word] = terms.get(word, 0.0) + float(value)
    for word, coeff in terms.items():
        for col, state_value in enumerate(sector):
            state, phase = int(state_value), 1.0 + 0.0j
            for q, symbol in enumerate(word):
                mask = 1 << (n - 1 - q)
                bit = bool(state & mask)
                if symbol == "X":
                    state ^= mask
                elif symbol == "Y":
                    state ^= mask
                    phase *= -1j if bit else 1j
                elif symbol == "Z" and bit:
                    phase *= -1.0
            row = lookup.get(state)
            if row is None:
                # Projection discards matrix elements leaving the chosen sector;
                # individual Pauli words need not conserve number even when the
                # molecular Hamiltonian does.
                continue
            matrix[row, col] += float(coeff) * phase
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12, rtol=0.0):
        raise ValueError("Hamiltonian is not Hermitian")
    return (matrix + matrix.conj().T) / 2.0, sector


def solve_case(payload: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    matrix, sector = _sector_matrix(payload, config)
    values, vectors = np.linalg.eigh(matrix)
    probabilities = np.abs(vectors[:, 0]) ** 2
    counts = np.array([int(x).bit_count() for x in sector], dtype=float)
    expectation = float(probabilities @ counts)
    variance = float(probabilities @ ((counts - expectation) ** 2))
    energy_ha = float(values[0])
    return {"energy_ha": energy_ha, "energy_ev": energy_ha * float(config["hartree_to_ev"]), "number_expectation": expectation, "number_variance": variance}


def build_reference(input_dir: Path, hidden_configs: list[Path]) -> dict[str, Any]:
    payloads: dict[str, dict[str, Any]] = {}
    for _, (filename, molecule, _) in CASES.items():
        payload = _load_hamiltonian(input_dir / filename)
        if payload.get("molecule") != molecule:
            raise ValueError(f"molecule mismatch for {filename}")
        payloads[molecule] = payload
    visible_config = _validated_config(input_dir / "field_config.json", payloads)
    configs = {"visible": visible_config}
    configs.update({f"hidden_{i+1}": _validated_config(path, payloads) for i, path in enumerate(hidden_configs)})
    result: dict[str, Any] = {"schema_version": 4, **{key: {} for key in configs}}
    for tier, (_, molecule, method) in CASES.items():
        for label, config in configs.items():
            result[label][tier] = {"molecule": molecule, "method": method, **solve_case(payloads[molecule], config)}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--hidden-config", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_reference(args.input_dir, args.hidden_config), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
