#!/usr/bin/env python3
"""Run the benchmark-scale QE 6.7/BerkeleyGW 4.0 silicon workflow.

This author-side oracle intentionally shells out to the scientific executables.
It contains no substitute Hamiltonian or synthetic spectrum implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def parse_poscar(path: Path, scale: float) -> tuple[list[list[float]], list[list[float]]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    factor = float(lines[1]) * scale
    lattice = [[factor * float(value) for value in lines[row].split()[:3]] for row in range(2, 5)]
    counts = [int(value) for value in lines[6].split()]
    if sum(counts) != 2 or lines[7].lower()[0] != "d":
        raise ValueError("expected the two-atom direct-coordinate silicon POSCAR")
    atoms = [[float(value) for value in lines[row].split()[:3]] for row in range(8, 10)]
    return lattice, atoms


def kpoints(grid: int, shift_z: float = 0.0) -> list[tuple[float, float, float]]:
    return [
        (i / grid, j / grid, k / grid + shift_z)
        for i in range(grid) for j in range(grid) for k in range(grid)
    ]


def qe_input(
    calculation: str,
    lattice: list[list[float]],
    atoms: list[list[float]],
    pseudo_dir: Path,
    scratch: Path,
    grid: int,
    bands: int,
    cutoff_ry: float,
    shift_z: float = 0.0,
) -> str:
    points = kpoints(grid, shift_z)
    lines = [
        "&control", " prefix='silicon',", f" pseudo_dir='{pseudo_dir}',",
        f" outdir='{scratch}',", f" calculation='{calculation}',", "/",
        "&system", " ibrav=0,", " nat=2,", " ntyp=1,", f" ecutwfc={cutoff_ry:.8f},",
        f" nbnd={bands},", " nosym=.true.,", " noinv=.true.,", "/",
        "&electrons", " conv_thr=1.0d-9,", " diagonalization='david',", "/",
        "CELL_PARAMETERS angstrom",
        *[" ".join(f"{value:.12f}" for value in row) for row in lattice],
        "ATOMIC_SPECIES", "Si 28.0855 Si.UPF", "ATOMIC_POSITIONS crystal",
        *["Si " + " ".join(f"{value:.12f}" for value in atom) for atom in atoms],
        "K_POINTS crystal", str(len(points)),
        *[f"{x:.12f} {y:.12f} {z:.12f} 1.0" for x, y, z in points],
    ]
    return "\n".join(lines) + "\n"


def converter_input(grid: int, bands: int, shift_z: float, include_density: bool) -> str:
    density = """
 rhog_flag=.true.,
 rhog_file='RHO',
 vxc_flag=.true.,
 vxc_file='vxc.dat',
 vxc_diag_nmin=1,
 vxc_diag_nmax={bands},
""".format(bands=bands) if include_density else ""
    return f"""&input_pw2bgw
 prefix='silicon',
 real_or_complex=2,
 wfng_flag=.true.,
 wfng_file='WFN.generated',
{density} wfng_kgrid=.true.,
 wfng_nk1={grid}, wfng_nk2={grid}, wfng_nk3={grid},
 wfng_dk1=0.0, wfng_dk2=0.0, wfng_dk3={shift_z},
/
"""


def run_stage(name: str, command: list[str], directory: Path, stdin: Path | None = None) -> None:
    log = directory / f"{name}.log"
    available_cpus = os.cpu_count() or 1
    family_variable = "ALE_QE_MPI_PROCS" if name.startswith("pw") else "ALE_BGW_MPI_PROCS"
    configured_processes = os.environ.get(family_variable, os.environ.get("ALE_MPI_PROCS"))
    mpi_processes = int(configured_processes or min(16, available_cpus))
    if mpi_processes < 1 or mpi_processes > available_cpus:
        raise ValueError(f"ALE_MPI_PROCS must be between 1 and {available_cpus}")
    parallel_command = ["mpirun", "-np", str(mpi_processes), *command]
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
    }
    with log.open("wb") as output:
        input_handle = stdin.open("rb") if stdin else None
        try:
            result = subprocess.run(
                parallel_command, cwd=directory, stdin=input_handle, stdout=output,
                stderr=subprocess.STDOUT, timeout=1800, check=False,
                env=environment,
            )
        finally:
            if input_handle:
                input_handle.close()
    if result.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise RuntimeError(f"{name} failed ({result.returncode})\n{tail}")


def link(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source.resolve())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execute(case_path: Path, structure: Path, pseudo: Path, work: Path, output: Path) -> None:
    case = json.loads(case_path.read_text(encoding="utf-8"))
    grid = int(case["kgrid"])
    bands = int(case["bands"])
    screening_bands = int(case["screening_bands"])
    cutoff = float(case["ecutwfc_ry"])
    lattice, atoms = parse_poscar(structure, float(case["lattice_scale"]))
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f"work directory is not empty: {work}")
    work.mkdir(parents=True, exist_ok=True)
    pseudo_dir = work / "pseudo"
    pseudo_dir.mkdir()
    shutil.copy2(pseudo, pseudo_dir / "Si.UPF")
    scratch = work / "scratch"
    scratch.mkdir()

    scf = work / "scf.in"
    scf.write_text(qe_input("scf", lattice, atoms, pseudo_dir, scratch, grid, bands, cutoff), encoding="utf-8")
    run_stage("pw_scf", ["pw.x", "-in", scf.name], work)

    wfn = work / "wfn.in"
    wfn.write_text(qe_input("bands", lattice, atoms, pseudo_dir, scratch, grid, bands, cutoff), encoding="utf-8")
    run_stage("pw_wfn", ["pw.x", "-in", wfn.name], work)
    link(scratch / "silicon.save", work / "silicon.save")
    convert = work / "wfn.pp.in"
    convert.write_text(converter_input(grid, bands, 0.0, True), encoding="utf-8")
    run_stage("pw2bgw_wfn", ["pw2bgw.x", "-in", convert.name], work)
    (work / "WFN.generated").rename(work / "WFN")

    wfnq = work / "wfnq.in"
    qshift = float(case["qshift"])
    wfnq.write_text(qe_input("bands", lattice, atoms, pseudo_dir, scratch, grid, bands, cutoff, qshift), encoding="utf-8")
    run_stage("pw_wfnq", ["pw.x", "-in", wfnq.name], work)
    convert_q = work / "wfnq.pp.in"
    convert_q.write_text(converter_input(grid, bands, qshift, False), encoding="utf-8")
    run_stage("pw2bgw_wfnq", ["pw2bgw.x", "-in", convert_q.name], work)
    (work / "WFN.generated").rename(work / "WFNq")

    qpoint_lines = [f"0.0 0.0 {qshift:.12f} 1.0 1"]
    qpoint_lines.extend(
        f"{x:.12f} {y:.12f} {z:.12f} 1.0 0"
        for x, y, z in kpoints(grid) if (x, y, z) != (0.0, 0.0, 0.0)
    )
    (work / "epsilon.inp").write_text(
        f"epsilon_cutoff {float(case['epsilon_cutoff_ry']):.8f}\nnumber_bands {screening_bands}\n"
        "begin qpoints\n" + "\n".join(qpoint_lines) + "\nend\n",
        encoding="utf-8",
    )
    run_stage("epsilon", ["epsilon.cplx.x"], work)

    sigma_points = "\n".join(f"{x:.12f} {y:.12f} {z:.12f} 1.0" for x, y, z in kpoints(grid))
    (work / "sigma.inp").write_text(
        f"number_bands {int(case['sigma_bands'])}\n"
        "number_sigma_pools 8\n"
        "band_index_min 1\nband_index_max 8\nscreening_semiconductor\n"
        "no_symmetries_q_grid\n"
        "begin kpoints\n" + sigma_points + "\nend\n", encoding="utf-8")
    link(work / "WFN", work / "WFN_inner")
    run_stage("sigma", ["sigma.cplx.x"], work)

    (work / "kernel.inp").write_text(
        "number_val_bands 4\nnumber_cond_bands 4\nno_symmetries_coarse_grid\n"
        "screening_semiconductor\n", encoding="utf-8")
    link(work / "WFN", work / "WFN_co")
    run_stage("kernel", ["kernel.cplx.x"], work)

    absorption = (
        "number_val_bands_coarse 4\nnumber_val_bands_fine 4\n"
        "number_cond_bands_coarse 4\nnumber_cond_bands_fine 4\n"
        f"energy_resolution {float(case['energy_resolution_ev']):.8f}\n"
        "no_symmetries_coarse_grid\nno_symmetries_fine_grid\nno_symmetries_shifted_grid\n"
        "screening_semiconductor\nuse_momentum\nlorentzian_broadening\n"
        "write_eigenvectors -1\neqp_co_corrections\n"
    )
    (work / "absorption.inp").write_text(absorption, encoding="utf-8")
    link(work / "WFN", work / "WFN_fi")
    link(work / "WFNq", work / "WFNq_fi")
    link(work / "eqp1.dat", work / "eqp_co.dat")
    run_stage("absorption", ["absorption.cplx.x"], work)

    (work / "inteqp.inp").write_text(
        "number_val_bands_coarse 4\nnumber_val_bands_fine 4\n"
        "number_cond_bands_coarse 4\nnumber_cond_bands_fine 4\n"
        "no_symmetries_coarse_grid\nno_symmetries_fine_grid\nuse_momentum\n",
        encoding="utf-8",
    )
    run_stage("inteqp", ["inteqp.cplx.x"], work)

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_files = {
        "bandstructure.dat": work / "bandstructure.dat",
        "eqp.dat": work / "eqp.dat",
        "eqp_q.dat": work / "eqp1.dat",
        "absorption_eh.dat": work / "absorption_b1_eh.dat",
        "absorption_noeh.dat": work / "absorption_b1_noeh.dat",
        "eigenvalues.dat": work / "eigenvalues_b1.dat",
        "eigenvalues_noeh.dat": work / "eigenvalues_b1_noeh.dat",
    }
    for name, source in source_files.items():
        shutil.copy2(source, output / name)
    shutil.copy2(Path(__file__), output / "workflow.py")

    stage_executables = {
        "pw_scf": "pw.x", "pw_wfn": "pw.x", "pw2bgw_wfn": "pw2bgw.x",
        "pw_wfnq": "pw.x", "pw2bgw_wfnq": "pw2bgw.x", "epsilon": "epsilon.cplx.x",
        "sigma": "sigma.cplx.x", "kernel": "kernel.cplx.x",
        "absorption": "absorption.cplx.x", "inteqp": "inteqp.cplx.x",
    }
    deck_for_stage = {
        "pw_scf": "scf.in", "pw_wfn": "wfn.in", "pw2bgw_wfn": "wfn.pp.in",
        "pw_wfnq": "wfnq.in", "pw2bgw_wfnq": "wfnq.pp.in", "epsilon": "epsilon.inp",
        "sigma": "sigma.inp", "kernel": "kernel.inp",
        "absorption": "absorption.inp", "inteqp": "inteqp.inp",
    }
    stages = []
    for stage, executable in stage_executables.items():
        deck = work / deck_for_stage[stage]
        log = work / f"{stage}.log"
        stages.append({
            "stage": stage,
            "executable": executable,
            "deck": deck.name,
            "deck_sha256": sha256(deck),
            "log": log.name,
            "log_sha256": sha256(log),
            "return_code": 0,
        })
    manifest = {
        "schema_version": 2,
        "runtime_image": "ale/base-physical-qe-bgw:6.7.0-4.0-20260920",
        "case": case,
        "input_sha256": {
            "case": sha256(case_path), "structure": sha256(structure), "pseudo": sha256(pseudo),
        },
        "stages": stages,
        "outputs": {name: sha256(output / name) for name in source_files},
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--structure", type=Path, required=True)
    parser.add_argument("--pseudo", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    execute(
        args.case.resolve(), args.structure.resolve(), args.pseudo.resolve(),
        args.work_dir.resolve(), args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
