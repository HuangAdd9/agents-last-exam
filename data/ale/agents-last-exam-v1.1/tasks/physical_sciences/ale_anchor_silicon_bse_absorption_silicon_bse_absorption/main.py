"""ALE hooks for a real QE 6.7/BerkeleyGW 4.0 silicon GW-BSE task."""

from __future__ import annotations

import importlib.util
import json
import logging
import shlex
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
DOMAIN_NAME = "physical_sciences"
TASK_NAME = "ale_anchor_silicon_bse_absorption_silicon_bse_absorption"
INPUT_FILES = ("problem_spec.md", "workflow_case.json", "silicon.vasp", "Si.UPF", "runtime_manifest.json")
OUTPUT_FILES = (
    "bandstructure.dat", "eqp.dat", "eqp_q.dat", "absorption_eh.dat",
    "absorption_noeh.dat", "eigenvalues.dat", "eigenvalues_noeh.dat",
    "workflow.py", "run_manifest.json",
)
HIDDEN_CASE = {
    "bands": 41,
    "ecutwfc_ry": 11.0,
    "energy_resolution_ev": 0.12,
    "epsilon_cutoff_ry": 10.5,
    "kgrid": 5,
    "lattice_scale": 1.008,
    "qshift": 0.003,
    "screening_bands": 27,
    "sigma_bands": 23,
}
RUNTIME_ROOT = "/opt/qe-bgw-6.7.0-4.0"
PRIVATE_REFERENCE = ROOT / "evaluator_private"
CANDIDATE_UID = 60000
EVALUATOR_UID = 1000


def _load_oracle():
    path = ROOT / "scripts" / "oracle.py"
    spec = importlib.util.spec_from_file_location(f"{TASK_NAME}_oracle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load task oracle")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _return_code(result) -> int:
    if isinstance(result, dict):
        return int(result.get("return_code", result.get("returncode", 1)))
    return int(getattr(result, "return_code", getattr(result, "returncode", 1)))


def _command_diagnostics(result) -> str:
    if isinstance(result, dict):
        stdout, stderr = result.get("stdout", ""), result.get("stderr", "")
    else:
        stdout = getattr(result, "stdout", "")
        stderr = getattr(result, "stderr", "")
    return (str(stdout) + "\n" + str(stderr)).strip()[-2000:]


@dataclass
class Config(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = "base"
    OS_TYPE: str = "linux"

    @property
    def task_description(self) -> str:
        return json.loads((ROOT / "task_card.json").read_text(encoding="utf-8"))["taskPrompt"]

    def to_metadata(self) -> dict:
        metadata = super().to_metadata()
        metadata.update({
            "task_id": f"{DOMAIN_NAME}/{TASK_NAME}",
            "input_files": list(INPUT_FILES),
            "output_files": list(OUTPUT_FILES),
            "work_dir": f"{self.task_dir}/workflow",
            "software_bin_dir": f"{self.software_dir}/bin",
            "evaluator_dir": f"{self.task_dir}/.evaluator_only",
        })
        return metadata


CONFIG = Config()
_setup = BaseTaskSetup()


@cb.tasks_config(split="train")
def load():
    return [cb.Task(
        description=CONFIG.task_description,
        metadata=CONFIG.to_metadata(),
        computer={"provider": "computer", "setup_config": {"os_type": CONFIG.OS_TYPE}},
    )]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    """Verify the framework-staged inputs and the pinned offline runtime."""
    await _setup(task_cfg, session)
    metadata = task_cfg.metadata
    for directory in (metadata["remote_output_dir"], metadata["work_dir"]):
        await session.interface.create_dir(directory)

    # Input and software trees are task data, staged by the ALE runner before
    # this hook executes.  Probe them through the VM shell instead of reading
    # host-side package paths or relying on a non-portable session read helper.
    for name in INPUT_FILES:
        staged = f"{metadata['input_dir']}/{name}"
        probe = await session.run_command(f"test -s {shlex.quote(staged)}", check=False)
        if _return_code(probe) != 0:
            raise RuntimeError(f"staged input is unreadable: {name}")

    bootstrap_remote = f"{metadata['software_dir']}/bootstrap_remote_software.sh"
    command = (
        f"chmod 0700 {shlex.quote(bootstrap_remote)} && "
        f"QE_BGW_BIN={shlex.quote(metadata['software_bin_dir'])} "
        f"RUNTIME_MANIFEST={shlex.quote(metadata['input_dir'] + '/runtime_manifest.json')} "
        f"bash {shlex.quote(bootstrap_remote)}"
    )
    result = await session.run_command(command, timeout=120, check=False)
    if _return_code(result) != 0:
        raise RuntimeError(
            "pinned QE/BerkeleyGW runtime verification failed: "
            + _command_diagnostics(result)
        )


async def _remote_files(session, directory: str) -> set[str] | None:
    result = await session.run_command(
        f"find {shlex.quote(directory)} -mindepth 1 -maxdepth 1 -type f -printf '%f\\n' | LC_ALL=C sort",
        check=False,
    )
    if _return_code(result) != 0:
        return None
    stdout = result.get("stdout", "") if isinstance(result, dict) else getattr(result, "stdout", "")
    return {line for line in str(stdout).splitlines() if line}


async def _pull_output(session, remote: str, local: Path) -> bool:
    local.mkdir(parents=True, exist_ok=True)
    try:
        for name in OUTPUT_FILES:
            blob = await session.interface.read_bytes(f"{remote}/{name}")
            if not blob or len(blob) > 16 * 1024 * 1024:
                return False
            (local / name).write_bytes(bytes(blob))
    except Exception:
        return False
    return True


async def _pull_work(session, remote: str, output: Path, local: Path) -> bool:
    try:
        manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
        local.mkdir(parents=True, exist_ok=True)
        names = set()
        for record in manifest.get("stages", []):
            for key in ("deck", "log"):
                name = record.get(key)
                if not isinstance(name, str) or Path(name).name != name or name in names:
                    return False
                names.add(name)
        for name in names:
            blob = await session.interface.read_bytes(f"{remote}/{name}")
            if not blob or len(blob) > 16 * 1024 * 1024:
                return False
            (local / name).write_bytes(bytes(blob))
        return True
    except Exception:
        return False


async def _pull_input(session, remote: str, local: Path) -> bool:
    """Materialize one staged input for the local semantic oracle."""
    try:
        blob = await session.interface.read_bytes(remote)
        if not blob or len(blob) > 16 * 1024 * 1024:
            return False
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(bytes(blob))
        return True
    except Exception:
        return False


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    metadata = task_cfg.metadata
    if await _remote_files(session, metadata["remote_output_dir"]) != set(OUTPUT_FILES):
        return [0.0]
    oracle = _load_oracle()
    with tempfile.TemporaryDirectory(prefix="ale_si_bse_eval_") as temporary:
        root = Path(temporary)
        visible_output, visible_work = root / "visible_output", root / "visible_work"
        if not await _pull_output(session, metadata["remote_output_dir"], visible_output):
            return [0.0]
        if not await _pull_work(session, metadata["work_dir"], visible_output, visible_work):
            return [0.0]

        visible_case = root / "workflow_case.json"
        visible_structure = root / "silicon.vasp"
        visible_pseudo = root / "Si.UPF"
        visible_expected = PRIVATE_REFERENCE / "visible_reference"
        visible_expected_work = PRIVATE_REFERENCE / "visible_evidence"
        input_dir = metadata["input_dir"]
        if not all((
            await _pull_input(session, f"{input_dir}/workflow_case.json", visible_case),
            await _pull_input(session, f"{input_dir}/silicon.vasp", visible_structure),
            await _pull_input(session, f"{input_dir}/Si.UPF", visible_pseudo),
        )):
            raise RuntimeError("could not retrieve staged public inputs")
        if not visible_expected.is_dir() or not visible_expected_work.is_dir():
            raise RuntimeError("evaluator-local visible reference is unavailable")
        visible = oracle.score_run(
            visible_output, visible_work, visible_case, visible_structure,
            visible_pseudo, visible_expected, visible_expected_work,
        )
        visible_score = float(visible["score"])
        if visible["failures"]:
            logger.warning("visible workflow failures: %s", visible["failures"])
            return [0.0]

        evaluator = metadata["evaluator_dir"]
        candidate_sandbox = f"{metadata['task_dir']}/.candidate_rerun"
        candidate_work = f"{candidate_sandbox}/work"
        candidate_output = f"{candidate_sandbox}/output"
        oracle_work = f"{evaluator}/oracle_work"
        oracle_output = f"{evaluator}/oracle_output"
        hidden_case_remote = f"{candidate_sandbox}/case.json"
        hidden_structure_remote = f"{candidate_sandbox}/silicon_hidden.vasp"
        hidden_pseudo_remote = f"{candidate_sandbox}/Si_hidden.UPF"
        candidate_script_remote = f"{candidate_sandbox}/workflow.py"
        oracle_case_remote = f"{evaluator}/case.json"
        oracle_structure_remote = f"{evaluator}/silicon_hidden.vasp"
        oracle_pseudo_remote = f"{evaluator}/Si_hidden.UPF"
        oracle_script_remote = f"{evaluator}/oracle_workflow.py"
        public_structure = visible_structure.read_bytes()
        hidden_structure = public_structure.replace(b"2.715500000", b"2.688000000")
        public_pseudo = visible_pseudo.read_bytes()
        hidden_pseudo = public_pseudo.replace(
            b"6.749115800389129E-001", b"7.749115800389129E-001", 1
        )
        hidden_case = (json.dumps(HIDDEN_CASE, indent=2, sort_keys=True) + "\n").encode()
        try:
            task_path = Path(metadata["task_dir"])
            access_dirs = [str(task_path.parents[index]) for index in (2, 1, 0)] + [str(task_path)]
            access_args = " ".join(shlex.quote(path) for path in access_dirs)
            setup_command = (
                f"sudo -n rm -rf {shlex.quote(evaluator)} {shlex.quote(candidate_sandbox)} && "
                f"chmod 0711 {access_args} && "
                f"install -d -m 0700 {shlex.quote(evaluator)} && "
                f"install -d -m 0755 {shlex.quote(candidate_sandbox)} && "
                f"install -d -m 0777 {shlex.quote(candidate_work)} {shlex.quote(candidate_output)}"
            )
            setup = await session.run_command(setup_command, check=False)
            if _return_code(setup) != 0:
                raise RuntimeError("could not create isolated evaluator directories")
            for remote_path, data in (
                (hidden_case_remote, hidden_case),
                (hidden_structure_remote, hidden_structure),
                (hidden_pseudo_remote, hidden_pseudo),
                (candidate_script_remote, (visible_output / "workflow.py").read_bytes()),
                (oracle_case_remote, hidden_case),
                (oracle_structure_remote, hidden_structure),
                (oracle_pseudo_remote, hidden_pseudo),
                (oracle_script_remote, (ROOT / "scripts/oracle_workflow.py").read_bytes()),
            ):
                await session.interface.write_bytes(remote_path, data)
            await session.run_command(
                f"chmod 0444 {shlex.quote(hidden_case_remote)} "
                f"{shlex.quote(hidden_structure_remote)} {shlex.quote(hidden_pseudo_remote)} "
                f"{shlex.quote(candidate_script_remote)} && "
                f"chmod 0600 {shlex.quote(oracle_case_remote)} "
                f"{shlex.quote(oracle_structure_remote)} {shlex.quote(oracle_pseudo_remote)} "
                f"{shlex.quote(oracle_script_remote)} && "
                f"sudo -n chown -R root:root {shlex.quote(evaluator)} && "
                f"sudo -n chmod 0700 {shlex.quote(evaluator)}",
                check=True,
            )
            # start() verifies this immutable runtime and creates executable
            # links, but shell exports made by the bootstrap do not persist to
            # later DesktopSession commands. Carry the matching OpenMPI
            # runtime environment into both independent workflow processes.
            candidate_environment = (
                f"OPAL_PREFIX={shlex.quote(RUNTIME_ROOT + '/envs/qe-bgw')} "
                f"LD_LIBRARY_PATH={shlex.quote(RUNTIME_ROOT + '/envs/qe-bgw/lib')} "
                f"PATH={shlex.quote(metadata['software_bin_dir'])}:"
                f"{shlex.quote(RUNTIME_ROOT + '/envs/qe-bgw/bin')}:/usr/bin:/bin HOME=/tmp"
            )
            candidate_command = (
                f"sudo -n env {candidate_environment} setpriv --reuid={CANDIDATE_UID} "
                f"--regid={CANDIDATE_UID} --clear-groups timeout --kill-after=10 3200 /usr/bin/python3 "
                f"{shlex.quote(candidate_script_remote)} "
                f"--case {shlex.quote(hidden_case_remote)} --structure {shlex.quote(hidden_structure_remote)} "
                f"--pseudo {shlex.quote(hidden_pseudo_remote)} --work-dir {shlex.quote(candidate_work)} "
                f"--output-dir {shlex.quote(candidate_output)}"
            )
            candidate_run = await session.run_command(candidate_command, timeout=3300, check=False)
            await session.run_command(
                f"sudo -n pkill -KILL -u {CANDIDATE_UID} 2>/dev/null || true",
                check=False,
            )
            candidate_files = await _remote_files(session, candidate_output)
            if _return_code(candidate_run) != 0 or candidate_files != set(OUTPUT_FILES):
                logger.warning(
                    "candidate workflow failed (files=%s): %s",
                    candidate_files,
                    _command_diagnostics(candidate_run),
                )
                return [min(0.69, 0.35 * visible_score)]

            # Freeze the candidate evidence locally before the private oracle
            # starts. A process left behind by the original agent can neither
            # read the root-owned oracle directory nor change this local copy.
            local_case, local_structure, local_pseudo = root / "case.json", root / "silicon.vasp", root / "Si.UPF"
            local_case.write_bytes(hidden_case)
            local_structure.write_bytes(hidden_structure)
            local_pseudo.write_bytes(hidden_pseudo)
            hidden_output, hidden_work = root / "hidden_output", root / "hidden_work"
            hidden_expected, hidden_expected_work = root / "hidden_expected", root / "hidden_expected_work"
            if not await _pull_output(session, candidate_output, hidden_output):
                return [min(0.69, 0.35 * visible_score)]
            if not await _pull_work(session, candidate_work, hidden_output, hidden_work):
                return [min(0.69, 0.35 * visible_score)]

            oracle_command = (
                f"sudo -n env {candidate_environment} timeout 3200 /usr/bin/python3 "
                f"{shlex.quote(oracle_script_remote)} "
                f"--case {shlex.quote(oracle_case_remote)} --structure {shlex.quote(oracle_structure_remote)} "
                f"--pseudo {shlex.quote(oracle_pseudo_remote)} --work-dir {shlex.quote(oracle_work)} "
                f"--output-dir {shlex.quote(oracle_output)}"
            )
            oracle_run = await session.run_command(oracle_command, timeout=3300, check=False)
            if _return_code(oracle_run) != 0:
                raise RuntimeError("evaluator-only QE/BerkeleyGW oracle failed")
            await session.run_command(
                f"sudo -n chown -R {EVALUATOR_UID}:{EVALUATOR_UID} {shlex.quote(evaluator)}",
                check=True,
            )
            if not await _pull_output(session, oracle_output, hidden_expected):
                raise RuntimeError("could not retrieve evaluator oracle outputs")
            if not await _pull_work(session, oracle_work, hidden_expected, hidden_expected_work):
                raise RuntimeError("could not retrieve evaluator oracle stage evidence")
            hidden = oracle.score_run(
                hidden_output, hidden_work, local_case, local_structure, local_pseudo,
                hidden_expected, hidden_expected_work,
            )
            hidden_score = float(hidden["score"])
            if hidden["failures"]:
                logger.warning("hidden workflow failures: %s", hidden["failures"])
                hidden_score = 0.0
            total = 0.35 * visible_score + 0.65 * hidden_score
            if hidden_score < 0.75:
                total = min(total, 0.69)
            return [max(0.0, min(1.0, total))]
        finally:
            await session.run_command(
                f"sudo -n pkill -KILL -u {CANDIDATE_UID} 2>/dev/null || true; "
                f"sudo -n rm -rf {shlex.quote(evaluator)} {shlex.quote(candidate_sandbox)}",
                check=False,
            )
