"""Site-resolved fixed-sector molecular energy benchmark."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

_setup = BaseTaskSetup()
logger = logging.getLogger(__name__)
DOMAIN_NAME = "physical_sciences"
TASK_NAME = "ale_anchor_adapt_vqe_molecular_energy_adapt_vqe_molecular_ground_state_energy"
VARIANT_NAME = "base"
SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"


def _load_scorer() -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{__name__}_score_outputs", SCRIPTS_DIR / "score_outputs.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load evaluator scorer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORER = _load_scorer()

_ALLOWED_IMPORTS = {"__future__", "argparse", "collections", "functools", "itertools", "json", "math", "numpy", "pathlib", "re", "scipy", "typing"}
_TRANSITIVE_BLOCKLIST = {"os", "sys", "subprocess", "socket", "shutil", "ctypes", "resource", "importlib", "builtins", "pickle", "marshal", "ssl"}


def _candidate_imports_allowed(tree: ast.AST) -> bool:
    """Reject direct and transitive standard-library escape hatches."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] not in _ALLOWED_IMPORTS for alias in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if node.module is None or node.module.split(".")[0] not in _ALLOWED_IMPORTS:
                return False
            if any(alias.name.split(".")[0] in _TRANSITIVE_BLOCKLIST for alias in node.names):
                return False
        elif isinstance(node, ast.Attribute) and node.attr in _TRANSITIVE_BLOCKLIST:
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in _ALLOWED_IMPORTS:
                return False
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"getattr", "__import__"}:
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and str(node.args[1].value).split(".")[0] in _TRANSITIVE_BLOCKLIST:
                return False
    return True


@dataclass
class AdaptVqeConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME
    TASK_NAME: str = TASK_NAME
    VARIANT_NAME: str = VARIANT_NAME

    @property
    def problem_spec(self) -> str:
        return f"{self.input_dir}/problem_spec.md"

    @property
    def output_results(self) -> str:
        return f"{self.remote_output_dir}/results.json"

    @property
    def output_solver(self) -> str:
        return f"{self.remote_output_dir}/solve.py"

    @property
    def eval_dir(self) -> str:
        return f"{self.task_dir}/eval_data"

    @property
    def task_description(self) -> str:
        return f"""\
You are solving a site-resolved, fixed-electron-number molecular energy
benchmark. Read `{self.problem_spec}` and every JSON file in `{self.input_dir}`.

Implement the Hamiltonian assembly and genuine VQE/ADAPT-VQE workflow using
the Hartree--Fock reference, physical excitation pool, commutator-gradient
selection, iterative joint reoptimisation, and stopping rule. Exact
diagonalisation is only a private cross-check and is not an acceptable
submitted method. Include reconstructable parameters and iteration evidence,
unit conversion, and state-derived invariants using only the standard library,
NumPy, and SciPy. Produce exactly two files:

- `{self.output_results}` for the visible configuration;
- `{self.output_solver}`, a reusable CLI solver with the interface specified in
  `problem_spec.md`.

After your interactive solve, the evaluator will invoke `solve.py` on two hidden
site-resolved field configurations and reconstruct every submitted ansatz. Do not hard-code the visible coefficients or
assume the field alternates. Follow the exact JSON schema and units. Do not
modify input files, inspect gated reference data, or use network access.
"""

    def to_metadata(self) -> dict[str, Any]:
        metadata = super().to_metadata()
        metadata.update({
            "task_id": f"{DOMAIN_NAME}/{TASK_NAME}",
            "problem_spec": self.problem_spec,
            "output_results": self.output_results,
            "output_solver": self.output_solver,
            "eval_dir": self.eval_dir,
        })
        return metadata


config = AdaptVqeConfig()


@cb.tasks_config(split="train")
def load():
    cfg = AdaptVqeConfig()
    return [cb.Task(description=cfg.task_description, metadata=cfg.to_metadata(), computer={"provider": "computer", "setup_config": {"os_type": "linux"}})]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


async def _read_candidate_json(session: cb.DesktopSession, path: str) -> Any | None:
    try:
        if not await session.file_exists(path):
            return None
        raw = await session.read_file(path)
        if not isinstance(raw, str) or len(raw.encode("utf-8")) > 100_000:
            return None
        return json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return None


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    results_path, solver_path = meta["output_results"], meta["output_solver"]
    if not await session.file_exists(results_path) or not await session.file_exists(solver_path):
        return [0.0]
    listing_result = await session.run_command(
        f'find "{meta["remote_output_dir"]}" -mindepth 1 -maxdepth 1 -printf "%f\\n"',
        check=False,
    )
    if not isinstance(listing_result, dict) or listing_result.get("return_code", 1) != 0:
        return [0.0]
    names = {line.strip() for line in listing_result.get("stdout", "").splitlines() if line.strip()}
    if names != {"results.json", "solve.py"}:
        return [0.0]
    try:
        solver_source = await session.read_file(solver_path)
    except Exception:
        return [0.0]
    if not isinstance(solver_source, str) or len(solver_source.encode("utf-8")) > 100_000:
        return [0.0]
    # Enforce a real import policy structurally; filesystem/network isolation
    # below is the security boundary rather than a token blacklist.
    try:
        tree = ast.parse(solver_source, filename="solve.py")
    except SyntaxError:
        return [0.0]
    if not _candidate_imports_allowed(tree):
        return [0.0]
    visible = await _read_candidate_json(session, results_path)
    if visible is None:
        return [0.0]

    reference_results = f'{meta["reference_dir"]}/results.json'
    hidden_configs = [f'{meta["reference_dir"]}/hidden_case.json', f'{meta["reference_dir"]}/hidden_case_2.json']
    if not await session.file_exists(reference_results) or not all([await session.file_exists(x) for x in hidden_configs]):
        raise RuntimeError("evaluator reference bundle is missing")
    try:
        reference = json.loads(await session.read_file(reference_results))
        hidden_config_texts = [await session.read_file(x) for x in hidden_configs]
    except Exception as exc:
        raise RuntimeError("evaluator reference bundle is unreadable") from exc

    # Verify every public scientific input before it can influence grading.
    expected_hashes = {"problem_spec.md":"1eb2fef84589b674dc2eea8b63c4c9b296c03ae7d3065614e80542566b350250","h2_hamiltonian.json":"89d07a971740b91f7f9848c4bded71ae47cb75b6c9eb226f8b94a97ff88ff226","lih_hamiltonian.json":"b698dca8a625fb3b43cc3a0d885c5c6d2b6710b817b94a563c125e3e85f7fa8e","beh2_hamiltonian.json":"2d3cbe1e95d8c815cb233ee9da383c039de1dc589b895605715181d6eb30642f","field_config.json":"7ac9f6d9972af95cecf1713cfeaa58905bc79b8ee1d6afb5e305671a4faabd7a"}
    public_text = {}
    for name, digest in expected_hashes.items():
        text = await session.read_file(f'{meta["input_dir"]}/{name}')
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != digest:
            raise RuntimeError(f"governed input hash mismatch: {name}")
        public_text[name] = text
    await session.interface.create_dir(meta["eval_dir"])
    staged_input = f'{meta["eval_dir"]}/input'
    staged_out = f'{meta["eval_dir"]}/out'
    await session.interface.create_dir(staged_input); await session.interface.create_dir(staged_out)
    for name in ("h2_hamiltonian.json","lih_hamiltonian.json","beh2_hamiltonian.json"):
        await session.write_file(f"{staged_input}/{name}", public_text[name])
    await session.write_file(f'{meta["eval_dir"]}/solve.py', solver_source)
    runner = '''import builtins,runpy,sys\n_allowed={"__future__","argparse","collections","functools","itertools","json","math","numpy","pathlib","re","scipy","typing"}\n_real=builtins.__import__\ndef guarded(name,*a,**k):\n    if sys._getframe(1).f_globals.get("__name__")=="__main__" and name.split(".")[0] not in _allowed: raise ImportError("import denied")\n    return _real(name,*a,**k)\nbuiltins.__import__=guarded\nrunpy.run_path("/work/solve.py",run_name="__main__")\n'''
    await session.write_file(f'{meta["eval_dir"]}/runner.py', runner)
    probe=(f'timeout 30s bwrap --unshare-all --die-with-parent --new-session --ro-bind /usr /usr --ro-bind /lib /lib '
           f'--ro-bind /lib64 /lib64 --proc /proc --dev /dev --tmpfs /tmp --ro-bind "{meta["eval_dir"]}" /work '
           f'--chdir /work /usr/bin/python3 -c \'import pathlib,sys; sys.exit(any(pathlib.Path(x).exists() for x in ["/reference","/task/reference","/work/../reference"]))\'')
    probe_result=await session.run_command(probe,check=False)
    if not isinstance(probe_result,dict) or probe_result.get("return_code",1)!=0:
        logger.info("filesystem traversal isolation probe failed: %r",probe_result); return [0.0]
    runtime_probe=(f'timeout 30s bwrap --unshare-all --die-with-parent --new-session --ro-bind /usr /usr --ro-bind /lib /lib '
                   f'--ro-bind /lib64 /lib64 --proc /proc --dev /dev --tmpfs /tmp --ro-bind "{meta["eval_dir"]}" /work '
                   f'--chdir /work /usr/bin/python3 -c \'import numpy,scipy,shutil,sys; assert sys.executable=="/usr/bin/python3"; assert numpy.__version__=="2.2.6"; assert scipy.__version__=="1.15.3"; assert shutil.which("bwrap")\'')
    runtime_result=await session.run_command(runtime_probe,check=False)
    if not isinstance(runtime_result,dict) or runtime_result.get("return_code",1)!=0:
        logger.info("immutable runtime probe failed: %r",runtime_result); return [0.0]
    # Exercise the submitted solver itself on malformed scientific inputs.
    # This prevents a solver from accepting malformed data while the fixture
    # alone rejects it.
    visible_config_obj = json.loads(public_text["field_config.json"])
    malformed_cases = []
    bad = dict(visible_config_obj); bad["identity_shift_ha"] = float("nan"); malformed_cases.append(("nonfinite", bad, None))
    bad = json.loads(public_text["field_config.json"]); bad["site_z_coefficients_ha"]["H2"] = bad["site_z_coefficients_ha"]["H2"][:-1]; malformed_cases.append(("wrong_vector", bad, None))
    bad = json.loads(public_text["field_config.json"]); bad["field_model"] = "wrong"; malformed_cases.append(("wrong_model", bad, None))
    bad = json.loads(public_text["field_config.json"]); bad["electron_sector_required"] = False; malformed_cases.append(("wrong_sector", bad, None))
    bad = json.loads(public_text["field_config.json"]); bad["site_z_coefficients_ha"]["H2"][0] = float("inf"); malformed_cases.append(("nonfinite_site_field", bad, None))
    bad = json.loads(public_text["field_config.json"]); bad["site_z_coefficients_ha"]["LiH"][0] = float("nan"); malformed_cases.append(("nan_site_field", bad, None))
    bad = json.loads(public_text["field_config.json"]); bad["hartree_to_ev"] = 0.0; malformed_cases.append(("zero_conversion", bad, None))
    bad = json.loads(public_text["field_config.json"]); bad["hartree_to_ev"] = -27.2; malformed_cases.append(("negative_conversion", bad, None))
    bad = json.loads(public_text["field_config.json"]); bad["hartree_to_ev"] = float("nan"); malformed_cases.append(("nan_conversion", bad, None))
    bad = json.loads(public_text["field_config.json"]); bad["hartree_to_ev"] = float("inf"); malformed_cases.append(("inf_conversion", bad, None))
    bad = json.loads(public_text["field_config.json"]); bad.pop("hartree_to_ev"); malformed_cases.append(("missing_conversion", bad, None))
    bad = json.loads(public_text["field_config.json"]); bad["unexpected_schema_field"] = 1; malformed_cases.append(("extra_config_schema", bad, None))
    # Exercise metadata and Pauli-word validation independently for every
    # required Hamiltonian, not just H2.  A solver that silently ignores LiH
    # or BeH2 corruption must fail the malformed-input gate as well.
    for molecule, filename, bad_word in (
        ("H2", "h2_hamiltonian.json", "QIII"),
        ("LiH", "lih_hamiltonian.json", "QIIIIIIIII"),
        ("BeH2", "beh2_hamiltonian.json", "QIIIIIIIII"),
    ):
        bad_ham = json.loads(public_text[filename]); bad_ham["n_pauli_terms"] += 1
        malformed_cases.append((f"bad_metadata_{molecule}", visible_config_obj, (filename, bad_ham)))
        bad_ham = json.loads(public_text[filename]); bad_ham["pauli_terms"][bad_word] = 0.1
        malformed_cases.append((f"bad_pauli_{molecule}", visible_config_obj, (filename, bad_ham)))
        bad_ham = json.loads(public_text[filename]); first_word = next(iter(bad_ham["pauli_terms"])); bad_ham["pauli_terms"][first_word] = float("nan")
        malformed_cases.append((f"nonfinite_pauli_nan_{molecule}", visible_config_obj, (filename, bad_ham)))
        bad_ham = json.loads(public_text[filename]); first_word = next(iter(bad_ham["pauli_terms"])); bad_ham["pauli_terms"][first_word] = float("inf")
        malformed_cases.append((f"nonfinite_pauli_inf_{molecule}", visible_config_obj, (filename, bad_ham)))
        bad_ham = json.loads(public_text[filename]); bad_ham["molecule"] = "WrongMolecule"
        malformed_cases.append((f"wrong_molecule_{molecule}", visible_config_obj, (filename, bad_ham)))
        bad_ham = json.loads(public_text[filename]); bad_ham.pop("basis")
        malformed_cases.append((f"missing_schema_field_{molecule}", visible_config_obj, (filename, bad_ham)))
        bad_ham = json.loads(public_text[filename]); bad_ham["n_qubits"] = 0
        malformed_cases.append((f"nonpositive_qubits_{molecule}", visible_config_obj, (filename, bad_ham)))
        bad_ham = json.loads(public_text[filename]); bad_ham["n_electrons"] = 0
        malformed_cases.append((f"nonpositive_electrons_{molecule}", visible_config_obj, (filename, bad_ham)))
        if molecule == "H2":
            bad_ham = json.loads(public_text[filename]); bad_ham["n_qubits"] = -1
            malformed_cases.append(("negative_qubits_H2", visible_config_obj, (filename, bad_ham)))
            bad_ham = json.loads(public_text[filename]); bad_ham["n_electrons"] = -1
            malformed_cases.append(("negative_electrons_H2", visible_config_obj, (filename, bad_ham)))
        bad_ham = json.loads(public_text[filename]); bad_ham["n_electrons"] = bad_ham["n_qubits"]
        malformed_cases.append((f"inconsistent_electron_sector_{molecule}", visible_config_obj, (filename, bad_ham)))
    malformed_command = (f'timeout 120s bwrap --unshare-all --die-with-parent --new-session --ro-bind /usr /usr --ro-bind /lib /lib '
                         f'--ro-bind /lib64 /lib64 --proc /proc --dev /dev --tmpfs /tmp --ro-bind "{meta["eval_dir"]}" /work '
                         f'--bind "{staged_out}" /work/out --chdir /work /usr/bin/python3 /work/runner.py --input-dir /work/input '
                         f'--field-config /work/input/field_config.json --output /work/out/malformed.json')
    for label, cfg_obj, ham_override in malformed_cases:
        for name in ("h2_hamiltonian.json", "lih_hamiltonian.json", "beh2_hamiltonian.json"):
            await session.write_file(f"{staged_input}/{name}", public_text[name])
        if ham_override is not None:
            await session.write_file(f"{staged_input}/{ham_override[0]}", json.dumps(ham_override[1]))
        await session.write_file(f"{staged_input}/field_config.json", json.dumps(cfg_obj, allow_nan=True))
        run_result = await session.run_command(malformed_command.replace("malformed.json", f"malformed_{label}.json"), check=False)
        if not isinstance(run_result, dict) or run_result.get("return_code", 1) == 0:
            logger.info("candidate accepted malformed input %s: %r", label, run_result); return [0.0]
        if await session.file_exists(f"{staged_out}/malformed_{label}.json"):
            logger.info("candidate emitted output for malformed input %s", label); return [0.0]
    for name in ("h2_hamiltonian.json", "lih_hamiltonian.json", "beh2_hamiltonian.json"):
        await session.write_file(f"{staged_input}/{name}", public_text[name])
    hidden_runs=[]
    for i,hidden_text in enumerate(hidden_config_texts,1):
        await session.write_file(f"{staged_input}/field_config.json", hidden_text)
        hidden_output=f"{staged_out}/results_{i}.json"
        command=(f'timeout 900s bwrap --unshare-all --die-with-parent --new-session --ro-bind /usr /usr --ro-bind /lib /lib '
                 f'--ro-bind /lib64 /lib64 --proc /proc --dev /dev --tmpfs /tmp --ro-bind "{meta["eval_dir"]}" /work '
                 f'--bind "{staged_out}" /work/out --chdir /work /usr/bin/python3 /work/runner.py '
                 f'--input-dir /work/input --field-config /work/input/field_config.json --output /work/out/results_{i}.json')
        run_result=await session.run_command(command,check=False)
        if not isinstance(run_result,dict) or run_result.get("return_code",1)!=0: logger.info("isolated hidden run failed: %r",run_result); return [0.0]
        payload=await _read_candidate_json(session,hidden_output)
        if payload is None:return [0.0]
        hidden_runs.append(payload)
    payloads={json.loads(public_text[name])["molecule"]:json.loads(public_text[name]) for name in ("h2_hamiltonian.json","lih_hamiltonian.json","beh2_hamiltonian.json")}
    context={"payloads":payloads,"configs":{"visible":json.loads(public_text["field_config.json"]),**{f"hidden_{i+1}":json.loads(x) for i,x in enumerate(hidden_config_texts)}}}
    try:
        result = SCORER.score_payloads(visible, hidden_runs, reference, context)
        score = float(result["score"])
    except Exception as exc:
        logger.info("candidate scoring failed: %s", exc)
        return [0.0]
    logger.info("evaluation=%s", json.dumps(result, sort_keys=True))
    return [max(0.0, min(1.0, score))]
