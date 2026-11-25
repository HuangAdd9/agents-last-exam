"""ALE task: EGT710 Table 1 reconstruction with unit/audit transformation."""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig
if __name__ not in sys.modules:
    sys.modules[__name__] = sys.modules.get(__name__, type(sys)(__name__))

logger = logging.getLogger(__name__)
_setup = BaseTaskSetup()
VARIANTS = [("table1", "Table 1 compounds 1–9; transformed assay and audit contract")]
ROOT = Path(__file__).resolve().parent


def _load_script(name: str):
    package_name = f"{__name__}._task_scripts"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(ROOT / "scripts")]
        sys.modules[package_name] = package
    module_name = f"{package_name}.{name}"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "scripts" / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load task script: {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


verify = _load_script("verify_submission").verify


@dataclass
class EGT710Config(LinuxTaskConfig):
    DOMAIN_NAME: str = "physical_sciences"
    TASK_NAME: str = "ale_anchor_egt710_table1_smiles_extraction_egt710_table1_smiles_extraction"
    VARIANT_NAME: str = "table1"
    VARIANT_LABEL: str = ""

    @property
    def input_pdf(self) -> str:
        return f"{self.input_dir}/source_paper.pdf"

    @property
    def output_csv(self) -> str:
        return f"{self.remote_output_dir}/submission.csv"

    @property
    def task_description(self):
        return json.loads(Path(__file__).with_name("task_card.json").read_text(encoding="utf-8"))["taskPrompt"]

    def to_metadata(self) -> dict:
        m = super().to_metadata()
        m.update({"task_id": f"{self.DOMAIN_NAME}/{self.TASK_NAME}", "variant_label": self.VARIANT_LABEL, "input_pdf": self.input_pdf, "output_csv": self.output_csv})
        return m


@cb.tasks_config(split="train")
def load():
    return [cb.Task(description=EGT710Config(VARIANT_LABEL=label).task_description,
                    metadata=EGT710Config(VARIANT_LABEL=label).to_metadata(),
                    computer={"provider": "computer", "setup_config": {"os_type": "linux"}})
            for _, label in VARIANTS]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    names = ["submission.csv", "audit.json", "transform.py"]
    with tempfile.TemporaryDirectory(prefix="egt710_eval_") as td:
        root = Path(td); out = root / "output"; out.mkdir()
        for name in names:
            p = f'{meta["remote_output_dir"]}/{name}'
            if not await session.file_exists(p):
                return [0.0]
            (out / name).write_bytes(await session.read_bytes(p))
        ref_dir = root / "reference"; ref_dir.mkdir()
        for name in ["reference.csv", "reference_audit.json", "hidden_transform_case.json"]:
            p = f'{meta["reference_dir"]}/{name}'
            if not await session.file_exists(p):
                return [0.0]
            (ref_dir / name).write_bytes(await session.read_bytes(p))
        try:
            payload = verify(out / "submission.csv", ref_dir / "reference.csv", ref_dir / "reference_audit.json", ref_dir / "hidden_transform_case.json", out)
        except Exception:
            return [0.0]
    logger.info("EGT710 evaluation %s", json.dumps(payload, sort_keys=True))
    return [float(payload.get("score", 0.0))]
