"""ALE task hooks for the Lenacapavir Table 2 extraction benchmark."""
from __future__ import annotations
import importlib.util
import json, shutil, sys, tempfile
from dataclasses import dataclass
from pathlib import Path
try:
    import cua_bench as cb
    from tasks.common_setup import BaseTaskSetup
    from tasks.linux_runtime import LinuxTaskConfig
except ImportError:
    class _CB:
        class Task:
            def __init__(self, **kw): self.__dict__.update(kw)
        @staticmethod
        def tasks_config(split="train"): return lambda f:f
        @staticmethod
        def setup_task(split="train"): return lambda f:f
        @staticmethod
        def evaluate_task(split="train"): return lambda f:f
    cb=_CB()
    class LinuxTaskConfig: pass
    class BaseTaskSetup:
        async def __call__(self,*a,**k): return None

ROOT=Path(__file__).resolve().parent

def _load_script(name):
    module_name=f"{__name__}._{name}"
    spec=importlib.util.spec_from_file_location(module_name,ROOT/"scripts"/f"{name}.py")
    if spec is None or spec.loader is None: raise ImportError(f"cannot load task script: {name}")
    module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module;spec.loader.exec_module(module)
    return module

@dataclass
class Config(LinuxTaskConfig):
    DOMAIN_NAME: str="physical_sciences"; TASK_NAME: str="ale_anchor_lenacapavir_sar_table2_extraction_lenacapavir_sar_table2_extraction"; VARIANT_NAME: str="table2"
    def to_metadata(self):
        if hasattr(super(), "to_metadata"):
            metadata = super().to_metadata()
        else:
            task_dir = f"base/{self.VARIANT_NAME}"
            metadata = {"task_dir": task_dir, "input_dir": f"{task_dir}/input", "software_dir": f"{task_dir}/software", "remote_output_dir": f"{task_dir}/output", "reference_dir": f"{task_dir}/reference"}
        metadata.update({"task_id":f"physical_sciences/{self.TASK_NAME}","output_dir":metadata["remote_output_dir"]})
        return metadata
    @property
    def task_description(self):
        return json.loads(Path(__file__).with_name("task_card.json").read_text(encoding="utf-8"))["taskPrompt"]
config=Config(); _setup=BaseTaskSetup()
@cb.tasks_config(split="train")
def load(): return [cb.Task(description=config.task_description,metadata=config.to_metadata(),computer={"provider":"computer","setup_config":{"os_type":"linux"}})]
@cb.setup_task(split="train")
async def start(task_cfg, session):
    await _setup(task_cfg, session)
    await session.interface.create_dir(task_cfg.metadata["remote_output_dir"])
@cb.evaluate_task(split="train")
async def evaluate(task_cfg,session):
    m=task_cfg.metadata
    try:
        with tempfile.TemporaryDirectory(prefix="lenacapavir_eval_") as td:
            t=Path(td); out=t/"output"; ref=t/"reference"; out.mkdir();ref.mkdir()
            for name in ("submission.csv","analysis.json","analyze_table.py"):
                remote=f"{m['output_dir']}/{name}"
                if await session.file_exists(remote): (out/name).write_bytes(await session.read_bytes(remote))
            for name in ("reference.csv","reference_analysis.json","hidden_table.csv","hidden_analysis.json"):
                (ref/name).write_bytes(await session.read_bytes(f"{m['reference_dir']}/{name}"))
            score=_load_script("score_outputs").score
            return [float(score(out,ref).get("score",0.0))]
    except Exception:
        return [0.0]
