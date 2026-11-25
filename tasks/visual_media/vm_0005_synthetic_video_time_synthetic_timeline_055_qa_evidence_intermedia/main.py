"""ALE task: diagnose and repair a contaminated exposure-test ingest."""
from __future__ import annotations
import json
import logging
import os
import posixpath
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
import cua_bench as cb
from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path: sys.path.insert(0, str(SCRIPTS_DIR))
from score_exposure_report import score_package

_setup = BaseTaskSetup(); logger = logging.getLogger(__name__)
EVALUATOR_PYTHON = os.environ.get("ALE_EVALUATOR_PYTHON", "/opt/ale-run/.venv/bin/python")
DOMAIN_NAME = "visual_media"
TASK_NAME = "vm_0005_synthetic_video_time_synthetic_timeline_055_qa_evidence_intermedia"
TASK_ID = f"{DOMAIN_NAME}/{TASK_NAME}"; VARIANT_NAME = "base"

@dataclass
class ExposureTimelineConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN_NAME; TASK_NAME: str = TASK_NAME; VARIANT_NAME: str = VARIANT_NAME
    @property
    def output_report(self) -> str: return f"{self.remote_output_dir}/exposure_report.json"
    @property
    def output_continuity(self) -> str: return f"{self.remote_output_dir}/continuity_log.csv"
    @property
    def output_trace(self) -> str: return f"{self.remote_output_dir}/frame_trace.csv"
    @property
    def output_video(self) -> str: return f"{self.remote_output_dir}/stabilized.mp4"
    @property
    def output_recipe(self) -> str: return f"{self.remote_output_dir}/repair_recipe.json"
    @property
    def probe_result(self) -> str: return f"{self.remote_output_dir}/video_probe.json"
    @property
    def probe_script(self) -> str: return f"{self.software_dir}/probe_stabilized_video.py"
    @property
    def reference_events(self) -> str: return f"{self.reference_dir}/reference_event_table.json"
    @property
    def reference_metrics(self) -> str: return f"{self.reference_dir}/reference_media_metrics.json"
    @property
    def expected_media(self) -> str: return f"{self.reference_dir}/expected_media.json"
    @property
    def reference_inventory(self) -> str: return f"{self.reference_dir}/reference_inventory.json"
    @property
    def task_description(self) -> str:
        return f"""\
You are given a messy camera ingest for a restoration QC handoff. Work as the finishing
technician: identify the usable full-resolution source among the delivered files, inspect
its stream and pixels, and produce a small, self-contained QC package under
`{self.remote_output_dir}`.

The delivery is intentionally not arranged as a tutorial. There are incomplete proxies,
review grabs, notes, and multiple differently encoded media files in nested folders under `{self.input_dir}`.
Do not trust filenames or the order of the files. Inspect the media and establish the
actual frame rate, dimensions, duration, and frame count before processing.

Your package must contain:
- `exposure_report.json`: a concise temporal report with media facts, chronological
  intervals, relative exposure levels, classifications, timestamped measured brightness
  evidence, and observations about the persistent composition.
- `continuity_log.csv`: one row for each interval, synchronized to the JSON IDs and
  times, documenting the uninterrupted framing and evidence timestamps.
- `frame_trace.csv`: one row for every decoded frame of the selected full-resolution
  source, with frame index, time in seconds, a repeatable brightness statistic from a
  fixed image patch, and its log2 relative value. State the measurement convention in
  a header comment or an accompanying field.
- `stabilized.mp4`: a full-length 24 fps H.264 video with the same dimensions and frame
  count as the selected source. Correct the global exposure drift so the chosen dark
  reference region remains approximately constant while preserving the neutral timecode
  strip and the scene geometry. Do not crop, resize, add text, or replace the scene.
- `repair_recipe.json`: the selected relative input path and hash, a conform inventory of
  every candidate media file with accept/reject reasons, stream facts, the measurement
  convention, correction method and target, and output hash.

Use the supplied media only. The source contains a single continuous fixed composition;
report what is visible and measurable rather than inventing a narrative. Keep all five
artifacts in the output directory and do not alter input files.

Input root: `{self.input_dir}`
Output root: `{self.remote_output_dir}`
Available runtime tools include FFmpeg/ffprobe, Python, Pillow, NumPy and OpenCV.
"""
    def to_metadata(self) -> dict:
        m = super().to_metadata(); m.update({"task_id":TASK_ID,
            "output_report":self.output_report,"output_continuity":self.output_continuity,
            "output_trace":self.output_trace,"output_video":self.output_video,"output_recipe":self.output_recipe,"probe_result":self.probe_result,
            "probe_script":self.probe_script,"reference_events":self.reference_events,"reference_metrics":self.reference_metrics,
            "expected_media":self.expected_media,"reference_inventory":self.reference_inventory})
        return m

config = ExposureTimelineConfig()
@cb.tasks_config(split="train")
def load():
    cfg=ExposureTimelineConfig()
    return [cb.Task(description=cfg.task_description,metadata=cfg.to_metadata(),computer={"provider":"computer","setup_config":{"os_type":"linux"}})]
@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession): await _setup(task_cfg, session)
@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta=task_cfg.metadata
    try:
        async def optional_bytes(key):
            path=meta[key]
            return await session.read_bytes(path) if await session.file_exists(path) else b""
        report=await optional_bytes("output_report"); continuity=await optional_bytes("output_continuity")
        trace=await optional_bytes("output_trace"); recipe_bytes=await optional_bytes("output_recipe")
        try: recipe=json.loads(recipe_bytes)
        except (ValueError,TypeError): recipe={}
        if not isinstance(recipe,dict): recipe={}
        inventory=json.loads(await session.read_file(meta["reference_inventory"]))
        default_source=next(row["path"] for row in inventory if row.get("decision")=="selected")
        rel_source=recipe.get("source_path") or default_source
        normalized=posixpath.normpath(str(rel_source))
        valid_source=normalized and not normalized.startswith("/") and normalized != ".." and not normalized.startswith("../") and re.fullmatch(r"[A-Za-z0-9_./ -]+",normalized)
        chosen_source=f'{meta["input_dir"]}/{normalized}' if valid_source else ""
        probe={"ok":False,"error":"missing video or selected source"}
        if chosen_source and await session.file_exists(chosen_source) and await session.file_exists(meta["output_video"]):
            measurement=recipe.get("measurement",{})
            patch=measurement.get("patch_xyxy",[82,150,112,176]) if isinstance(measurement,dict) else [82,150,112,176]
            if not isinstance(patch,list) or len(patch)!=4 or any(not isinstance(x,int) or isinstance(x,bool) for x in patch):
                patch=[82,150,112,176]
            command=(f'{shlex.quote(EVALUATOR_PYTHON)} "{meta["probe_script"]}" --source "{chosen_source}" '
                     f'--output "{meta["output_video"]}" --result "{meta["probe_result"]}" '
                     f'--patch {",".join(map(str,patch))}')
            await session.run_command(command,check=True)
            probe=json.loads(await session.read_file(meta["probe_result"]))
        refs=[await session.read_bytes(meta[k]) for k in ("reference_events","reference_metrics","expected_media","reference_inventory")]
        result=score_package(report,continuity,trace,recipe_bytes,*refs,probe)
    except Exception as exc:
        logger.exception("evaluation failed: %s",exc); return [0.0]
    logger.info("evaluation=%s",json.dumps(result,sort_keys=True)); return [float(result.get("score",0.0))]
