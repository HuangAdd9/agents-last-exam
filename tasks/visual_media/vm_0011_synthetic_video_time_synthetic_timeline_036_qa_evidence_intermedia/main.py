"""ALE task: reconstruct and document timeline 036 from a mixed ingest."""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import cua_bench as cb

from tasks.common_setup import BaseTaskSetup
from tasks.linux_runtime import LinuxTaskConfig


SCRIPTS = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from score_timeline import score_submission  # noqa: E402


DOMAIN = "visual_media"
NAME = "vm_0011_synthetic_video_time_synthetic_timeline_036_qa_evidence_intermedia"
logger = logging.getLogger(__name__)
_setup = BaseTaskSetup()
MEDIA_SCORER = SCRIPTS / "score_media.py"


@dataclass
class TimelineConfig(LinuxTaskConfig):
    DOMAIN_NAME: str = DOMAIN
    TASK_NAME: str = NAME
    VARIANT_NAME: str = "base"
    OS_TYPE: str = "linux"

    @property
    def task_description(self) -> str:
        return f"""\
Conform the clean original sequence from an uncurated media handoff, then prepare a frame-accurate edit record and evidence report. Determine the usable frames and their order from the media.

Inputs:
- Ingest folder: `{self.input_dir}/ingest`
- QC questions: `{self.input_dir}/questions.json`

Available tools include FFplay, FFmpeg/FFprobe, and Python 3.

Deliver a 640 × 360, 15 fps, 180-frame H.264 MP4. Each output frame must be the corresponding clean source frame, with original picture area and color. Do not blend, repeat, omit, crop, or add frames. Use zero-based, half-open frame intervals throughout.

Save four files under `{self.remote_output_dir}`:

1. `rebuilt.mp4`: the reconstructed video described above.
2. `assembly.csv`: columns `timeline_in,timeline_out,asset,asset_in`. Each row maps a contiguous source range to a timeline interval; `asset` is relative to the ingest folder and `asset_in` is its first local frame. Cover frames 0–179 exactly once and describe the delivered edit.
3. `shot_log.json`: UTF-8 JSON with integer `fps`, integer `total_frames`, and a chronological `shots` array. Each shot has integer `id`, `start_frame`, `end_frame`, string `shot_type`, two-integer array `evidence_frames`, string `description`, and string array `visible_details`. Use standard framing terms and observable details; cite two interior frames at least eight frames apart per shot.
4. `qa_answers.json`: UTF-8 JSON with an `answers` array; each entry has `id`, `answer`, `shot_id`, and `evidence_frame`. Answer every question in `questions.json` from the delivered timeline. Cite the incoming frame for a cut answer, use an integer for a shot-ID answer, and transcribe visible text accurately.

The current working directory may differ from the task directory. Create `{self.remote_output_dir}` if needed.
"""


config = TimelineConfig()


@cb.tasks_config(split="train")
def load():
    return [
        cb.Task(
            description=config.task_description,
            metadata=config.to_metadata(),
            computer={"provider": "computer", "setup_config": {"os_type": "linux"}},
        )
    ]


@cb.setup_task(split="train")
async def start(task_cfg, session: cb.DesktopSession):
    await _setup(task_cfg, session)


@cb.evaluate_task(split="train")
async def evaluate(task_cfg, session: cb.DesktopSession) -> list[float]:
    meta = task_cfg.metadata
    gold_path = f"{meta['reference_dir']}/gold_solution.json"
    if not await session.file_exists(gold_path):
        raise RuntimeError(f"Missing evaluator reference: {gold_path}")
    gold_bytes = await session.read_bytes(gold_path)
    output_paths = {
        "shot_log": f"{meta['remote_output_dir']}/shot_log.json",
        "qa_answers": f"{meta['remote_output_dir']}/qa_answers.json",
    }
    outputs = {}
    for label, path in output_paths.items():
        outputs[label] = await session.read_bytes(path) if await session.file_exists(path) else None
    text_result = score_submission(outputs["shot_log"], outputs["qa_answers"], gold_bytes)

    media_result = {"video": 0.0, "assembly": 0.0}
    video_path = f"{meta['remote_output_dir']}/rebuilt.mp4"
    assembly_path = f"{meta['remote_output_dir']}/assembly.csv"
    if await session.file_exists(video_path) or await session.file_exists(assembly_path):
        for name in ("timeline_036.mp4", "source_map.json"):
            path = f"{meta['reference_dir']}/{name}"
            if not await session.file_exists(path):
                raise RuntimeError(f"Missing evaluator reference: {path}")
        eval_dir = "/tmp/agenthle_timeline_036_eval"
        await session.interface.create_dir(eval_dir)
        script_path = f"{eval_dir}/score_media.py"
        report_path = f"{eval_dir}/report.json"
        await session.write_file(script_path, MEDIA_SCORER.read_text(encoding="utf-8"))
        args = [
            # The task image's helper venv has NumPy but no OpenCV.  The image
            # system interpreter carries the bundled cv2 package used by this
            # media scorer.
            "/usr/bin/python3", "-B", script_path,
            "--input-dir", f"{meta['input_dir']}/ingest",
            "--reference-video", f"{meta['reference_dir']}/timeline_036.mp4",
            "--source-map", f"{meta['reference_dir']}/source_map.json",
            "--output-video", video_path,
            "--assembly", assembly_path,
            "--report", report_path,
        ]
        result = await asyncio.wait_for(session.run_command(shlex.join(args), check=False), timeout=120)
        if result.get("exit_code", 0) != 0 or not await session.file_exists(report_path):
            raise RuntimeError(f"Media scorer failed: {result.get('stderr', '')[:1000]}")
        media_result = json.loads(await session.read_file(report_path))

    score = (0.35 * text_result["score"] + 0.45 * media_result["video"]
             + 0.20 * media_result["assembly"])
    logger.info("timeline evaluation: %s", json.dumps({"text": text_result, "media": media_result, "score": score}, sort_keys=True))
    return [float(score)]
