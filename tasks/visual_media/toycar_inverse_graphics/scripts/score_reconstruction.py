"""Topology-independent score for the Toy Car multiview reconstruction."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def directed_nearest(source: np.ndarray, target: np.ndarray, chunk: int = 128) -> np.ndarray:
    output = np.empty(len(source), dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    for start in range(0, len(source), chunk):
        block = np.asarray(source[start : start + chunk], dtype=np.float64)
        distances = ((block[:, None, :] - target[None, :, :]) ** 2).sum(axis=2)
        output[start : start + len(block)] = np.sqrt(distances.min(axis=1))
    return output


def surface_metrics(reference: np.ndarray, candidate: np.ndarray, diagonal: float) -> dict:
    ref_to_candidate = directed_nearest(reference, candidate)
    candidate_to_ref = directed_nearest(candidate, reference)
    scale = max(float(diagonal), 1e-9)
    mean = 0.5 * (float(ref_to_candidate.mean()) + float(candidate_to_ref.mean())) / scale
    p90 = 0.5 * (
        float(np.quantile(ref_to_candidate, 0.90)) + float(np.quantile(candidate_to_ref, 0.90))
    ) / scale
    score = math.exp(-36.0 * mean - 7.0 * p90)
    return {
        "reference_to_candidate_mean": float(ref_to_candidate.mean()),
        "candidate_to_reference_mean": float(candidate_to_ref.mean()),
        "normalized_symmetric_mean": mean,
        "normalized_symmetric_p90": p90,
        "score": clamp01(score),
    }


def bbox_metrics(reference: dict, candidate: dict) -> dict:
    ref_center = np.asarray(reference["bbox_center"], dtype=np.float64)
    ref_extent = np.asarray(reference["bbox_extent"], dtype=np.float64)
    cand_center = np.asarray(candidate["bbox_center"], dtype=np.float64)
    cand_extent = np.asarray(candidate["bbox_extent"], dtype=np.float64)
    diagonal = max(float(reference["bbox_diagonal"]), 1e-9)
    center_error = float(np.linalg.norm(cand_center - ref_center) / diagonal)
    log_extent_error = float(np.mean(np.abs(np.log(np.maximum(cand_extent, 1e-9) / np.maximum(ref_extent, 1e-9)))))
    score = math.exp(-12.0 * center_error - 2.8 * log_extent_error)
    return {
        "normalized_center_error": center_error,
        "mean_absolute_log_extent_error": log_extent_error,
        "extent_ratios": (cand_extent / np.maximum(ref_extent, 1e-9)).tolist(),
        "score": clamp01(score),
    }


def alpha_mask(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
    return alpha >= 16


def silhouette_metrics(contract: dict, reference_dir: Path, candidate_dir: Path) -> dict:
    per_view = {}
    scores = []
    for spec in contract["hidden_views"]:
        filename = spec["mask_file"]
        ref = alpha_mask(reference_dir / "hidden_masks" / filename)
        cand = alpha_mask(candidate_dir / filename)
        if ref.shape != cand.shape:
            raise RuntimeError(f"silhouette shape mismatch for {filename}: {ref.shape} vs {cand.shape}")
        intersection = int(np.logical_and(ref, cand).sum())
        union = int(np.logical_or(ref, cand).sum())
        iou = float(intersection / union) if union else 0.0
        coverage = float(cand.sum() / max(int(ref.sum()), 1))
        score = iou * math.exp(-0.35 * abs(math.log(max(coverage, 1e-6))))
        per_view[spec["name"]] = {"iou": iou, "coverage_ratio": coverage, "score": score}
        scores.append(score)
    return {"per_view": per_view, "score": float(sum(scores) / len(scores)) if scores else 0.0}


def run_blender(args: argparse.Namespace, contract: dict) -> tuple[dict, np.ndarray, Path, dict]:
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    points = work / "candidate_surface.npz"
    metrics = work / "candidate_metrics.json"
    renders = work / "candidate_hidden_masks"
    renders.mkdir(parents=True, exist_ok=True)
    blender = shutil.which("blender")
    if blender is not None:
        command = [
            blender, "--background", "--python", args.blender_helper, "--",
            "--model", args.candidate, "--contract", args.contract,
            "--points", str(points), "--metrics", str(metrics), "--render-dir", str(renders),
        ]
    elif shutil.which("docker") is not None:
        # The author-side ALE adapter smoke executes on a host without Blender.
        # Mount only the smoke workspace into the exact assigned runtime; real task
        # VMs take the direct Blender branch above.
        paths = [
            Path(args.candidate).resolve(), Path(args.contract).resolve(),
            Path(args.reference_points).resolve(), Path(args.reference_dir).resolve(),
            Path(args.blender_helper).resolve(), work.resolve(),
        ]
        common = Path(os.path.commonpath([str(path) for path in paths]))
        if common.is_file():
            common = common.parent

        def bridged(value: str | Path) -> str:
            return str(Path("/bridge") / Path(value).resolve().relative_to(common))

        command = [
            "docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
            "-e", "XDG_CONFIG_HOME=/tmp/ale-blender-config",
            "-v", f"{common}:/bridge", "-w", "/bridge",
            "ale/base-visual-media:20260918-1",
            "blender", "--background", "--python", bridged(args.blender_helper), "--",
            "--model", bridged(args.candidate), "--contract", bridged(args.contract),
            "--points", bridged(points), "--metrics", bridged(metrics),
            "--render-dir", bridged(renders),
        ]
    else:
        raise RuntimeError("Neither Blender nor the assigned Docker runtime is available")
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240, check=False)
    execution = {
        "command": command,
        "exit_code": completed.returncode,
        "stdout_tail": completed.stdout[-3000:],
        "stderr_tail": completed.stderr[-3000:],
    }
    if completed.returncode or not metrics.is_file() or not points.is_file():
        raise RuntimeError(f"Blender measurement failed: {execution}")
    return json.loads(metrics.read_text(encoding="utf-8")), np.load(points)["points"], renders, execution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report")
    parser.add_argument("--contract", required=True)
    parser.add_argument("--reference-points", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--blender-helper", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--result", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_path = Path(args.result)
    try:
        contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
        candidate_metrics, candidate_points, candidate_renders, execution = run_blender(args, contract)
        reference_points = np.load(args.reference_points)["points"]
        bbox = bbox_metrics(contract["reference_geometry"], candidate_metrics)
        surface = surface_metrics(
            reference_points, candidate_points, float(contract["reference_geometry"]["bbox_diagonal"])
        )
        silhouettes = silhouette_metrics(contract, Path(args.reference_dir), candidate_renders)
        triangle_sanity = clamp01(candidate_metrics["triangle_count"] / 500.0)
        report_path = Path(args.report) if args.report else None
        report_ok = 1.0 if report_path and report_path.is_file() and len(report_path.read_text(encoding="utf-8", errors="ignore").split()) >= 12 else 0.0
        geometry_score = 0.47 * surface["score"] + 0.18 * bbox["score"] + 0.28 * silhouettes["score"] + 0.05 * triangle_sanity + 0.02 * report_ok
        payload = {
            "score": clamp01(geometry_score),
            "components": {
                "surface": {"weight": 0.47, **surface},
                "bbox": {"weight": 0.18, **bbox},
                "hidden_silhouettes": {"weight": 0.28, **silhouettes},
                "valid_geometry": {"weight": 0.05, "triangle_sanity": triangle_sanity},
                "report": {"weight": 0.02, "present_and_nontrivial": bool(report_ok)},
            },
            "candidate_metrics": candidate_metrics,
            "blender_execution": execution,
        }
    except Exception as exc:
        payload = {"score": 0.0, "error": f"{type(exc).__name__}: {exc}"}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
