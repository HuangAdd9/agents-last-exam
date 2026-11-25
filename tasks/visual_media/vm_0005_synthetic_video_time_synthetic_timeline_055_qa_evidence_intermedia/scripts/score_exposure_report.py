"""Deterministic evaluator for the exposure timeline evidence package."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from typing import Any


def _num(value: Any) -> float | None:
    try:
        if isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _direction(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    if text in {"stable", "hold", "constant", "no change", "unchanged"}:
        return "stable"
    if "bright" in text or "up" in text or "increase" in text:
        return "brightening"
    if "dark" in text or "down" in text or "decrease" in text:
        return "darkening"
    if "return" in text or "baseline" in text or "recover" in text:
        return "return"
    return text


def _expected_mean(checkpoints: list[dict[str, Any]], time_s: float) -> float | None:
    points = [(float(p["time_s"]), float(p["mean_rgb"])) for p in checkpoints]
    if not points:
        return None
    if time_s <= points[0][0]:
        return points[0][1]
    if time_s >= points[-1][0]:
        return points[-1][1]
    for (ta, va), (tb, vb) in zip(points, points[1:]):
        if ta <= time_s <= tb:
            alpha = (time_s - ta) / (tb - ta)
            return va + alpha * (vb - va)
    return None


def _parse_evidence_values(value: Any) -> list[float]:
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        raw = [x.strip() for x in value.replace("|", ",").split(",") if x.strip()]
    else:
        raw = []
    out = []
    for item in raw:
        number = _num(item)
        if number is not None:
            out.append(number)
    return out


def _load_csv(data: bytes) -> list[dict[str, str]]:
    text = data.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    return [dict(row) for row in reader]


def _match_segments(candidates: list[dict[str, Any]], refs: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    remaining = list(candidates)
    matched: list[dict[str, Any] | None] = []
    for ref in refs:
        rs = float(ref["start_s"])
        re = float(ref["end_s"])
        if not remaining:
            matched.append(None)
            continue
        def distance(candidate: dict[str, Any]) -> float:
            cs = _num(candidate.get("start_s"))
            ce = _num(candidate.get("end_s"))
            return abs((cs if cs is not None else 999.0) - rs) + abs((ce if ce is not None else 999.0) - re)
        best = min(remaining, key=distance)
        remaining.remove(best)
        matched.append(best)
    return matched


def score_submission(
    report_bytes: bytes,
    continuity_bytes: bytes,
    reference_event_bytes: bytes,
    reference_metrics_bytes: bytes,
    expected_media_bytes: bytes,
    probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    try:
        report = json.loads(report_bytes.decode("utf-8"))
    except Exception as exc:
        return {"score": 0.0, "error": f"invalid exposure_report.json: {exc}"}
    if not isinstance(report, dict):
        return {"score": 0.0, "error": "exposure_report.json must be an object"}
    try:
        ref = json.loads(reference_event_bytes.decode("utf-8"))
        metrics = json.loads(reference_metrics_bytes.decode("utf-8"))
        expected_media = json.loads(expected_media_bytes.decode("utf-8"))
    except Exception as exc:
        return {"score": 0.0, "error": f"invalid evaluator reference: {exc}"}

    events = ref.get("events", [])
    checkpoints = metrics.get("checkpoints", [])
    if probe and probe.get("ok") and probe.get("source_sha256") == expected_media.get("source_sha256"):
        checkpoints = [{"time_s": row["time_s"], "mean_rgb": row["source_frame_luma"]}
                       for row in probe.get("frames", [])]
    # 0.12: media manifest and schema.
    media = report.get("media") if isinstance(report, dict) else None
    manifest_scores: list[float] = []
    if isinstance(report, dict) and report.get("schema_version") == 1:
        manifest_scores.append(1.0)
    else:
        manifest_scores.append(0.0)
    if isinstance(media, dict):
        for key, tol in (("duration_s", 0.15), ("fps", 0.15), ("width", 1.0), ("height", 1.0)):
            got, want = _num(media.get(key)), _num(expected_media.get(key))
            manifest_scores.append(1.0 if got is not None and want is not None and abs(got - want) <= tol else 0.0)
        manifest_scores.append(1.0 if str(media.get("filename", "")) == expected_media.get("filename") else 0.0)
        manifest_scores.append(1.0 if str(media.get("audio", "none")).lower() in {"none", "no", "0"} else 0.0)
    else:
        manifest_scores.extend([0.0] * 6)
    manifest_score = _mean(manifest_scores)
    details["manifest"] = manifest_score

    candidates = report.get("segments", []) if isinstance(report, dict) else []
    if not isinstance(candidates, list):
        candidates = []
    candidates = [c for c in candidates if isinstance(c, dict)]
    matched = _match_segments(candidates, events)

    boundary_scores: list[float] = []
    kind_scores: list[float] = []
    level_scores: list[float] = []
    evidence_timestamp_scores: list[float] = []
    evidence_metric_scores: list[float] = []
    segment_rows: list[dict[str, Any]] = []
    for event, candidate in zip(events, matched):
        if candidate is None:
            boundary_scores.append(0.0); kind_scores.append(0.0); level_scores.append(0.0)
            evidence_timestamp_scores.append(0.0); evidence_metric_scores.append(0.0)
            segment_rows.append({"matched": False})
            continue
        start, end = _num(candidate.get("start_s")), _num(candidate.get("end_s"))
        if start is None or end is None or end <= start:
            boundary_scores.append(0.0)
        else:
            boundary_scores.append(_mean([_clamp(1.0 - abs(start - float(event["start_s"])) / 0.5), _clamp(1.0 - abs(end - float(event["end_s"])) / 0.5)]))
        cand_kind = str(candidate.get("kind", "")).strip().lower()
        kind_scores.append(1.0 if cand_kind == str(event.get("kind", "")).lower() else 0.0)
        ls, le = _num(candidate.get("level_start")), _num(candidate.get("level_end"))
        if ls is None or le is None:
            level_scores.append(0.0)
        else:
            level_scores.append(_mean([_clamp(1.0 - abs(ls - float(event["ev_start"])) / 1.0), _clamp(1.0 - abs(le - float(event["ev_end"])) / 1.0)]))

        timestamps = _parse_evidence_values(candidate.get("evidence_timestamps"))
        if start is None or end is None or not timestamps:
            evidence_timestamp_scores.append(0.0)
        else:
            inside = [t for t in timestamps if start - 0.05 <= t <= end + 0.05]
            evidence_timestamp_scores.append(_clamp(len(inside) / 2.0))
        means = _parse_evidence_values(candidate.get("evidence_mean_rgb"))
        if not timestamps or not means:
            evidence_metric_scores.append(0.0)
        else:
            pairs = min(len(timestamps), len(means))
            close = []
            for t, observed in zip(timestamps[:pairs], means[:pairs]):
                expected = _expected_mean(checkpoints, t)
                close.append(_clamp(1.0 - max(0.0, abs(observed - expected) - 4.0) / 12.0) if expected is not None else 0.0)
            evidence_metric_scores.append(_mean(close))
        segment_rows.append({"matched": True, "id": candidate.get("id"), "start": start, "end": end})

    boundary_score = _mean(boundary_scores)
    kind_score = _mean(kind_scores)
    level_score = _mean(level_scores)
    timestamp_score = _mean(evidence_timestamp_scores)
    metric_score = _mean(evidence_metric_scores)
    details.update({"boundaries": boundary_score, "event_kind": kind_score, "relative_levels": level_score, "evidence_timestamps": timestamp_score, "evidence_metrics": metric_score, "candidate_segment_count": len(candidates)})

    # Direction and continuity semantics are separate from hold/ramp labels.
    direction_scores = []
    for event, candidate in zip(events, matched):
        if candidate is None:
            direction_scores.append(0.0); continue
        expected_start, expected_end = float(event["ev_start"]), float(event["ev_end"])
        expected_direction = "stable" if event["kind"] == "hold" and expected_start == expected_end else ("return" if float(event.get("start_s", 0)) >= 10.0 else ("brightening" if expected_end > expected_start else "darkening"))
        got = _direction(candidate.get("direction"))
        # A baseline recovery can be called return or brightening; it must not be called darkening.
        ok = got == expected_direction or (expected_direction == "return" and got == "brightening")
        direction_scores.append(1.0 if ok else 0.0)
    direction_score = _mean(direction_scores)
    details["direction"] = direction_score

    try:
        rows = _load_csv(continuity_bytes)
    except Exception:
        rows = []
    row_by_id = {str(row.get("segment_id", "")).strip(): row for row in rows if row.get("segment_id")}
    continuity_scores = []
    cross_scores = []
    seen_shots = set()
    for event, candidate in zip(events, matched):
        if candidate is None:
            continuity_scores.append(0.0); cross_scores.append(0.0); continue
        seg_id = str(candidate.get("id", "")).strip()
        row = row_by_id.get(seg_id)
        if not row:
            continuity_scores.append(0.0); cross_scores.append(0.0); continue
        shot_id = str(row.get("shot_id", "")).strip()
        seen_shots.add(shot_id)
        framing = str(row.get("framing", "")).lower()
        obs = str(row.get("continuity_observation", "")).lower()
        row_start, row_end = _num(row.get("start_s")), _num(row.get("end_s"))
        c_start, c_end = _num(candidate.get("start_s")), _num(candidate.get("end_s"))
        row_score = _mean([
            1.0 if shot_id else 0.0,
            1.0 if any(word in framing for word in ("fixed", "static", "single", "locked")) else 0.0,
            1.0 if any(word in obs for word in ("same", "continuous", "unchanged", "no cut", "single")) else 0.0,
            _clamp(1.0 - abs(row_start - c_start) / 0.5) if row_start is not None and c_start is not None else 0.0,
            _clamp(1.0 - abs(row_end - c_end) / 0.5) if row_end is not None and c_end is not None else 0.0,
        ])
        continuity_scores.append(row_score)
        csv_times = _parse_evidence_values(row.get("evidence_timestamps"))
        json_times = _parse_evidence_values(candidate.get("evidence_timestamps"))
        if csv_times and json_times:
            cross_scores.append(1.0 if all(any(abs(a - b) <= 0.06 for b in json_times) for a in csv_times) else 0.5)
        else:
            cross_scores.append(0.0)
    continuity_score = _mean(continuity_scores)
    cross_score = _mean(cross_scores)
    if isinstance(report.get("single_shot"), bool):
        continuity_score = 0.85 * continuity_score + 0.15 * (1.0 if report["single_shot"] else 0.0)
    details.update({"continuity": continuity_score, "cross_file": cross_score, "csv_row_count": len(rows), "shot_ids": sorted(seen_shots)})

    # Weighted score emphasizes temporal reconstruction and evidence grounding.
    score = (
        0.10 * manifest_score
        + 0.22 * boundary_score
        + 0.10 * kind_score
        + 0.12 * direction_score
        + 0.13 * level_score
        + 0.12 * timestamp_score
        + 0.08 * metric_score
        + 0.08 * continuity_score
        + 0.05 * cross_score
    )
    details["score"] = round(float(_clamp(score)), 6)
    details["matched_segments"] = segment_rows
    return details


def score_frame_trace(trace_bytes: bytes, probe: dict[str, Any]) -> dict[str, Any]:
    """Compare submitted rows against this evaluation's decoded source and output."""
    try:
        rows = _load_csv(trace_bytes)
    except Exception as exc:
        return {"score": 0.0, "error": f"invalid frame trace: {exc}"}
    actual = probe.get("frames", []) if probe.get("ok") else []
    if not actual:
        return {"score": 0.0, "error": "no decoded source/output measurements"}
    by_frame = {}
    duplicates = 0
    for row in rows:
        value = _num(row.get("frame"))
        if value is not None and math.isfinite(value) and value.is_integer() and 0 <= value < len(actual):
            if int(value) in by_frame:
                duplicates += 1
            by_frame[int(value)] = row
    scores = []
    for measured in actual:
        row = by_frame.get(measured["frame"])
        if row is None:
            scores.append(0.0)
            continue
        field_scores = []
        for key, allowance, tolerance in (("time_s", 0.0, 0.023), ("dark_chip_mean", 0.05, 3.0),
                               ("source_patch_mean", 0.05, 3.0), ("relative_ev", 0.01, 0.15),
                               ("corrected_patch_mean", 3.0, 5.0), ("corrected_residual_ev", 0.19, 0.18),
                               ("source_frame_luma", 0.1, 4.0), ("corrected_frame_luma", 3.2, 5.0)):
            got = _num(row.get(key))
            field_scores.append(_clamp(1.0 - max(0.0, abs(got - measured["source_patch_mean" if key == "dark_chip_mean" else key]) - allowance) / tolerance)
                                if got is not None and math.isfinite(got) else 0.0)
        scores.append(_mean(field_scores))
    coverage = len(by_frame) / len(actual)
    row_count_factor = min(len(rows) / len(actual), len(actual) / len(rows)) if rows else 0.0
    score = _mean(scores) * max(0.0, 1.0 - duplicates / len(actual)) * row_count_factor
    return {"score": round(score, 6), "coverage": round(coverage, 6), "rows": len(rows), "duplicates": duplicates}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def score_repair_recipe(recipe_bytes: bytes, probe: dict[str, Any], expected_media: dict[str, Any], reference_inventory_bytes: bytes) -> dict[str, Any]:
    try:
        recipe = json.loads(recipe_bytes.decode("utf-8"))
    except Exception as exc:
        return {"score": 0.0, "error": f"invalid repair_recipe.json: {exc}"}
    if not isinstance(recipe, dict):
        return {"score": 0.0, "error": "recipe must be an object"}
    source = recipe.get("source_stream", {})
    measurement = recipe.get("measurement", {})
    repair = recipe.get("repair", {})
    output = recipe.get("output", {})
    for obj_name, obj in (("source_stream", source), ("measurement", measurement), ("repair", repair), ("output", output)):
        if not isinstance(obj, dict):
            return {"score": 0.0, "error": f"{obj_name} must be an object"}
    expected_inventory = json.loads(reference_inventory_bytes.decode("utf-8"))
    inventory = recipe.get("inventory", [])
    inventory_by_path = {str(row.get("path")): row for row in inventory if isinstance(row, dict) and row.get("path")} if isinstance(inventory, list) else {}
    expected_by_path = {str(row.get("path")): row for row in expected_inventory if isinstance(row, dict) and row.get("path")}
    inventory_score = 0.0
    if set(inventory_by_path) == set(expected_by_path) and inventory_by_path:
        inventory_score = _mean([1.0 if inventory_by_path[path].get("decision") == expected_by_path[path].get("decision") else 0.0 for path in expected_by_path] +
                                [1.0 if str(inventory_by_path[path].get("reason", "")).strip() else 0.0 for path in expected_by_path])
    baseline = _num(measurement.get("baseline_patch_mean"))
    actual_baseline = _num(probe.get("baseline_patch_mean"))
    patch = measurement.get("patch_xyxy")
    scores = [
        1.0 if recipe.get("schema_version") == 1 else 0.0,
        1.0 if isinstance(recipe.get("source_path"), str) and recipe.get("source_path") and not recipe["source_path"].startswith("/") else 0.0,
        1.0 if recipe.get("source_sha256") == expected_media.get("source_sha256") == probe.get("source_sha256") else 0.0,
        1.0 if source.get("width") == probe.get("width") and source.get("height") == probe.get("height") and source.get("frame_count") == probe.get("decoded_frames") else 0.0,
        1.0 if _num(source.get("fps")) is not None and _num(probe.get("fps")) is not None and abs(float(source["fps"]) - float(probe["fps"])) <= 0.15 else 0.0,
        1.0 if patch == probe.get("patch_xyxy") else 0.0,
        _clamp(1.0 - abs(baseline - actual_baseline) / 5.0) if baseline is not None and actual_baseline is not None else 0.0,
        1.0 if "log2" in str(measurement.get("relative_value", "")).lower() else 0.0,
        1.0 if "gain" in str(repair.get("method", "")).lower() or "normal" in str(repair.get("method", "")).lower() else 0.0,
        1.0 if repair.get("preserve_timecode") is True and repair.get("preserve_geometry") is True else 0.0,
        1.0 if output.get("sha256") and output.get("sha256") == probe.get("output_sha256") else 0.0,
        inventory_score,
    ]
    return {"score": round(_mean(scores), 6), "checks": len(scores)}


def score_stabilized_video(probe: dict[str, Any], expected_media: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(probe, dict) or not probe.get("ok"):
        return {"score": 0.0, "error": probe.get("error", "missing or unreadable video") if isinstance(probe, dict) else "missing video probe"}
    stream = probe.get("stream", {})
    format_score = _mean([
        1.0 if probe.get("width") == expected_media.get("width") and probe.get("height") == expected_media.get("height") else 0.0,
        1.0 if _num(probe.get("fps")) is not None and abs(float(probe["fps"]) - float(expected_media["fps"])) <= 0.15 else 0.0,
        1.0 if probe.get("decoded_frames") == 288 else 0.0,
        1.0 if stream.get("codec_name") == "h264" else 0.0,
        1.0 if stream.get("video_streams") == 1 and stream.get("audio_streams") == 0 else 0.0,
    ])
    frames = probe.get("frames", [])
    if not frames:
        return {"score": round(0.15 * format_score, 6), "format": format_score, "scene": 0.0, "stability": 0.0}
    scene = _mean([_mean([_clamp(1.0 - max(0.0, f["scene_mae"] - 4.0) / 22.0),
                          _clamp((f["scene_tile_fraction"] - 0.55) / 0.40),
                          _clamp(1.0 - max(0.0, f["strip_mae"] - 4.0) / 16.0)]) for f in frames])
    patch = [f["corrected_patch_mean"] for f in frames]
    baseline = probe.get("baseline_patch_mean", 0.0)
    mean = _mean(patch)
    std = (_mean([(v - mean) ** 2 for v in patch])) ** 0.5
    stability = _mean([_clamp(1.0 - max(0.0, abs(mean - baseline) - 3.0) / max(5.0, baseline * .25)),
                       _clamp(1.0 - std / max(3.0, baseline * .18))])
    return {"score": round(0.15 * format_score + 0.65 * scene + 0.20 * stability, 6),
            "format": format_score, "scene": round(scene, 6), "stability": round(stability, 6), "checked_frames": len(frames)}


def score_package(
    report_bytes: bytes,
    continuity_bytes: bytes,
    trace_bytes: bytes,
    recipe_bytes: bytes,
    reference_event_bytes: bytes,
    reference_metrics_bytes: bytes,
    expected_media_bytes: bytes,
    reference_inventory_bytes: bytes,
    video_probe: dict[str, Any],
) -> dict[str, Any]:
    expected_media = json.loads(expected_media_bytes.decode("utf-8"))
    report = score_submission(report_bytes, continuity_bytes, reference_event_bytes, reference_metrics_bytes, expected_media_bytes, video_probe)
    trace = score_frame_trace(trace_bytes, video_probe)
    video = score_stabilized_video(video_probe, expected_media)
    recipe = score_repair_recipe(recipe_bytes, video_probe, expected_media, reference_inventory_bytes)
    score = 0.34 * float(report.get("score", 0.0)) + 0.22 * float(trace["score"]) + 0.30 * float(video["score"]) + 0.14 * float(recipe["score"])
    if video_probe.get("ok") and video_probe.get("source_sha256") != expected_media.get("source_sha256"):
        score = min(score, 0.40)
    if video_probe.get("ok") and video_probe.get("stream", {}).get("codec_name") != "h264":
        score = min(score, 0.75)
    if video.get("scene", 0.0) < 0.55 and video_probe.get("ok"):
        score = min(score, 0.65)
    if trace.get("score", 0.0) < 0.80 and video_probe.get("ok"):
        score = min(score, 0.85)
    return {"score": round(score, 6), "report": report, "frame_trace": trace, "stabilized_video": video, "repair_recipe": recipe}
