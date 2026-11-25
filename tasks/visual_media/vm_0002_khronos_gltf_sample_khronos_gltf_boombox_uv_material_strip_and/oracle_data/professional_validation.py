from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("boombox_scorer", root / "scripts" / "score_boombox.py")
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

reference = root / "reference" / "boombox_intact.glb"
gold = root / "oracle_data" / "reference_output" / "boombox_repaired.glb"
starter = root / "oracle_data" / "starter_output" / "boombox_repaired.glb"
scores = {"reference_output": mod.score(gold, reference), "starter_output": mod.score(starter, reference)}
print(json.dumps(scores, indent=2, sort_keys=True))
if scores["reference_output"].get("score", 0.0) < 0.99:
    raise SystemExit("reference output did not score as a complete repair")
if scores["starter_output"].get("score", 1.0) > 0.80:
    raise SystemExit("starter output is insufficiently damaged")
