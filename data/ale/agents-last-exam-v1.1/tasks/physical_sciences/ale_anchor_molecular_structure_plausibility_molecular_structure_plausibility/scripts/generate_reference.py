from pathlib import Path
import json
from oracle import analyze_dir

root = Path(__file__).resolve().parents[1]
payload = {"visible": analyze_dir(root / "input" / "xyz_files"), "hidden": analyze_dir(root / "reference" / "hidden_xyz")}
(root / "reference" / "expected.json").write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
