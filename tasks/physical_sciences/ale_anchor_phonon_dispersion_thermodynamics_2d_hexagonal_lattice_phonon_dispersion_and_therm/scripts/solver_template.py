from __future__ import annotations
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import solve_case,write_outputs
p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output-dir',required=True); a=p.parse_args()
write_outputs(solve_case(json.loads(Path(a.config).read_text())),Path(a.output_dir))
