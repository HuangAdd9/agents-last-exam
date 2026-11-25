from __future__ import annotations
import json
from pathlib import Path
try:
 from .oracle import metrics
except ImportError:
 from oracle import metrics

def score(output_dir, manifest=None): return metrics(output_dir,manifest)
def score_remote(output_dir, manifest): return metrics(output_dir,manifest)['score']

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser(); ap.add_argument('--output-dir',required=True); ap.add_argument('--manifest'); a=ap.parse_args(); print(json.dumps(score(a.output_dir,a.manifest),sort_keys=True))
