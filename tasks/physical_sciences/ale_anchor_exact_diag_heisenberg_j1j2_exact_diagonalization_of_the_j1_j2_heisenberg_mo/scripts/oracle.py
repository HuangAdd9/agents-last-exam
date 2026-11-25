from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from model import solve_case,write_outputs

def main():
 p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output-dir',required=True); a=p.parse_args()
 c=json.loads(Path(a.config).read_text()); r=solve_case(float(c['J2']),int(c['nx']),int(c['ny']),int(c['n_lanczos']),int(c['n_omega'])); write_outputs(r,Path(a.output_dir))
if __name__=='__main__': main()
