import json
from pathlib import Path
case={"model_name":"benchmark-authored hidden silicon valley mutation","units":{"energy":"eV","k_coordinates":"fractional primitive reciprocal coordinates","lattice":"angstrom"},"grid":{"n":13,"includes_endpoint":False},"valence":{"e0":-0.10,"curvature":1.28,"warping":0.05},"conduction":{"e0":1.16,"curvature":[0.60,0.74,0.92],"valley":[0.6923076923,0.0769230769,0.0],"warping":0.04},"gw_self_energy":{"valence_shift":-0.045,"conduction_shift":0.43,"nonlocal_strength":0.13,"screening_length":0.21},"path":{"labels":["Gamma","X","L","Gamma"],"points":[[0,0,0],[0.5,0,0],[0.5,0.5,0.5],[0,0,0]],"segments":19}}
Path(__file__).resolve().parents[1].joinpath('reference/hidden_case.json').write_text(json.dumps(case,sort_keys=True,indent=2)+'\n')
