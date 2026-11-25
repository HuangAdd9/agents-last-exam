from __future__ import annotations
import math
from pathlib import Path

RADII = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57, "S": 1.05, "Cl": 1.02}
MAX_DEG = {"H": 1, "C": 4, "N": 3, "O": 2, "F": 1, "S": 6, "Cl": 1}

def parse_xyz(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2: raise ValueError("too_few_lines")
    n = int(lines[0].strip())
    if n <= 0 or len(lines) != n + 2: raise ValueError("atom_count_mismatch")
    atoms = []
    for row in lines[2:]:
        p = row.split()
        if len(p) != 4: raise ValueError("bad_atom_row")
        el = p[0]; xyz = tuple(float(x) for x in p[1:])
        if not all(math.isfinite(x) for x in xyz): raise ValueError("nonfinite_coordinate")
        atoms.append((el, xyz))
    return atoms

def analyze(path: Path):
    try: atoms = parse_xyz(path)
    except Exception as exc:
        return {"problematic": True, "reasons": ["parse_error:" + str(exc)], "atom_count": 0, "bond_count": 0, "min_distance_A": 0.0}
    reasons, edges, mind = set(), [], math.inf
    for el, _ in atoms:
        if el not in RADII: reasons.add("unknown_element")
    for i in range(len(atoms)):
        for j in range(i + 1, len(atoms)):
            d = math.sqrt(sum((atoms[i][1][k]-atoms[j][1][k])**2 for k in range(3)))
            mind = min(mind, d)
            if d < 0.45: reasons.add("overlap")
            if atoms[i][0] in RADII and atoms[j][0] in RADII and d >= 0.45 and d <= 1.25*(RADII[atoms[i][0]]+RADII[atoms[j][0]]): edges.append((i,j))
    deg = [0]*len(atoms)
    for i,j in edges: deg[i]+=1; deg[j]+=1
    if any(d == 0 for d in deg) or not edges: reasons.add("disconnected")
    if edges:
        seen={0}; changed=True
        while changed:
            changed=False
            for i,j in edges:
                if i in seen or j in seen:
                    old=len(seen); seen.update((i,j)); changed |= len(seen) != old
        if len(seen) != len(atoms): reasons.add("disconnected")
    for i,(el,_) in enumerate(atoms):
        if el in MAX_DEG and deg[i] > MAX_DEG[el]: reasons.add("valence_exceeded")
    return {"problematic": bool(reasons), "reasons": sorted(reasons), "atom_count": len(atoms), "bond_count": len(edges), "min_distance_A": float(mind if math.isfinite(mind) else 0.0)}

def analyze_dir(directory: Path):
    files = {p.name: analyze(p) for p in sorted(directory.glob("*.xyz"))}
    return {"files": files, "summary": {"problematic_count": sum(1 for r in files.values() if r["problematic"]), "file_count": len(files), "all_finite": all(math.isfinite(r["min_distance_A"]) for r in files.values())}}

if __name__ == "__main__":
    import argparse, json
    ap=argparse.ArgumentParser(); ap.add_argument("directory", type=Path); ap.add_argument("output", type=Path); a=ap.parse_args()
    a.output.write_text(json.dumps(analyze_dir(a.directory), sort_keys=True, indent=2)+"\n", encoding="utf-8")
