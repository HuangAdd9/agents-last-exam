"""Deterministically generate physically valid straight-ray benchmark fixtures."""
from __future__ import annotations
import json, math, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def boundary_point(rng, width, height):
    side = rng.randrange(4)
    u = rng.uniform(0.03, 0.97)
    if side == 0: return (0.0, u * height)
    if side == 1: return (width, u * height)
    if side == 2: return (u * width, 0.0)
    return (u * width, height)

def ray_segments(p0, p1, nx, nz, cw, ch):
    x0, z0 = p0; x1, z1 = p1
    dx, dz = x1 - x0, z1 - z0
    total = math.hypot(dx, dz)
    if total < 1e-9: return [0.0] * (nx * nz)
    cuts = [0.0, 1.0]
    for k in range(1, nx):
        if abs(dx) > 1e-12:
            t = (k * cw - x0) / dx
            if 1e-10 < t < 1 - 1e-10: cuts.append(t)
    for k in range(1, nz):
        if abs(dz) > 1e-12:
            t = (k * ch - z0) / dz
            if 1e-10 < t < 1 - 1e-10: cuts.append(t)
    cuts = sorted(set(round(t, 14) for t in cuts))
    out = [0.0] * (nx * nz)
    for a, b in zip(cuts, cuts[1:]):
        mid = (a + b) * 0.5
        x, z = x0 + mid * dx, z0 + mid * dz
        ix = min(nx - 1, max(0, int(math.floor(x / cw))))
        iz = min(nz - 1, max(0, int(math.floor(z / ch))))
        out[iz * nx + ix] += total * (b - a)
    return out

def make_case(case_id, nx, nz, nrays, seed, path):
    rng = random.Random(seed)
    cw, ch = 15.0, 4.0
    width, height = nx * cw, nz * ch
    # A heterogeneous but positive slowness field (s/km), row-major in z then x.
    truth = [0.182 + 0.006 * z + 0.004 * x + 0.003 * math.sin(0.8 * x + 0.5 * z)
             for z in range(nz) for x in range(nx)]
    rays = []
    attempts = 0
    while len(rays) < nrays and attempts < nrays * 100:
        attempts += 1
        p0, p1 = boundary_point(rng, width, height), boundary_point(rng, width, height)
        lengths = ray_segments(p0, p1, nx, nz, cw, ch)
        if sum(lengths) < min(cw, ch) * 0.75 or sum(v > 1e-9 for v in lengths) < 2:
            continue
        # Tiny deterministic pick noise keeps the inversion non-trivial while
        # retaining a physically recomputable straight-ray forward model.
        sigma = rng.choice([0.0035, 0.0045, 0.0055, 0.0065, 0.0075])
        clean = sum(a * s for a, s in zip(lengths, truth))
        observed = clean + rng.uniform(-0.0012, 0.0012)
        rays.append({"id": f"ray_{len(rays):03d}",
                     "source_xy_km": [round(p0[0], 8), round(p0[1], 8)],
                     "receiver_xy_km": [round(p1[0], 8), round(p1[1], 8)],
                     "path_lengths_km": [round(v, 10) for v in lengths],
                     "observed_time_s": round(observed, 9),
                     "uncertainty_s": sigma})
    if len(rays) != nrays:
        raise RuntimeError("could not generate enough rays")
    max_seg = math.hypot(cw, ch) + 1e-9
    if any(v > max_seg for r in rays for v in r["path_lengths_km"]):
        raise AssertionError("segment exceeds cell diagonal")
    case = {"schema_version": "tomography-case-1", "case_id": case_id,
            "grid": {"nx": nx, "nz": nz, "cell_width_km": cw, "cell_height_km": ch},
            "rays": rays}
    path.write_text(json.dumps(case, indent=2) + "\n")

if __name__ == "__main__":
    make_case("visible_jimmu_multicell", 6, 4, 64, 821121, ROOT / "input/tomography_case.json")
    make_case("hidden_jimmu_changed_grid", 5, 3, 96, 991217, ROOT / "reference/hidden_case.json")
