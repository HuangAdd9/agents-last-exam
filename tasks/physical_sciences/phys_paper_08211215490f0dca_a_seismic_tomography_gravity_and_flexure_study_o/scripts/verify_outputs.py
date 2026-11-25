from __future__ import annotations
import csv, json, math
from pathlib import Path

LAMBDA = 1.0e-4
PRIOR = 0.2
MAX_ARTIFACT_BYTES = 2_000_000
TOL = 2.0e-6
GEOM_TOL = 3.0e-7

def _finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))

def _regular(path: Path, limit: int = MAX_ARTIFACT_BYTES):
    st = path.lstat()
    if not path.is_file() or path.is_symlink() or st.st_size > limit:
        raise ValueError("artifact must be a bounded regular file")

def _read_json(path):
    _regular(Path(path)); return json.loads(Path(path).read_text())

def _segment_lengths(p0, p1, nx, nz, cw, ch):
    """Deterministically recompute row-major rectangular-cell intersections."""
    x0, z0 = map(float, p0); x1, z1 = map(float, p1)
    dx, dz = x1 - x0, z1 - z0
    total = math.hypot(dx, dz)
    out = [0.0] * (nx * nz)
    if total < 1e-12:
        return out
    cuts = [0.0, 1.0]
    for k in range(1, nx):
        if abs(dx) > 1e-14:
            t = (k * cw - x0) / dx
            if 1e-10 < t < 1.0 - 1e-10: cuts.append(t)
    for k in range(1, nz):
        if abs(dz) > 1e-14:
            t = (k * ch - z0) / dz
            if 1e-10 < t < 1.0 - 1e-10: cuts.append(t)
    cuts = sorted(set(round(t, 14) for t in cuts))
    for a, b in zip(cuts, cuts[1:]):
        if b - a <= 1e-14: continue
        mid = (a + b) * 0.5
        x, z = x0 + mid * dx, z0 + mid * dz
        ix = min(nx - 1, max(0, int(math.floor(x / cw))))
        iz = min(nz - 1, max(0, int(math.floor(z / ch))))
        out[iz * nx + ix] += total * (b - a)
    return out

def _case(path):
    c = _read_json(path)
    if c.get("schema_version") != "tomography-case-1": raise ValueError("case schema")
    g = c.get("grid", {}); nx, nz = int(g.get("nx", 0)), int(g.get("nz", 0))
    if nx < 1 or nz < 1 or set(g) != {"nx", "nz", "cell_width_km", "cell_height_km"}: raise ValueError("case grid")
    if not _finite(g["cell_width_km"]) or not _finite(g["cell_height_km"]): raise ValueError("case grid values")
    rays = c.get("rays"); n = nx * nz
    if not isinstance(rays, list) or len(rays) < n: raise ValueError("case rays")
    diag = math.hypot(float(g["cell_width_km"]), float(g["cell_height_km"]))
    width, height = nx * float(g["cell_width_km"]), nz * float(g["cell_height_km"])
    for r in rays:
        if not isinstance(r, dict): raise ValueError("ray")
        supplied_paths = r.get("path_lengths_km")
        if supplied_paths is not None and len(supplied_paths) != n: raise ValueError("ray length")
        if supplied_paths is not None and any(not _finite(x) or float(x) < 0 for x in supplied_paths): raise ValueError("ray path")
        if not _finite(r.get("observed_time_s")) or not _finite(r.get("uncertainty_s", .006)) or float(r.get("uncertainty_s", .006)) <= 0: raise ValueError("ray datum")
        # Fixtures are physical rectangular-grid straight rays, not arbitrary
        # sensitivity rows.  Endpoints and segment lengths must agree and no
        # cell segment may exceed that cell's diagonal.
        if set(r) < {"source_xy_km", "receiver_xy_km"}: raise ValueError("ray geometry missing")
        p0, p1 = r["source_xy_km"], r["receiver_xy_km"]
        if (not isinstance(p0, list) or not isinstance(p1, list) or len(p0) != 2 or len(p1) != 2 or
                any(not _finite(v) for v in p0 + p1) or
                any(float(v) < -1e-8 for v in p0 + p1) or
                float(p0[0]) > width + 1e-8 or float(p1[0]) > width + 1e-8 or
                float(p0[1]) > height + 1e-8 or float(p1[1]) > height + 1e-8): raise ValueError("ray endpoints")
        boundary = lambda p: (abs(float(p[0])) <= GEOM_TOL or abs(float(p[0])-width) <= GEOM_TOL or
                              abs(float(p[1])) <= GEOM_TOL or abs(float(p[1])-height) <= GEOM_TOL)
        if not boundary(p0) or not boundary(p1): raise ValueError("ray endpoints must be on boundary")
        expected = _segment_lengths(p0, p1, nx, nz, float(g["cell_width_km"]), float(g["cell_height_km"]))
        supplied = expected if supplied_paths is None else [float(v) for v in supplied_paths]
        endpoint_length = math.hypot(float(p1[0])-float(p0[0]), float(p1[1])-float(p0[1]))
        if abs(sum(supplied) - endpoint_length) > 2e-5: raise ValueError("ray geometry mismatch")
        if any(abs(a-b) > max(2e-6, GEOM_TOL*max(1.0, abs(b))) for a,b in zip(supplied, expected)):
            raise ValueError("ray cell intersections mismatch")
        if any(float(v) > diag + 2e-6 for v in supplied): raise ValueError("ray segment exceeds cell diagonal")
    return c

def _model(path, case=None):
    d = path if isinstance(path, dict) else _read_json(path); req = {"grid", "velocity_km_s", "slowness_s_per_km", "rms_s", "ray_predictions_s"}
    if set(d) != req: raise ValueError("model keys")
    g = d["grid"]
    if set(g) != {"nx", "nz", "cell_width_km", "cell_height_km"}: raise ValueError("grid")
    nx, nz = int(g["nx"]), int(g["nz"])
    if nx < 1 or nz < 1 or any(not _finite(g[k]) or float(g[k]) <= 0 for k in ("cell_width_km", "cell_height_km")): raise ValueError("grid values")
    if case is not None and g != case["grid"]: raise ValueError("grid does not match case")
    for k in ("velocity_km_s", "slowness_s_per_km"):
        a = d[k]
        if not isinstance(a, list) or len(a) != nz or any(not isinstance(r, list) or len(r) != nx for r in a): raise ValueError("array shape")
        if any(not _finite(x) or float(x) <= 0 for r in a for x in r): raise ValueError("array values")
    if case is not None and len(d["ray_predictions_s"]) != len(case["rays"]): raise ValueError("prediction length")
    if not isinstance(d["ray_predictions_s"], list) or any(not _finite(x) for x in d["ray_predictions_s"]) or not _finite(d["rms_s"]): raise ValueError("finite")
    flat_s = _flat(d["slowness_s_per_km"]); flat_v = _flat(d["velocity_km_s"])
    if any(abs(float(v) * float(s) - 1.0) > 3e-6 * max(1.0, abs(float(v) * float(s))) for v, s in zip(flat_v, flat_s)): raise ValueError("velocity/slowness reciprocity")
    if case is not None:
        pred = _predictions(case, flat_s)
        if _rmse(pred, d["ray_predictions_s"]) > TOL: raise ValueError("ray predictions inconsistent")
        rms = _rmse(pred, [r["observed_time_s"] for r in case["rays"]])
        if abs(rms - float(d["rms_s"])) > TOL: raise ValueError("rms inconsistent")
    return d

def _trace(path, case=None, model=None):
    _regular(Path(path))
    with open(path, newline="") as f: rows = list(csv.DictReader(f))
    n = int(case["grid"]["nx"])*int(case["grid"]["nz"]) if case is not None else None
    required = {"iteration", "objective", "rms_s", "step"}
    state_cols = {f"s_{i}" for i in range(n)} if n is not None else set()
    if len(rows) < 4 or not rows or set(rows[0]) != required | state_cols: raise ValueError("trace schema")
    vals=[]; states=[]
    for r in rows:
        if any(not _finite(float(r[k])) for k in required | state_cols): raise ValueError("trace finite")
        s = [float(r[f"s_{i}"]) for i in range(n)]
        if any(x <= 0 for x in s): raise ValueError("trace positivity")
        vals.append((int(float(r["iteration"])), float(r["objective"]), float(r["rms_s"]), float(r["step"])))
        states.append(s)
    if [x[0] for x in vals] != list(range(len(vals))) or any(vals[i+1][1] > vals[i][1] + 1e-8 for i in range(len(vals)-1)): raise ValueError("trace monotonicity")
    # Every row is an auditable state, rather than an unbound scalar progress claim.
    if sum(v[3] > 1e-10 for v in vals[1:]) < 1 or len({tuple(round(x, 12) for x in s) for s in states}) < 4: raise ValueError("trace is not iterative")
    if case is not None and model is not None:
        observed = [r["observed_time_s"] for r in case["rays"]]
        for i, (it, obj, rms, step) in enumerate(vals):
            pred = _predictions(case, states[i])
            calc_obj = _objective(case, states[i])
            calc_rms = _rmse(pred, observed)
            if abs(obj-calc_obj) > max(2e-8, 2e-6*abs(calc_obj)) or abs(rms-calc_rms) > TOL: raise ValueError("trace state metrics inconsistent")
            if i == 0 and abs(step) > 2e-8: raise ValueError("initial trace step")
            if i:
                calc_step = math.sqrt(sum((a-b)**2 for a,b in zip(states[i], states[i-1])))
                if abs(step-calc_step) > max(2e-8, 2e-6*calc_step): raise ValueError("trace step inconsistent")
                g=_gradient(case, states[i-1]); diag=[LAMBDA+sum((float(_ray_paths(case,r)[k])/float(r.get("uncertainty_s", .006)))**2 for r in case["rays"]) for k in range(n)]
                d=[-g[k]/diag[k] for k in range(n)]; hd=_hessian_apply(case,d)
                den=sum(a*b for a,b in zip(d,hd))
                if den <= 0: raise ValueError("trace curvature")
                alpha=-sum(a*b for a,b in zip(g,d))/den
                expected=[states[i-1][k]+alpha*d[k] for k in range(n)]
                if any(abs(a-b) > max(3e-7,3e-6*abs(b)) for a,b in zip(states[i],expected)): raise ValueError("trace update is not iterative gradient state")
        final_obj = _objective(case, _flat(model["slowness_s_per_km"]))
        if any(abs(a-b) > 2e-7 for a,b in zip(states[-1], _flat(model["slowness_s_per_km"]))): raise ValueError("trace final state")
        if abs(vals[-1][1] - final_obj) > max(2e-8, 2e-6 * abs(final_obj)): raise ValueError("trace final objective")
    return rows

def _report(path, case):
    d = _read_json(path); req={"method","units","case_id","invariant"}
    if set(d) != req or not isinstance(d["method"], str) or not ("iter" in d["method"].lower() or "gradient" in d["method"].lower() or "damped" in d["method"].lower()) or d["case_id"] != case["case_id"]: raise ValueError("report")
    if d["units"] != {"velocity":"km/s","slowness":"s/km","time":"s"}: raise ValueError("units")
    if d["invariant"] != "forward_time_sum_matches_reported_rms": raise ValueError("invariant")

def _flat(a): return [x for r in a for x in r]
def _ray_paths(case, r):
    if "path_lengths_km" in r: return [float(x) for x in r["path_lengths_km"]]
    g=case["grid"]
    return _segment_lengths(r["source_xy_km"], r["receiver_xy_km"], int(g["nx"]), int(g["nz"]), float(g["cell_width_km"]), float(g["cell_height_km"]))

def _predictions(case, slowness): return [sum(float(a) * slowness[i] for i, a in enumerate(_ray_paths(case,r))) for r in case["rays"]]
def _rmse(a,b): return math.sqrt(sum((float(x)-float(y))**2 for x,y in zip(a,b))/len(a))
def _objective(case, slowness):
    return sum(((p-float(r["observed_time_s"]))/float(r.get("uncertainty_s", .006)))**2 for p,r in zip(_predictions(case, slowness), case["rays"])) + LAMBDA*sum((x-PRIOR)**2 for x in slowness)

def _gradient(case, slowness):
    g=[LAMBDA*(x-PRIOR) for x in slowness]
    for r in case["rays"]:
        sig=float(r.get("uncertainty_s", .006)); w=1.0/(sig*sig)
        paths=_ray_paths(case,r); e=sum(float(a)*slowness[i] for i,a in enumerate(paths))-float(r["observed_time_s"])
        for i,a in enumerate(paths): g[i]+=w*float(a)*e
    return g

def _hessian_apply(case, direction):
    z=[LAMBDA*d for d in direction]
    for r in case["rays"]:
        sig=float(r.get("uncertainty_s", .006)); w=1.0/(sig*sig)
        paths=_ray_paths(case,r); ad=sum(float(a)*direction[i] for i,a in enumerate(paths))
        for i,a in enumerate(paths): z[i]+=w*float(a)*ad
    return z

def validate_run(out, case_path):
    """Validate a solver run, including every auditable intermediate state."""
    out=Path(out); case=_case(case_path)
    try:
        for name in ("model.json","trace.csv","report.json"): _regular(out/name)
        model=_model(out/"model.json",case)
        _trace(out/"trace.csv",case,model)
        _report(out/"report.json",case)
        return True
    except Exception:
        return False

def score_dir(out, case_path, reference_model=None):
    out=Path(out); case = _case(case_path)
    files=[out/"model.json",out/"trace.csv",out/"report.json",out/"tomography_solver.py"]
    try:
        for p in files: _regular(p)
        m=_model(out/"model.json", case)
        if not validate_run(out,case_path): raise ValueError("invalid run")
    except Exception: return {"score":0.0,"components":{"schema":0.0}}
    if reference_model is None: return {"score":1.0,"components":{"schema":1.0,"physics":1.0}}
    try: ref=_model(reference_model, case)
    except Exception: return {"score":0.0,"components":{"schema":1.0}}
    vm=_rmse(_flat(m["velocity_km_s"]),_flat(ref["velocity_km_s"]))
    visible=max(0.0,min(1.0,math.exp(-vm/0.20)))
    return {"score":0.20*visible,"components":{"schema":1.0,"visible_model":visible},"visible_velocity_rmse":vm,"hidden_required":True}

def score_hidden(out, hidden_ref, case_path=None):
    try:
        if case_path is None: h=_model(hidden_ref); m=_model(Path(out)/"model.json")
        else:
            case=_case(case_path); m=_model(Path(out)/"model.json",case); h=_model(hidden_ref,case)
        vel=max(0.0,min(1.0,math.exp(-_rmse(_flat(m["velocity_km_s"]),_flat(h["velocity_km_s"]))/0.12)))
        pred=max(0.0,min(1.0,math.exp(-_rmse(m["ray_predictions_s"],h["ray_predictions_s"])/0.03)))
        rms=max(0.0,min(1.0,math.exp(-abs(float(m["rms_s"])-float(h["rms_s"]))/0.01)))
        return {"score":0.50*vel+0.25*pred+0.05*rms,"components":{"hidden_velocity":vel,"hidden_predictions":pred,"hidden_rms":rms}}
    except Exception: return {"score":0.0,"components":{"hidden_velocity":0.0,"hidden_predictions":0.0,"hidden_rms":0.0}}
