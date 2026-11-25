"""Focused local checks for physics invariants and hidden-case isolation."""
from __future__ import annotations
import json, shutil, tempfile, subprocess, sys, hashlib
from pathlib import Path
import importlib.util

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("v",ROOT/"scripts/verify_outputs.py"); v=importlib.util.module_from_spec(spec); spec.loader.exec_module(v)

def run_solver(src, case, out):
    out.mkdir(parents=True, exist_ok=True); shutil.copy(src, out/"tomography_solver.py")
    subprocess.run([sys.executable,str(out/"tomography_solver.py"),"--input",str(case),"--output",str(out)],check=True)

def main():
    # Loader smoke test catches f-string/import failures that py_compile
    # cannot detect when the task description is evaluated.
    mspec=importlib.util.spec_from_file_location("jimmu_main",ROOT/"main.py"); mm=importlib.util.module_from_spec(mspec); sys.modules["jimmu_main"]=mm; mspec.loader.exec_module(mm)
    assert "s_{n-1}" in mm._description()
    vis_case=json.loads((ROOT/"input/tomography_case.json").read_text())
    hid_case=json.loads((ROOT/"reference/hidden_case.json").read_text())
    assert (vis_case["grid"]["nx"],vis_case["grid"]["nz"]) != (hid_case["grid"]["nx"],hid_case["grid"]["nz"])
    manifest=json.loads((ROOT/"GENERATION_MANIFEST.json").read_text())
    staged=ROOT/"input/pangaea_941101_source.tsv"
    assert staged.is_file() and "sha256:"+hashlib.sha256(staged.read_bytes()).hexdigest() == manifest["paper_to_artifact_resolution"]["staged_source_sha256"]
    diag=(vis_case["grid"]["cell_width_km"]**2+vis_case["grid"]["cell_height_km"]**2)**0.5
    assert all(max(r["path_lengths_km"]) <= diag + 2e-6 for r in vis_case["rays"])
    malformed=json.loads(json.dumps(vis_case)); malformed["rays"][0]["path_lengths_km"][0]=diag*2
    # Keep the malformed fixture outside the package and remove it after
    # validation; _case must reject impossible cell segments.
    bad_case=Path(tempfile.mkstemp(suffix=".json")[1]); bad_case.write_text(json.dumps(malformed))
    try:
        try: v._case(bad_case); raise AssertionError("invalid geometry accepted")
        except ValueError: pass
    finally: bad_case.unlink(missing_ok=True)
    # Preserve total length and all per-cell bounds while swapping two crossed
    # cells: exact intersection recomputation must still reject the ray.
    permuted=json.loads(json.dumps(vis_case)); ray=permuted["rays"][0]
    nz=[i for i,x in enumerate(ray["path_lengths_km"]) if x > 1e-8]
    assert len(nz) >= 2
    nz0,nz1=next((a,b) for a in nz for b in nz if a < b and
                 abs(ray["path_lengths_km"][a]-ray["path_lengths_km"][b]) > 1e-5)
    ray["path_lengths_km"][nz0],ray["path_lengths_km"][nz1]=ray["path_lengths_km"][nz1],ray["path_lengths_km"][nz0]
    swap_case=Path(tempfile.mkstemp(suffix=".json")[1]); swap_case.write_text(json.dumps(permuted))
    try:
        try: v._case(swap_case); raise AssertionError("permuted intersections accepted")
        except ValueError: pass
    finally: swap_case.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as td:
        t=Path(td); perfect=t/"perfect"; run_solver(ROOT/"scripts/perfect_solver.py",ROOT/"input/tomography_case.json",perfect)
        vis=v.score_dir(perfect,ROOT/"input/tomography_case.json",ROOT/"reference/visible_reference_model.json"); assert vis["score"]>.19, vis
        hidden=t/"perfect_hidden"; run_solver(ROOT/"scripts/perfect_solver.py",ROOT/"reference/hidden_case.json",hidden)
        assert v.score_hidden(hidden,ROOT/"reference/hidden_reference/model.json",ROOT/"reference/hidden_case.json")["score"]>.79
        empty=t/"empty"; empty.mkdir(); assert v.score_dir(empty,ROOT/"input/tomography_case.json",ROOT/"reference/visible_reference_model.json")["score"]==0.0
        wrong=t/"wrong"; run_solver(ROOT/"scripts/perfect_solver.py",ROOT/"reference/hidden_case.json",wrong)
        d=json.loads((wrong/"model.json").read_text()); d["velocity_km_s"]=[[x*100 for x in r] for r in d["velocity_km_s"]]; d["ray_predictions_s"]=[x*100 for x in d["ray_predictions_s"]]; d["rms_s"]*=100; (wrong/"model.json").write_text(json.dumps(d)); assert v.score_hidden(wrong,ROOT/"reference/hidden_reference/model.json",ROOT/"reference/hidden_case.json")["score"]==0.0
        # A direct normal-equation model with a fabricated scalar-only/decreasing
        # trace is rejected: intermediate slowness states are mandatory and all
        # row metrics/steps are recomputed from them.
        fabricated=t/"fabricated"; run_solver(ROOT/"scripts/perfect_solver.py",ROOT/"input/tomography_case.json",fabricated)
        shutil.copy(ROOT/"reference/visible_reference_model.json",fabricated/"model.json")
        lines=(fabricated/"trace.csv").read_text().splitlines(); hdr=lines[0].split(",")
        scalar_hdr=",".join(hdr[:4]); scalar_rows=[",".join(row.split(",")[:4]) for row in lines[1:5]]
        (fabricated/"trace.csv").write_text(scalar_hdr+"\n"+"\n".join(scalar_rows)+"\n")
        assert v.score_dir(fabricated,ROOT/"input/tomography_case.json",ROOT/"reference/visible_reference_model.json")["score"]==0.0
        # A solver that searches the authoring/evaluator filesystem must not see
        # the private oracle when rerun in the evaluator's true sandbox.
        mm = sys.modules["jimmu_main"]
        attack=t/"attack"; attack.mkdir();
        (attack/"tomography_solver.py").write_text('''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument("--input"); p.add_argument("--output"); a=p.parse_args()
hits=list(Path("/").rglob("hidden_reference/model.json"))
Path(a.output,"model.json").write_text(json.dumps({"oracle_hits": [str(x) for x in hits]}))
''')
        sandbox_out=t/"sandbox_out"; p=mm._run_isolated(attack/"tomography_solver.py",ROOT/"reference/hidden_case.json",sandbox_out)
        assert p.returncode == 0, p.stderr
        got=json.loads((sandbox_out/"model.json").read_text()); assert got["oracle_hits"] == []
        assert not v.validate_run(sandbox_out,ROOT/"reference/hidden_case.json")
    print("adversarial checks passed")

if __name__=="__main__": main()
