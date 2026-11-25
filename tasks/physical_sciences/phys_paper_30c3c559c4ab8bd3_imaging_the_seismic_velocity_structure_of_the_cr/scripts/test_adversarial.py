import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

root = Path(__file__).parents[1]
sys.path.insert(0, str(root))
from oracle import run, solve_reference
import main

inp = json.loads((root / 'input/tomography_input.json').read_text())
hid = json.loads((root / 'reference/hidden_case.json').read_text())
dims = json.loads((root / 'reference/hidden_dimensions_case.json').read_text())
assert np.asarray(inp['ray_matrix']).ndim == 3
assert np.allclose(np.asarray(inp['ray_matrix']).sum(axis=2), 1)
assert not np.allclose(np.asarray(inp['ray_matrix'])[0], np.asarray(inp['ray_matrix'])[1])
assert np.ptp(np.asarray(hid['model_variance_s2_per_km2'])) > 0
assert np.ptp(np.asarray(hid['data_variance_s2_per_km2'])) > 0
assert np.any(np.max(np.asarray(hid['ray_matrix']), axis=2) < 0.5)

with tempfile.TemporaryDirectory() as td:
    p = Path(td)
    run(str(root / 'input/tomography_input.json'), str(p))
    hp = p / 'hidden'
    run(str(root / 'reference/hidden_case.json'), str(hp))
    np.savez(p / 'hidden_result.npz', **solve_reference(hid))
    (p / 'hidden_diagnostics.json').write_bytes((hp / 'diagnostics.json').read_bytes())
    ref = solve_reference(inp)
    href = solve_reference(hid)
    good = main.score_dir(p, ref, href, np.asarray(inp['observed_slowness_s_per_km']),
                          np.asarray(hid['observed_slowness_s_per_km']))
    assert good['score'] > 0.99, good
    (p / 'result.npz').write_bytes(b'bad')
    assert main.score_dir(p, ref, None, np.asarray(inp['observed_slowness_s_per_km']))['score'] == 0.0
    run(str(root / 'input/tomography_input.json'), str(p))
    z = np.load(p / 'result.npz')
    bad = {k: z[k] for k in z.files}
    bad['velocity_km_s'] *= 1000
    np.savez(p / 'result.npz', **bad)
    assert main.score_dir(p, ref, None, np.asarray(inp['observed_slowness_s_per_km']))['components']['visible'] < 1
    # A candidate cannot earn the diagnostics credit by changing predictions
    # and reporting the matching self-computed residual: the expected maximum
    # is derived from the private oracle result.
    run(str(root / 'input/tomography_input.json'), str(p))
    z = np.load(p / 'result.npz')
    self_consistent = {k: z[k] for k in z.files}
    self_consistent['predicted_slowness_s_per_km'] += 0.01
    np.savez(p / 'result.npz', **self_consistent)
    fake_max = float(np.max(np.abs(self_consistent['predicted_slowness_s_per_km'] - np.asarray(inp['observed_slowness_s_per_km']))))
    (p / 'diagnostics.json').write_text(json.dumps({'max_abs_residual_s_per_km': fake_max, 'units': main.UNITS}))
    diag_score = main.score_dir(p, ref, None, np.asarray(inp['observed_slowness_s_per_km']))
    assert diag_score['components']['diagnostics'] == 0.0, diag_score
    original = main._solve
    main._solve = lambda _: (_ for _ in ()).throw(RuntimeError('mutated'))
    assert np.allclose(solve_reference(inp)['velocity_km_s'], ref['velocity_km_s'])
    main._solve = original
    assert solve_reference(dims)['velocity_km_s'].shape == (3, 2, 2)

    # The allowed positive 1x1 grid has no finite-difference rows.  The oracle
    # must still construct an explicit (0, 1) L matrix and produce finite output.
    edge = dict(inp)
    edge.update({
        'grid': {'nx': 1, 'ny': 1}, 'periods_s': [10.0],
        'ray_matrix': [[[1.0]]], 'observed_slowness_s_per_km': [[0.2]],
        'data_variance_s2_per_km2': [[1e-4]],
        'initial_model_slowness_s_per_km': [0.2],
        'model_variance_s2_per_km2': [0.01], 'iterations': 2,
    })
    edge_ref = solve_reference(edge)
    assert edge_ref['velocity_km_s'].shape == (1, 1, 1)
    assert np.all(np.isfinite(edge_ref['roughness_slowness']))

# Reproduce the evaluator's staging contract with a solver attempting oracle,
# hidden-fixture discovery, and mutation tricks. Only solve.py and input are
# mounted in the private namespace.
with tempfile.TemporaryDirectory() as stage_dir:
    stage = Path(stage_dir)
    (stage / 'input.json').write_text('{}')
    hidden_path = root / 'reference/hidden_case.json'
    hidden_before = hidden_path.read_bytes()
    (stage / 'solve.py').write_text(f"""import pathlib, subprocess, sys
p = pathlib.Path('oracle_access.txt')
hidden = pathlib.Path({str(hidden_path)!r})
seen = pathlib.Path('_oracle.py').exists() or hidden.exists()
try:
    __import__('_oracle')
    seen = True
except Exception:
    pass
try:
    r = subprocess.run([sys.executable, '_oracle.py'], check=False, capture_output=True)
    seen = seen or r.returncode == 0
except Exception:
    pass
try:
    hidden.write_text('tampered')
except Exception:
    pass
p.write_text('1' if seen else '0')
""")
    (stage / 'run').mkdir()
    if shutil.which('bwrap') is None:
        raise AssertionError('bwrap is required for the isolation adversarial test')
    subprocess.run([
        'bwrap', '--die-with-parent', '--unshare-all',
        '--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib',
        '--ro-bind', '/lib64', '/lib64', '--ro-bind', '/bin', '/bin',
        '--ro-bind', '/sbin', '/sbin', '--ro-bind', '/etc', '/etc',
        '--dev', '/dev', '--proc', '/proc', '--tmpfs', '/tmp', '--chdir', '/work',
        '--bind', str(stage), '/work', '/usr/bin/python3', '/work/solve.py',
        '--input', '/work/input.json', '--output', '/work/run',
        '--diagnostics', '/work/run/diagnostics.json'], check=False)
    assert (stage / 'oracle_access.txt').read_text() == '0'
    assert not (stage / '_oracle.py').exists()
    assert hidden_path.read_bytes() == hidden_before

source = (root / 'main.py').read_text()
assert 'scripts/oracle.py' in source and 'bwrap --die-with-parent --unshare-all' in source and 'returncode' in source
assert "session.write_file(f'{out}/_oracle.py'" not in source
eval_source = source[source.index('async def evaluate'):]
assert eval_source.index('await _run_candidate_isolated') < eval_source.index('ref=oracle_mod.solve_reference(vis)')
print('adversarial checks passed')
