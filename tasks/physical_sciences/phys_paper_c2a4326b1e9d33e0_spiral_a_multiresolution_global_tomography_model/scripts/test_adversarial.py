"""Local grader fixtures: perfect solver, malformed output, unit error, shortcut."""
import asyncio, csv, json, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path('/data/ale/agents-last-exam')))
sys.path.insert(0, str(ROOT))
import main

class S:
    def __init__(self, files): self.files = files
    async def read_file(self, p): return self.files[p]

def run_solver(src, inp, out, tr):
    (out.parent/'solver.py').write_text(src)
    q = subprocess.run([sys.executable, str(out.parent/'solver.py'), '--input', str(inp), '--result', str(out), '--trace', str(tr)], check=False, capture_output=True, text=True, timeout=10)
    assert q.returncode == 0, q.stderr

def score(files): return asyncio.run(main.evaluate(type('Cfg', (), {'metadata': files})(), S(files))) [0]

def main_test():
    inp = ROOT/'input/problem.json'; hidden = ROOT/'reference/hidden_problem.json'; ref = ROOT/'reference/reference.json'
    # Science regression: the ray-time convention is the support function of
    # the phase-slowness surface, not the old reciprocal-cosine surrogate.
    # Use a deliberately stronger (still positive-definite) VTI cell so the
    # two conventions cannot coincide within the normal output tolerance.
    case = {'Vpv': 5.0, 'Vph': 8.0, 'Vsv': 3.0, 'Vsh': 4.0,
            'delta': 0.1, 'theta_deg': 45.0, 'length_km': 1.0}
    theta = main.math.radians(case['theta_deg'])
    lo = max(-main.math.pi/2 + 1e-6, theta-main.math.pi/2 + 1e-6)
    hi = min(main.math.pi/2 - 1e-6, theta+main.math.pi/2 - 1e-6)
    samples = []
    for j in range(20001):
        ph = lo + j*(hi-lo)/20000
        q = dict(case); q['theta_deg'] = main.math.degrees(ph)
        v = main._phase_segment(q)['v_qp_km_s']; c = main.math.cos(ph-theta)
        if c > 0: samples.append(c/v)
    support = max(samples)
    reciprocal_values = []
    for j in range(20001):
        ph = lo + j*(hi-lo)/20000
        q = dict(case); q['theta_deg'] = main.math.degrees(ph)
        v = main._phase_segment(q)['v_qp_km_s']; c = main.math.cos(ph-theta)
        if c > 0: reciprocal_values.append(1.0/(v*c))
    reciprocal = min(reciprocal_values)
    assert abs(main._ray_slowness(case, 'v_qp_km_s') - support) < 2e-6
    assert abs(main._ray_slowness(case, 'v_qp_km_s') - reciprocal) > 1e-3
    perfect = (ROOT/'scripts/oracle.py').read_text()
    with tempfile.TemporaryDirectory() as td:
        p=Path(td); vo=p/'v.json'; vt=p/'v.csv'; ho=p/'h.json'; ht=p/'h.csv'
        run_solver(perfect, inp, vo, vt); run_solver(perfect, hidden, ho, ht)
        base={'reference_dir':'ref','result_file':'out.json','trace_file':'trace.csv','solver_file':'solver.py'}
        files={**base,'ref/reference.json':ref.read_text(),'ref/hidden_problem.json':hidden.read_text(),'out.json':vo.read_text(),'trace.csv':vt.read_text(),'solver.py':perfect}
        assert score(files) > 0.999, score(files)
        bad={**files,'out.json':'{}','trace.csv':''}
        assert score(bad) == 0.0
        # Pre-parse byte caps reject oversized result/trace payloads.
        oversized={**files,'out.json':'{"x":"' + ('z'*300_000) + '"}'}
        assert score(oversized) == 0.0
        # Hidden staging rejects files other than solver/input/results, even
        # when the solver itself returns successfully.
        extra = perfect + "\nopen('junk.bin','wb').write(b'x')\n"
        assert score({**files,'solver.py':extra}) < 0.80
        wrong=json.loads(vo.read_text());
        for x in wrong['visible']: x['t_qp_s']*=1000; x['t_qsv_s']*=1000; x['t_qsh_s']*=1000
        bad={**files,'out.json':json.dumps(wrong)}
        assert 0.30 < score(bad) < 0.40, score(bad)
        # A plausible shortcut: visible oracle values are returned regardless of input.
        shortcut=perfect.replace("json.load(open(x.input))", "json.load(open(x.input))")
        # Make hidden execution fail while leaving visible artifacts perfect.
        shortcut += '\n# shortcut marker\n'
        # It still computes generically, so replace solver body with visible constants.
        shortcut = "import json,sys\nfrom pathlib import Path\nV=json.load(open('"+str(vo)+"'))\n" + "def main():\n import argparse; a=argparse.ArgumentParser(); a.add_argument('--input'); a.add_argument('--result'); a.add_argument('--trace'); x=a.parse_args(); json.dump(V,open(x.result,'w')); Path(x.trace).write_text('')\nif __name__=='__main__': main()\n"
        bad={**files,'solver.py':shortcut}
        s=score(bad); assert s < 0.85, s
        # A solver must not be able to escape the staged hidden sandbox and
        # read evaluator-only reference/reference.json through inherited cwd.
        # This emits the visible oracle only if that directory is reachable;
        # the isolated grader should therefore give no hidden credit.
        leak = """import argparse, json
from pathlib import Path
a=argparse.ArgumentParser(); a.add_argument('--input'); a.add_argument('--result'); a.add_argument('--trace'); x=a.parse_args()
ref=Path('reference/reference.json')
if ref.exists():
    data=json.loads(ref.read_text()); json.dump(data, open(x.result,'w')); Path(x.trace).write_text('')
else:
    json.dump({}, open(x.result,'w')); Path(x.trace).write_text('')
"""
        s=score({**files,'solver.py':leak})
        assert s < 0.80, s
        # Dynamic third-party imports are unavailable in the isolated runtime;
        # a regex-only source scan would miss this importlib form.
        dynamic_import = """import argparse, importlib
importlib.import_module('numpy')
"""
        s=score({**files,'solver.py':dynamic_import})
        assert s < 0.80, s
        # The hidden subprocess has no network namespace, even when the solver
        # uses the standard-library socket API directly.
        network = """import argparse, socket
a=argparse.ArgumentParser(); a.add_argument('--input'); a.add_argument('--result'); a.add_argument('--trace'); x=a.parse_args()
socket.create_connection(('1.1.1.1', 80), timeout=1)
"""
        s=score({**files,'solver.py':network})
        assert s < 0.80, s
        # Absolute paths and parent traversal cannot reach evaluator-only
        # reference files from the staged filesystem.
        traversal = f"""import argparse, json
from pathlib import Path
a=argparse.ArgumentParser(); a.add_argument('--input'); a.add_argument('--result'); a.add_argument('--trace'); x=a.parse_args()
target=Path({str(ROOT / 'reference' / 'reference.json')!r})
if not target.exists():
    target=Path('../../reference/reference.json')
data=json.loads(target.read_text()); json.dump(data, open(x.result,'w')); Path(x.trace).write_text('')
"""
        s=score({**files,'solver.py':traversal})
        assert s < 0.80, s
        # Fork/spin and allocation adversaries must hit process/resource limits
        # and leave no descendants capable of modifying staged outputs.
        fork_spin = """import argparse, os, time
a=argparse.ArgumentParser(); a.add_argument('--input'); a.add_argument('--result'); a.add_argument('--trace'); x=a.parse_args()
for _ in range(64):
    try: os.fork()
    except OSError: break
while True: time.sleep(1)
"""
        assert score({**files,'solver.py':fork_spin}) == 0.0
        allocator = """import argparse
a=argparse.ArgumentParser(); a.add_argument('--input'); a.add_argument('--result'); a.add_argument('--trace'); a.parse_args()
blob=[]
while True: blob.append(bytearray(16*1024*1024))
"""
        assert score({**files,'solver.py':allocator}) == 0.0
    print('adversarial checks passed')

if __name__=='__main__': main_test()
