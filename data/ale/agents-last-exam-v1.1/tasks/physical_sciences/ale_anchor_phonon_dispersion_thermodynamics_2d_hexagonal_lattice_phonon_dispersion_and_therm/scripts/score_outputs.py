from __future__ import annotations
import argparse, importlib.util, json, os, secrets, shutil, signal, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

REQUIRED=('diatomic_1d.npz','dispersion_2d.npz','dos.npz','thermodynamics.npz','results.json','solver.py')
EVAL_FILES=REQUIRED[:-1]
WEIGHTS={'tier1':0.20,'tier2':0.25,'dos':0.20,'thermo':0.20,'summary':0.05,'hidden':0.10}

def _load(p):
    with np.load(p,allow_pickle=False) as z: return {k:z[k] for k in z.files}
def _same(a,b,rtol,atol):
    return a.shape==b.shape and np.all(np.isfinite(a)) and np.allclose(a,b,rtol=rtol,atol=atol,equal_nan=False)
def _json(p):
    try:
        x=json.loads(p.read_text())
        return x if isinstance(x,dict) else None
    except (OSError, ValueError, TypeError): return None

def _finite_json(x):
    if isinstance(x,bool) or x is None or isinstance(x,str): return True
    if isinstance(x,(int,float)): return bool(np.isfinite(x))
    if isinstance(x,list): return all(_finite_json(v) for v in x)
    if isinstance(x,dict): return all(isinstance(k,str) and _finite_json(v) for k,v in x.items())
    return False

def _schema_gate(out, require_solver=False):
    """Reject the complete output tree before awarding any component."""
    out=Path(out)
    needed=REQUIRED if require_solver else EVAL_FILES
    if any(not (out/n).is_file() for n in needed): return False
    specs={
      'diatomic_1d.npz':({'q_points','omega_acoustic','omega_optical'},{'q_points':(121,), 'omega_acoustic':(121,), 'omega_optical':(121,)}),
      'dispersion_2d.npz':({'q_points','q_distance','frequencies'},{'q_points':(160,2), 'q_distance':(160,), 'frequencies':(160,4)}),
      'dos.npz':({'omega_bins','dos'},{'omega_bins':(320,), 'dos':(320,)}),
      'thermodynamics.npz':({'temperatures','energy','heat_capacity','free_energy'},{'temperatures':(18,), 'energy':(18,), 'heat_capacity':(18,), 'free_energy':(18,)})}
    try:
      for name,(keys,shapes) in specs.items():
        data=_load(out/name)
        if set(data)!=keys: return False
        for key,shape in shapes.items():
          if data[key].shape!=shape or not np.issubdtype(data[key].dtype,np.number) or not np.all(np.isfinite(data[key])): return False
      result=_json(out/'results.json')
      expected={'tier1':{'optical_frequency_at_gamma','max_acoustic_at_gamma','num_q_points'},'tier2':{'num_q_points','num_branches','freq_range','negative_eigenvalue_count','acoustic_at_gamma'},'tier3':{'omega_max','dos_integral','cv_high_T','cv_low_T','num_temperatures','q_grid_size'}}
      if result is None or set(result)!=set(expected) or not _finite_json(result): return False
      for tier,fields in expected.items():
        if not isinstance(result[tier],dict) or set(result[tier])!=fields: return False
      for tier,key in (('tier1','num_q_points'),('tier2','num_q_points'),('tier2','num_branches'),('tier2','negative_eigenvalue_count'),('tier3','num_temperatures')):
        if type(result[tier][key]) is not int: return False
      if type(result['tier1']['num_q_points']) is not int or type(result['tier3']['q_grid_size']) is not list or len(result['tier3']['q_grid_size'])!=2: return False
      if any(type(v) is not int for v in result['tier3']['q_grid_size']): return False
      if not isinstance(result['tier2']['freq_range'],list) or len(result['tier2']['freq_range'])!=2: return False
      if not isinstance(result['tier2']['acoustic_at_gamma'],list) or len(result['tier2']['acoustic_at_gamma'])!=2: return False
      numeric=(int,float)
      for tier, keys in (('tier1',('optical_frequency_at_gamma','max_acoustic_at_gamma')), ('tier2',('freq_range','acoustic_at_gamma')), ('tier3',('omega_max','dos_integral','cv_high_T','cv_low_T'))):
        vals=[]
        for key in keys:
          value=result[tier][key]; vals.extend(value if isinstance(value,list) else [value])
        if any(type(v) not in numeric for v in vals): return False
      # Integer fields also have governed values; this prevents 121.9/"18" probes.
      if result['tier1']['num_q_points']!=121 or result['tier2']['num_q_points']!=160 or result['tier2']['num_branches']!=4 or result['tier3']['num_temperatures']!=18 or result['tier3']['q_grid_size']!=[40,40]: return False
      return True
    except (OSError, ValueError, TypeError, KeyError, EOFError): return False

def _oracle_module():
    """Load the independent host-side oracle; never stage it for candidates."""
    here=Path(__file__).resolve().parent
    source=here/'_eval_oracle.py'
    if not source.is_file(): source=here/'oracle.py'
    spec=importlib.util.spec_from_file_location('_phonon_independent_oracle',source)
    if spec is None or spec.loader is None: raise RuntimeError('independent oracle unavailable')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def _hidden_config(ref):
    """Create a fresh valid probe configuration without storing its answers.

    Only positive spring constants and masses are sampled, which keeps the
    Gram dynamical matrix stable while making every probe different.  The
    candidate receives this config before execution; expected arrays are
    generated only after the candidate exits.
    """
    candidates=(ref.parent/'input/model_config.json', Path(__file__).resolve().parents[1]/'input/model_config.json')
    base=None
    for path in candidates:
        if path.is_file(): base=json.loads(path.read_text()); break
    if base is None: raise RuntimeError('visible model config unavailable')
    rng=secrets.SystemRandom()
    m2=round(rng.uniform(1.20,2.30),6)
    k3=round(rng.uniform(0.12,0.34),6)
    base['m2']=m2; base['chain_m2']=m2; base['k3']=k3
    # Evaluator-only replay record; this path is never copied into the guest.
    log_path=Path(os.environ.get('PHONON_EVALUATOR_RESULTS','/tmp/phonon_evaluator_results.jsonl'))
    try:
        with log_path.open('a',encoding='utf-8') as log:
            log.write(json.dumps({'hidden_config':base},sort_keys=True)+'\n')
    except OSError:
        pass
    return base

def _visible(out,ref):
    scores={k:0.0 for k in ('tier1','tier2','dos','thermo','summary')}
    if not _schema_gate(out,require_solver=False): return scores
    try:
      a1,r1=_load(out/'diatomic_1d.npz'),_load(ref/'diatomic_1d.npz')
      if set(a1)=={'q_points','omega_acoustic','omega_optical'} and all(_same(a1[k],r1[k],2e-7,2e-9) for k in a1): scores['tier1']=1.0
      a2,r2=_load(out/'dispersion_2d.npz'),_load(ref/'dispersion_2d.npz')
      if set(a2)=={'q_points','q_distance','frequencies'} and _same(a2['q_points'],r2['q_points'],2e-10,2e-11) and _same(a2['q_distance'],r2['q_distance'],2e-8,2e-9) and _same(a2['frequencies'],r2['frequencies'],3e-5,3e-7): scores['tier2']=1.0
      ad,rd=_load(out/'dos.npz'),_load(ref/'dos.npz')
      if set(ad)=={'omega_bins','dos'} and _same(ad['omega_bins'],rd['omega_bins'],1e-10,1e-12) and _same(ad['dos'],rd['dos'],4e-3,2e-5): scores['dos']=1.0
      at,rt=_load(out/'thermodynamics.npz'),_load(ref/'thermodynamics.npz')
      if set(at)=={'temperatures','energy','heat_capacity','free_energy'} and all(_same(at[k],rt[k],5e-3,2e-5) for k in at): scores['thermo']=1.0
      js,jr=_json(out/'results.json'),_json(ref/'results.json')
      def smatch(x,y):
          if not isinstance(x,dict) or set(x)!=set(y): return False
          for kk in x:
              if isinstance(y[kk],dict):
                  if not smatch(x[kk],y[kk]): return False
              elif isinstance(y[kk],list):
                  if not isinstance(x[kk],list) or len(x[kk])!=len(y[kk]): return False
                  if not all(np.isclose(float(a),float(b),rtol=2e-5,atol=2e-7) for a,b in zip(x[kk],y[kk])): return False
              elif isinstance(y[kk],(int,float)) and not isinstance(y[kk],bool):
                  if isinstance(y[kk],int) and not isinstance(y[kk],float):
                      if int(x[kk])!=int(y[kk]): return False
                  elif not np.isclose(float(x[kk]),float(y[kk]),rtol=2e-5,atol=2e-7): return False
              elif x[kk]!=y[kk]: return False
          return True
      invariant = False
      try:
          inv_d = _load(out/'dos.npz'); inv_t = _load(out/'thermodynamics.npz'); inv_p = _load(out/'dispersion_2d.npz')
          invariant = (abs(float(np.trapezoid(inv_d['dos'],inv_d['omega_bins']))-4.0) <= 0.02 and
                       np.all(np.isfinite(inv_p['frequencies'])) and np.min(inv_p['frequencies']) >= -1e-10 and
                       np.all(np.diff(inv_t['energy']) >= -1e-8) and np.all(np.diff(inv_t['free_energy']) <= 1e-8) and
                       isinstance(js,dict) and int(js['tier2']['negative_eigenvalue_count']) == 0 and
                       max(map(abs,js['tier2']['acoustic_at_gamma'])) <= 1e-6)
      except Exception: invariant = False
      if js is not None and jr is not None and smatch(js,jr) and invariant: scores['summary']=1.0
    except Exception: pass
    return scores

def score_visible(out,ref):
    """Pure host-side comparison used by the framework evaluator."""
    return _visible(Path(out),Path(ref))

def weighted_score(components, hidden_pass):
    if not hidden_pass:
        return 0.0
    c=dict(components); c['hidden']=1.0 if hidden_pass else 0.0
    return float(min(1.0,max(0.0,sum(WEIGHTS[k]*c[k] for k in WEIGHTS))))

def _hidden(out,ref):
    try:
      # A static hidden_case (or hidden_config) defeats the protocol: a
      # candidate can read the expected arrays before it is run.  Hidden
      # answers are deliberately absent from the package and are generated
      # below only after the candidate subprocess has terminated.
      if (Path(ref)/'hidden_case').exists() or (Path(ref)/'hidden_config.json').exists():
          return 0.0
      with tempfile.TemporaryDirectory(prefix='phonon_hidden_') as td:
        # Stage both the candidate and evaluator inputs under a fresh root.
        # The original task tree has output/ next to reference/, so running the
        # original solver in place would let ``Path(__file__).parents[1] /
        # reference`` exfiltrate hidden answers.  Copy regular files by value
        # and omit links so no path in the staged tree points back to the task.
        root=Path(td)
        candidate=root/'candidate'; candidate.mkdir()
        def copy_regular_tree(src,dst):
            for entry in src.iterdir():
                if entry.is_symlink():
                    continue
                target=dst/entry.name
                if entry.is_dir():
                    target.mkdir()
                    copy_regular_tree(entry,target)
                elif entry.is_file():
                    shutil.copyfile(entry,target)
        copy_regular_tree(out,candidate)
        sandbox=root/'sandbox'; sandbox.mkdir()
        hidden_cfg=_hidden_config(Path(ref))
        cfg=sandbox/'config.json'; cfg.write_text(json.dumps(hidden_cfg,sort_keys=True)+'\n')
        hout=sandbox/'output'; hout.mkdir()
        # Import the oracle before untrusted code starts, while its source is
        # still under evaluator control.  Only the numerical expectation is
        # deliberately deferred until after solver.py exits.
        oracle=_oracle_module()
        command=[sys.executable,str(candidate/'solver.py'),'--config',str(cfg),'--output-dir',str(hout)]
        proc=subprocess.Popen(command,cwd=str(candidate),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        try:
            _,_=proc.communicate(timeout=80)
        except subprocess.TimeoutExpired:
            try: os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            proc.communicate()
            return 0.0
        finally:
            # Reap any helper processes a malicious solver left behind before
            # the evaluator creates its post-execution expected directory.
            try: os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError: pass
        if proc.returncode!=0: return 0.0

        # Snapshot candidate outputs before creating any evaluator artifacts.
        # A solver that daemonizes a helper cannot race the later comparison
        # by mutating the live output directory.
        candidate_snapshot=root/'candidate_output_snapshot'; candidate_snapshot.mkdir()
        copy_regular_tree(hout,candidate_snapshot)

        # Compute the expected values in a fresh directory after execution.
        # There was no answer file on disk while solver.py was running.
        expected=root/'post_execution_expected'; expected.mkdir()
        oracle.write_outputs(oracle.solve_reference(hidden_cfg),expected)
        s=_visible(candidate_snapshot,expected)
        return 1.0 if all(s[k]>=1.0 for k in ('tier1','tier2','dos','thermo','summary')) else 0.0
    except Exception: return 0.0

def score(out,ref):
    out=Path(out); ref=Path(ref)
    if any(not (out/n).is_file() for n in REQUIRED): return {'score':0.0,'components':{k:0.0 for k in WEIGHTS},'reason':'missing required output'}
    if any((out/n).stat().st_size>6_000_000 for n in REQUIRED): return {'score':0.0,'components':{k:0.0 for k in WEIGHTS},'reason':'oversized output'}
    if not _schema_gate(out,require_solver=True): return {'score':0.0,'components':{k:0.0 for k in WEIGHTS},'reason':'schema gate failed'}
    c=_visible(out,ref); c['hidden']=_hidden(out,ref)
    # Hidden recomputation is a release gate: visible artifacts alone can
    # never produce a passing score.
    total=weighted_score(c,c['hidden']>=1.0)
    return {'score':float(total),'components':c}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output-dir',required=True); p.add_argument('--reference-dir',required=True); a=p.parse_args(); print(json.dumps(score(a.output_dir,a.reference_dir),sort_keys=True))
