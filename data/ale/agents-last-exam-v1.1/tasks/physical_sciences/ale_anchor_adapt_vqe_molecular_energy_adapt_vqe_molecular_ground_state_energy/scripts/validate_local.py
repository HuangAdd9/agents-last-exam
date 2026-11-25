"""Focused genuine-candidate and adversarial scientific grader checks."""
from __future__ import annotations
import ast, copy, json, math, subprocess, sys, tempfile
from pathlib import Path
from score_outputs import score_payloads

ROOT=Path(__file__).resolve().parents[1]; REF=json.loads((ROOT/"reference/results.json").read_text())
configs={"visible":json.loads((ROOT/"input/field_config.json").read_text()),"hidden_1":json.loads((ROOT/"reference/hidden_case.json").read_text()),"hidden_2":json.loads((ROOT/"reference/hidden_case_2.json").read_text())}
payloads={}
for name in ("h2_hamiltonian.json","lih_hamiltonian.json","beh2_hamiltonian.json"):
    p=json.loads((ROOT/"input"/name).read_text());payloads[p["molecule"]]=p
CTX={"payloads":payloads,"configs":configs}
with tempfile.TemporaryDirectory() as td:
    tmp=Path(td);outs=[]
    for i,cfg in enumerate((ROOT/"input/field_config.json",ROOT/"reference/hidden_case.json",ROOT/"reference/hidden_case_2.json")):
        out=tmp/f"out{i}.json";subprocess.run([sys.executable,str(ROOT/"scripts/perfect_solver_fixture.py"),"--input-dir",str(ROOT/"input"),"--field-config",str(cfg),"--output",str(out)],check=True);outs.append(json.loads(out.read_text()))
    visible=outs[0];hidden=outs[1:]
    perfect=score_payloads(visible,hidden,REF,CTX);assert perfect["score"]==1.0,perfect

    # Former principal shortcut: exact oracle scalars plus universal dummy metadata.
    shortcut={}
    for tier,row in REF["visible"].items():
        shortcut[tier]={**row,"n_parameters":1,"adapt_iterations":1,"operator_sequence":["S(0->1)"],"ansatz_parameters":[0.0],"adapt_history":[{"operator":"S(0->1)","gradient":0.0,"energy_ha":row["energy_ha"],"parameters":[0.0]}]}
    exact_shortcut=score_payloads(shortcut,hidden,REF,CTX);assert exact_shortcut["score"]<=.60 and exact_shortcut["components"]["ansatz"]<.20,exact_shortcut

    # Tier 1 is a small VQE construction, not an unrestricted ADAPT history
    # mislabeled as VQE.  The tier-specific bound must reject even otherwise
    # well-shaped payloads before reconstruction.
    vqe_overlong=copy.deepcopy(visible); vqe_row=vqe_overlong["tier1"]
    vqe_ops=__import__("score_outputs")._pool(payloads["H2"])
    vqe_row["operator_sequence"]=vqe_ops; vqe_row["n_parameters"]=vqe_row["adapt_iterations"]=len(vqe_ops)
    vqe_row["ansatz_parameters"]=[0.0]*len(vqe_ops)
    vqe_row["adapt_history"]=[{"operator":op,"gradient":0.0,"energy_ha":0.0,"parameters":[0.0]*(i+1)} for i,op in enumerate(vqe_ops)]
    vqe_overlong_result=score_payloads(vqe_overlong,hidden,REF,CTX)
    assert vqe_overlong_result["score"]==0.0 and any("VQE construction" in x for x in vqe_overlong_result["reasons"]),vqe_overlong_result

    fabricated_gradient=copy.deepcopy(visible);fabricated_gradient["tier2"]["adapt_history"][0]["gradient"]+=.1
    bad_gradient=score_payloads(fabricated_gradient,hidden,REF,CTX);assert bad_gradient["score"]<=.60,bad_gradient
    fabricated_parameter=copy.deepcopy(visible);fabricated_parameter["tier3"]["ansatz_parameters"][0]+=.2
    bad_parameter=score_payloads(fabricated_parameter,hidden,REF,CTX);assert bad_parameter["score"]<=.60,bad_parameter
    fabricated_invariant=copy.deepcopy(visible);fabricated_invariant["tier1"]["number_expectation"]=0.0
    bad_invariant=score_payloads(fabricated_invariant,hidden,REF,CTX);assert bad_invariant["score"]<=.60 and bad_invariant["components"]["number_sector"]<.10,bad_invariant
    malformed_operator=copy.deepcopy(visible);malformed_operator["tier1"]["operator_sequence"]=["S(0->999)"]
    bad_operator=score_payloads(malformed_operator,hidden,REF,CTX);assert bad_operator["score"]<=.60,bad_operator
    copied_hidden=score_payloads(visible,[visible,visible],REF,CTX);assert copied_hidden["components"]["hidden_energy"]<.30,copied_hidden
    partial_hidden=copy.deepcopy(hidden); partial_hidden[0]["tier1"]["energy_ha"] += .01
    partial=score_payloads(visible,partial_hidden,REF,CTX)
    assert partial["components"]["hidden_energy"] < .30 and partial["score"] <= .60, partial

    # A 33-operator payload must be rejected before reconstruction, even when
    # every operator is unique and all lengths agree.
    too_long=copy.deepcopy(visible); row=too_long["tier2"]; ops=__import__("score_outputs")._pool(payloads["LiH"])[:33]
    row["operator_sequence"]=ops; row["n_parameters"]=row["adapt_iterations"]=33
    row["ansatz_parameters"]=[0.0]*33
    row["adapt_history"]=[{"operator":op,"gradient":0.0,"energy_ha":0.0,"parameters":[0.0]*(i+1)} for i,op in enumerate(ops)]
    too_long_result=score_payloads(too_long,hidden,REF,CTX)
    assert too_long_result["score"]==0.0 and any("at most 32" in x for x in too_long_result["reasons"]),too_long_result

    # Zero parameters are valid when the independently recomputed initial
    # gradient is already below the stopping threshold.
    diagonal_payload={"molecule":"H2","n_qubits":2,"n_electrons":1,"pauli_terms":{"II":0.0,"ZI":1.0}}
    diagonal_config={"identity_shift_ha":0.0,"site_z_coefficients_ha":{"H2":[0.0,0.0]}}
    zero_row={"operator_sequence":[],"adapt_history":[],"ansatz_parameters":[],"n_parameters":0,"adapt_iterations":0}
    zero_state=__import__("score_outputs")._reconstruct(zero_row,diagonal_payload,diagonal_config)
    assert abs(zero_state["energy_ha"]+1.0)<1e-12,zero_state

    # The documented stopping threshold is exactly 1e-3 Hartree.  A zero-
    # length ansatz just above that threshold must not be accepted.
    near_threshold_payload={"molecule":"H2","n_qubits":2,"n_electrons":1,"pauli_terms":{"II":0.0,"XX":0.00055}}
    near_threshold_config={"identity_shift_ha":0.0,"site_z_coefficients_ha":{"H2":[0.0,0.0]}}
    try:
        __import__("score_outputs")._reconstruct(zero_row,near_threshold_payload,near_threshold_config)
    except ValueError as exc:
        assert "initial gradient threshold" in str(exc),exc
    else:
        raise AssertionError("zero-length ansatz accepted above the 1e-3 threshold")

    # Regression for the former <=8-parameter Hessian shortcut: a nine-single
    # ansatz has positive coordinate curvatures but a negative mixed direction.
    # The synthetic payload keeps all ADAPT snapshots stationary at the HF
    # state while the dense Hessian has one negative eigenvalue.
    scorer=__import__("score_outputs")
    saddle_payload={"molecule":"SYN","n_qubits":10,"n_electrons":1,"pauli_terms":{}}
    saddle_config={"identity_shift_ha":0.0,"site_z_coefficients_ha":{"SYN":[0.0]*10}}
    saddle_ops=[f"S(0->{a})" for a in range(1,10)]
    saddle_sec=__import__("numpy").array([s for s in range(1<<10) if s.bit_count()==1],dtype=__import__("numpy").int64)
    saddle_h=__import__("numpy").zeros((len(saddle_sec),len(saddle_sec)),complex)
    hf_idx=int(__import__("numpy").where(saddle_sec==512)[0][0])
    for i in range(len(saddle_sec)):
        if i!=hf_idx:saddle_h[i,i]=1.0
    virtual_indices=[int(__import__("numpy").where(saddle_sec==(1<<(9-a)))[0][0]) for a in range(1,10)]
    for i in virtual_indices:
        for j in virtual_indices:
            if i!=j:saddle_h[i,j]=-0.30
    original_ham,original_pool,original_grad=scorer._ham,scorer._pool,scorer._grad
    scorer._ham=lambda payload,config:(saddle_h,saddle_sec)
    scorer._pool=lambda payload:list(saddle_ops)
    scorer._grad=lambda h,sec,state,op: 0.02-0.001*saddle_ops.index(op)
    saddle_row={"operator_sequence":saddle_ops,"ansatz_parameters":[0.0]*9,"n_parameters":9,"adapt_iterations":9,
                "adapt_history":[{"operator":op,"gradient":0.02-0.001*i,"energy_ha":0.0,"parameters":[0.0]*(i+1)} for i,op in enumerate(saddle_ops)]}
    try:
        try:
            scorer._reconstruct(saddle_row,saddle_payload,saddle_config)
        except ValueError as exc:
            assert "saddle" in str(exc) or "maximal" in str(exc),exc
        else:
            raise AssertionError("nine-parameter negative mixed-curvature saddle was accepted")
    finally:
        scorer._ham,scorer._pool,scorer._grad=original_ham,original_pool,original_grad

    # Candidate subprocesses must cleanly reject malformed configurations and
    # Hamiltonians, rather than silently solving a different problem.
    malformed_cases=[]
    bad=copy.deepcopy(configs["visible"]);bad["identity_shift_ha"]=float("nan");malformed_cases.append(("nonfinite",bad,None))
    bad=copy.deepcopy(configs["visible"]);bad["site_z_coefficients_ha"]["H2"]=bad["site_z_coefficients_ha"]["H2"][:-1];malformed_cases.append(("wrong_vector_length",bad,None))
    bad=copy.deepcopy(configs["visible"]);bad["field_model"]="wrong";malformed_cases.append(("wrong_model",bad,None))
    bad=copy.deepcopy(configs["visible"]);bad["electron_sector_required"]=False;malformed_cases.append(("wrong_sector",bad,None))
    bad=copy.deepcopy(configs["visible"]);bad["site_z_coefficients_ha"]["H2"][0]=float("inf");malformed_cases.append(("nonfinite_site_field",bad,None))
    bad=copy.deepcopy(configs["visible"]);bad["site_z_coefficients_ha"]["LiH"][0]=float("nan");malformed_cases.append(("nan_site_field",bad,None))
    bad=copy.deepcopy(configs["visible"]);bad["hartree_to_ev"]=0.0;malformed_cases.append(("zero_conversion",bad,None))
    bad=copy.deepcopy(configs["visible"]);bad["hartree_to_ev"]=-27.2;malformed_cases.append(("negative_conversion",bad,None))
    bad=copy.deepcopy(configs["visible"]);bad["hartree_to_ev"]=float("nan");malformed_cases.append(("nan_conversion",bad,None))
    bad=copy.deepcopy(configs["visible"]);bad["hartree_to_ev"]=float("inf");malformed_cases.append(("inf_conversion",bad,None))
    bad=copy.deepcopy(configs["visible"]);bad.pop("hartree_to_ev");malformed_cases.append(("missing_conversion",bad,None))
    bad=copy.deepcopy(configs["visible"]);bad["unexpected_schema_field"]=1;malformed_cases.append(("extra_config_schema",bad,None))
    for molecule,filename,bad_word in (("H2","h2_hamiltonian.json","QIII"),("LiH","lih_hamiltonian.json","QIIIIIIIII"),("BeH2","beh2_hamiltonian.json","QIIIIIIIII")):
        bad_ham=copy.deepcopy(payloads[molecule]);bad_ham["n_pauli_terms"] += 1;malformed_cases.append((f"bad_hamiltonian_metadata_{molecule}",None,(filename,bad_ham)))
        bad_ham=copy.deepcopy(payloads[molecule]);bad_ham["pauli_terms"][bad_word]=0.1;malformed_cases.append((f"bad_pauli_term_{molecule}",None,(filename,bad_ham)))
        first_word=next(iter(bad_ham["pauli_terms"]))
        bad_ham=copy.deepcopy(payloads[molecule]);bad_ham["pauli_terms"][first_word]=float("nan");malformed_cases.append((f"nonfinite_pauli_nan_{molecule}",None,(filename,bad_ham)))
        bad_ham=copy.deepcopy(payloads[molecule]);bad_ham["pauli_terms"][first_word]=float("inf");malformed_cases.append((f"nonfinite_pauli_inf_{molecule}",None,(filename,bad_ham)))
        bad_ham=copy.deepcopy(payloads[molecule]);bad_ham["molecule"]="WrongMolecule";malformed_cases.append((f"wrong_molecule_{molecule}",None,(filename,bad_ham)))
        bad_ham=copy.deepcopy(payloads[molecule]);bad_ham.pop("basis");malformed_cases.append((f"missing_schema_field_{molecule}",None,(filename,bad_ham)))
        bad_ham=copy.deepcopy(payloads[molecule]);bad_ham["n_qubits"]=0;malformed_cases.append((f"nonpositive_qubits_{molecule}",None,(filename,bad_ham)))
        bad_ham=copy.deepcopy(payloads[molecule]);bad_ham["n_electrons"]=0;malformed_cases.append((f"nonpositive_electrons_{molecule}",None,(filename,bad_ham)))
        if molecule == "H2":
            bad_ham=copy.deepcopy(payloads[molecule]);bad_ham["n_qubits"]=-1;malformed_cases.append(("negative_qubits_H2",None,(filename,bad_ham)))
            bad_ham=copy.deepcopy(payloads[molecule]);bad_ham["n_electrons"]=-1;malformed_cases.append(("negative_electrons_H2",None,(filename,bad_ham)))
        bad_ham=copy.deepcopy(payloads[molecule]);bad_ham["n_electrons"]=bad_ham["n_qubits"];malformed_cases.append((f"inconsistent_electron_sector_{molecule}",None,(filename,bad_ham)))
    for label,bad_cfg,bad_ham_entry in malformed_cases:
        case_dir=tmp/label; case_dir.mkdir()
        for source in (ROOT/"input").glob("*.json"):
            (case_dir/source.name).write_text(source.read_text())
        cfg_path=case_dir/"field_config.json"
        if bad_cfg is not None: cfg_path.write_text(json.dumps(bad_cfg,allow_nan=True))
        if bad_ham_entry is not None: (case_dir/bad_ham_entry[0]).write_text(json.dumps(bad_ham_entry[1]))
        badout=case_dir/"out.json"
        proc=subprocess.run([sys.executable,str(ROOT/"scripts/perfect_solver_fixture.py"),"--input-dir",str(case_dir),"--field-config",str(cfg_path),"--output",str(badout)],capture_output=True,text=True)
        assert proc.returncode!=0 and not badout.exists(),(label,proc.stdout,proc.stderr)

    # Regression cases for transitive import escapes are kept beside the
    # evaluator policy: these snippets must be rejected before execution.
    policy_source=(ROOT/"main.py").read_text()
    def blocks_transitive(source):
        tree=ast.parse(source);blocked={"os","sys","subprocess","socket","shutil","ctypes","resource","importlib","builtins","pickle","marshal","ssl"}
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and any(a.name.split(".")[0] in blocked for a in node.names): return True
            if isinstance(node,ast.Attribute) and node.attr in blocked: return True
        return False
    assert blocks_transitive("from pathlib import os") and blocks_transitive("import pathlib; pathlib.os")
    assert "_TRANSITIVE_BLOCKLIST" in policy_source and "runtime_probe" in policy_source

print(json.dumps({"perfect":perfect["score"],"exact_diagonalization_dummy":exact_shortcut["score"],"fabricated_gradient":bad_gradient["score"],"fabricated_parameter":bad_parameter["score"],"fabricated_invariant":bad_invariant["score"],"malformed_operator":bad_operator["score"],"copied_hidden_energy_component":copied_hidden["components"]["hidden_energy"],"partial_hidden_cap":partial["score"],"overlong_ansatz_rejected":True,"malformed_input_rejected":True},sort_keys=True))
