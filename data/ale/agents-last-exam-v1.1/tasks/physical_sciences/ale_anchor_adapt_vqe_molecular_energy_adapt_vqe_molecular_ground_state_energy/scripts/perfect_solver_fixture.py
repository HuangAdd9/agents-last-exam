"""Genuine reusable ADAPT-VQE candidate used by evaluator regression tests."""
from __future__ import annotations
import argparse, itertools, json, math, re
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

CASES={"tier1":("h2_hamiltonian.json","H2","VQE"),"tier2":("lih_hamiltonian.json","LiH","ADAPT-VQE"),"tier3":("beh2_hamiltonian.json","BeH2","ADAPT-VQE")}
S=re.compile(r"S\((\d+)->(\d+)\)");D=re.compile(r"D\((\d+),(\d+)->(\d+),(\d+)\)")
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(float(x))
def load_inputs(root,config_path):
    data={}
    for _,(fn,mol,_) in CASES.items():
        p=json.loads((root/fn).read_text());n=p.get("n_qubits");ne=p.get("n_electrons");terms=p.get("pauli_terms")
        expected={"molecule","bond_length","bond_unit","basis","n_qubits","n_electrons","active_space","n_pauli_terms","pauli_terms"}
        if set(p)!=expected or p.get("molecule")!=mol or not isinstance(n,int) or isinstance(n,bool) or n<=0 or not isinstance(ne,int) or isinstance(ne,bool) or ne<=0 or ne>=n or not isinstance(terms,dict) or not isinstance(p.get("n_pauli_terms"),int) or p["n_pauli_terms"]<=0 or len(terms)!=p["n_pauli_terms"]:raise ValueError("bad Hamiltonian metadata")
        if not finite(p.get("bond_length")) or p.get("bond_length")<=0 or p.get("bond_unit") not in {"angstrom","bohr"} or not isinstance(p.get("basis"),str) or not p["basis"]:raise ValueError("bad molecular schema")
        if any(not isinstance(w,str) or len(w)!=n or set(w)-set("IXYZ") or not finite(c) for w,c in terms.items()):raise ValueError("bad Pauli term")
        data[mol]=p
    c=json.loads(config_path.read_text());keys={"schema_version","field_model","site_z_coefficients_ha","identity_shift_ha","electron_sector_required","hartree_to_ev"}
    if set(c)!=keys or c["schema_version"]!=2 or c["field_model"]!="site_resolved_z" or c["electron_sector_required"] is not True or not isinstance(c["site_z_coefficients_ha"],dict) or set(c["site_z_coefficients_ha"])!=set(data) or not finite(c["identity_shift_ha"]) or not finite(c["hartree_to_ev"]) or c["hartree_to_ev"]<=0:raise ValueError("bad field configuration")
    for mol,p in data.items():
        z=c["site_z_coefficients_ha"][mol]
        if not isinstance(z,list) or len(z)!=p["n_qubits"] or not all(finite(x) for x in z):raise ValueError("bad field vector")
    return data,c
def ham(p,c):
    n,ne=p["n_qubits"],p["n_electrons"];sec=np.array([x for x in range(1<<n) if x.bit_count()==ne],dtype=np.int64);idx={int(x):i for i,x in enumerate(sec)};terms=dict(p["pauli_terms"]);terms["I"*n]=terms.get("I"*n,0)+c["identity_shift_ha"]
    for q,z in enumerate(c["site_z_coefficients_ha"][p["molecule"]]):
        w="I"*q+"Z"+"I"*(n-q-1);terms[w]=terms.get(w,0)+z
    h=np.zeros((len(sec),len(sec)),complex)
    for w,coef in terms.items():
        for col,sv in enumerate(sec):
            state,phase=int(sv),1+0j
            for q,sym in enumerate(w):
                mask=1<<(n-1-q);bit=bool(state&mask)
                if sym=="X":state^=mask
                elif sym=="Y":state^=mask;phase*=-1j if bit else 1j
                elif sym=="Z" and bit:phase*=-1
            if state in idx:h[idx[state],col]+=coef*phase
    if not np.allclose(h,h.conj().T,atol=1e-10,rtol=0):raise ValueError("non-Hermitian")
    return (h+h.conj().T)/2,sec
def excitation(op):
    m=S.fullmatch(op)
    if m:return [int(m[1])],[int(m[2])]
    m=D.fullmatch(op)
    return [int(m[1]),int(m[2])],[int(m[3]),int(m[4])]
def apply(v,sec,op,t):
    src,dst=excitation(op);n=int(max(sec)).bit_length();idx={int(x):i for i,x in enumerate(sec)};out=v.copy();ct,st=math.cos(t),math.sin(t);seen=set()
    def act(state,ann,cre):
        sign=1
        for q in sorted(ann,reverse=True):
            mask=1<<(n-1-q)
            if not state&mask:return None
            sign*=-1 if (state&(mask-1)).bit_count()%2 else 1;state^=mask
        for q in sorted(cre):
            mask=1<<(n-1-q)
            if state&mask:return None
            sign*=-1 if (state&(mask-1)).bit_count()%2 else 1;state|=mask
        return state,sign
    for col,sv in enumerate(sec):
        a=act(int(sv),src,dst)
        if a is None:continue
        row,sign=idx[a[0]],a[1];key=tuple(sorted((row,col)))
        if key in seen:continue
        seen.add(key);x,y=v[col],v[row];out[col]=ct*x-st*sign*y;out[row]=ct*y+st*sign*x
    return out
def pool(p):
    occ=range(p["n_electrons"]);virt=range(p["n_electrons"],p["n_qubits"])
    return [f"S({i}->{a})" for i in occ for a in virt]+[f"D({i},{j}->{a},{b})" for i,j in itertools.combinations(occ,2) for a,b in itertools.combinations(virt,2)]
def solve_case(p,c,method):
    h,sec=ham(p,c);n,ne=p["n_qubits"],p["n_electrons"];hf=sum(1<<(n-1-q) for q in range(ne));base=np.zeros(len(sec),complex);base[np.where(sec==hf)[0][0]]=1;ops=[];parameters=np.array([],float);history=[];available=pool(p)
    def state(x):
        v=base
        for op,t in zip(ops,x):v=apply(v,sec,op,float(t))
        return v
    def energy(x):
        v=state(x);return float(np.vdot(v,h@v).real)
    def gradient(v,op):
        e=1e-5;vp=apply(v,sec,op,e);vm=apply(v,sec,op,-e);return float((np.vdot(vp,h@vp).real-np.vdot(vm,h@vm).real)/(2*e))
    for _ in range(32):
        v=state(parameters);gradients={op:gradient(v,op) for op in available if op not in ops};max_gradient=max((abs(x) for x in gradients.values()),default=0.0)
        if max_gradient<=1e-3:break
        chosen=max(gradients,key=lambda op:abs(gradients[op]));g=gradients[chosen]
        ops.append(chosen);parameters=np.append(parameters,0.0);result=minimize(energy,parameters,method="BFGS",options={"gtol":1e-9,"maxiter":1000});parameters=np.asarray(result.x,float)
        history.append({"operator":chosen,"gradient":g,"energy_ha":energy(parameters),"parameters":parameters.tolist()})
    v=state(parameters);e=energy(parameters);number=float(np.vdot(v,ne*v).real);variance=float(np.vdot(v,(ne-number)**2*v).real)
    return {"molecule":p["molecule"],"method":method,"energy_ha":e,"energy_ev":e*c["hartree_to_ev"],"number_expectation":number,"number_variance":variance,"n_parameters":len(ops),"adapt_iterations":len(ops),"operator_sequence":ops,"ansatz_parameters":parameters.tolist(),"adapt_history":history}
def solve(root,config_path):
    data,c=load_inputs(root,config_path);return {tier:solve_case(data[mol],c,method) for tier,(_,mol,method) in CASES.items()}
def main():
    a=argparse.ArgumentParser();a.add_argument("--input-dir",type=Path,required=True);a.add_argument("--field-config",type=Path,required=True);a.add_argument("--output",type=Path,required=True);x=a.parse_args();x.output.write_text(json.dumps(solve(x.input_dir,x.field_config),sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
