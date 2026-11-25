"""Scientific grader: reconstruct and independently evaluate a VQE ansatz."""
from __future__ import annotations
import itertools, math, re
from typing import Any
import numpy as np

HA_TO_EV = 27.211386245988
CASES = {"tier1": ("H2", "VQE"), "tier2": ("LiH", "ADAPT-VQE"), "tier3": ("BeH2", "ADAPT-VQE")}
TIER1_MAX_PARAMETERS = 4  # Tier 1 is the documented one/few-parameter VQE construction.
FIELDS = {"molecule","method","energy_ha","energy_ev","number_expectation","number_variance","n_parameters","adapt_iterations","operator_sequence","ansatz_parameters","adapt_history"}
SINGLE = re.compile(r"^S\((\d+)->(\d+)\)$"); DOUBLE = re.compile(r"^D\((\d+),(\d+)->(\d+),(\d+)\)$")
def _finite(x: Any) -> bool: return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(float(x))
def validate_payload(payload: Any) -> list[str]:
    out=[]
    if not isinstance(payload,dict) or set(payload)!=set(CASES): return ["top-level keys must be exactly tier1, tier2, tier3"]
    for tier,(mol,method) in CASES.items():
        r=payload[tier]
        if not isinstance(r,dict) or set(r)!=FIELDS: out.append(f"{tier}: exact field set required"); continue
        if (r["molecule"],r["method"])!=(mol,method): out.append(f"{tier}: molecule/method mismatch")
        if not all(_finite(r[k]) for k in ("energy_ha","energy_ev","number_expectation","number_variance")): out.append(f"{tier}: numeric fields must be finite")
        if not all(isinstance(r[k],int) and not isinstance(r[k],bool) and r[k]>=0 for k in ("n_parameters","adapt_iterations")): out.append(f"{tier}: counts must be non-negative integers")
        if r["n_parameters"] > 32 or r["adapt_iterations"] > 32: out.append(f"{tier}: at most 32 ADAPT iterations are permitted")
        if tier == "tier1" and (r["n_parameters"] > TIER1_MAX_PARAMETERS or r["adapt_iterations"] > TIER1_MAX_PARAMETERS):
            out.append(f"{tier}: VQE construction is limited to at most {TIER1_MAX_PARAMETERS} parameters")
        seq,pars,hist=r["operator_sequence"],r["ansatz_parameters"],r["adapt_history"]
        if not all(isinstance(x,list) for x in (seq,pars,hist)) or len(seq)!=r["n_parameters"] or len(pars)!=r["n_parameters"] or len(hist)!=r["adapt_iterations"] or r["adapt_iterations"]!=r["n_parameters"]: out.append(f"{tier}: ansatz lengths/counts inconsistent"); continue
        for op in seq:
            m=SINGLE.fullmatch(op) or DOUBLE.fullmatch(op)
            if m is None: out.append(f"{tier}: invalid excitation label {op!r}"); continue
            nums=[int(x) for x in m.groups()]; split=len(nums)//2
            if len(set(nums))!=len(nums) or nums[:split]!=sorted(nums[:split]) or nums[split:]!=sorted(nums[split:]): out.append(f"{tier}: non-canonical excitation label {op!r}")
        if not all(_finite(x) and abs(float(x))<100 for x in pars): out.append(f"{tier}: parameters must be finite and bounded")
        for k,item in enumerate(hist):
            if not isinstance(item,dict) or set(item)!={"operator","gradient","energy_ha","parameters"} or not isinstance(item.get("operator"),str) or not _finite(item.get("gradient")) or not _finite(item.get("energy_ha")) or not isinstance(item.get("parameters"),list) or len(item["parameters"])!=k+1 or not all(_finite(x) and abs(float(x))<100 for x in item["parameters"]): out.append(f"{tier}: malformed ADAPT history")
    return out
def _ham(payload,config):
    n,ne=int(payload["n_qubits"]),int(payload["n_electrons"]); sec=np.array([s for s in range(1<<n) if s.bit_count()==ne],dtype=np.int64); lookup={int(s):i for i,s in enumerate(sec)}; terms=dict(payload["pauli_terms"]); terms["I"*n]=terms.get("I"*n,0.)+float(config["identity_shift_ha"])
    for q,z in enumerate(config["site_z_coefficients_ha"][payload["molecule"]]):
        w="I"*q+"Z"+"I"*(n-q-1); terms[w]=terms.get(w,0.)+float(z)
    h=np.zeros((len(sec),len(sec)),complex)
    for w,c in terms.items():
        for col,sv in enumerate(sec):
            state,phase=int(sv),1+0j
            for q,sym in enumerate(w):
                mask=1<<(n-1-q); bit=bool(state&mask)
                if sym=="X": state^=mask
                elif sym=="Y": state^=mask; phase*=-1j if bit else 1j
                elif sym=="Z" and bit: phase*=-1
            row=lookup.get(state)
            if row is not None: h[row,col]+=float(c)*phase
    if not np.allclose(h,h.conj().T,atol=1e-10,rtol=0): raise ValueError("non-Hermitian Hamiltonian")
    return (h+h.conj().T)/2,sec
def _exc(op):
    m=SINGLE.fullmatch(op)
    if m:return [int(m.group(1))],[int(m.group(2))]
    m=DOUBLE.fullmatch(op)
    if m:return [int(m.group(1)),int(m.group(2))],[int(m.group(3)),int(m.group(4))]
    raise ValueError("bad operator")
def _apply(v,sec,op,theta):
    src,dst=_exc(op); n=max(1,int(max(sec)).bit_length()); idx={int(s):i for i,s in enumerate(sec)}; out=v.copy(); c,s=math.cos(theta),math.sin(theta); seen=set()
    def act(state,ann,cre):
        sign=1
        for q in sorted(ann,reverse=True):
            mask=1<<(n-1-q)
            if not state&mask:return None
            sign*=-1 if ((state&(mask-1)).bit_count()&1) else 1; state^=mask
        for q in sorted(cre):
            mask=1<<(n-1-q)
            if state&mask:return None
            sign*=-1 if ((state&(mask-1)).bit_count()&1) else 1; state|=mask
        return state,sign
    for col,sv in enumerate(sec):
        a=act(int(sv),src,dst)
        if a is None:continue
        row,sg=idx[a[0]],a[1]; key=tuple(sorted((col,row)))
        if key in seen:continue
        seen.add(key); x,y=v[col],v[row]; out[col]=c*x-s*sg*y; out[row]=c*y+s*sg*x
    return out
def _grad(h,sec,state,op):
    e=1e-5; return float((np.vdot(_apply(state,sec,op,e),h@_apply(state,sec,op,e)).real-np.vdot(_apply(state,sec,op,-e),h@_apply(state,sec,op,-e)).real)/(2*e))
def _pool(payload):
    n,ne=int(payload["n_qubits"]),int(payload["n_electrons"]); occ=range(ne); virt=range(ne,n)
    return [f"S({i}->{a})" for i in occ for a in virt]+[f"D({i},{j}->{a},{b})" for i,j in itertools.combinations(occ,2) for a,b in itertools.combinations(virt,2)]
def _reconstruct(row,payload,config):
    h,sec=_ham(payload,config); n,ne=int(payload["n_qubits"]),int(payload["n_electrons"]); hf=sum(1<<(n-1-q) for q in range(ne)); base=np.zeros(len(sec),complex); base[np.where(sec==hf)[0][0]]=1.; used=[]
    def state_for(ops,parameters):
        state=base
        for label,theta in zip(ops,parameters): state=_apply(state,sec,label,float(theta))
        return state
    previous=base
    if len(row["operator_sequence"]) > 32 or len(used) > 32:
        raise ValueError("ADAPT iteration limit exceeded")
    initial_remaining = _pool(payload)
    initial_max = max((abs(_grad(h,sec,base,x)) for x in initial_remaining), default=0.)
    if not row["operator_sequence"]:
        if row["adapt_iterations"] != 0 or initial_max > 1e-3:
            raise ValueError("zero-length ansatz is allowed only below the initial gradient threshold")
    for k,(op,item) in enumerate(zip(row["operator_sequence"],row["adapt_history"])):
        if item["operator"]!=op or op in used:raise ValueError("history/operator mismatch or duplicate")
        remaining=[x for x in _pool(payload) if x not in used]; gradients={x:_grad(h,sec,previous,x) for x in remaining}; g=gradients[op]; mx=max(map(abs,gradients.values()),default=0.)
        if abs(float(item["gradient"])-g)>2e-4 or abs(g)+2e-4<mx:raise ValueError("invalid ADAPT gradient/selection")
        used.append(op)
        if len(used) > 32:
            raise ValueError("ADAPT iteration limit exceeded")
        previous=state_for(used,item["parameters"]); history_energy=float(np.vdot(previous,h@previous).real)
        if abs(float(item["energy_ha"])-history_energy)>2e-7:raise ValueError("fabricated history energy")
        if k and history_energy > float(row["adapt_history"][k-1]["energy_ha"]) + 2e-7:
            raise ValueError("joint reoptimisation energy increased")
        for j in range(len(used)):
            plus=list(map(float,item["parameters"]));minus=plus.copy();plus[j]+=1e-5;minus[j]-=1e-5
            ep=float(np.vdot(state_for(used,plus),h@state_for(used,plus)).real);em=float(np.vdot(state_for(used,minus),h@state_for(used,minus)).real)
            if abs((ep-em)/2e-5)>2e-3:raise ValueError("history parameters were not jointly reoptimised")
            curvature=(ep-2.0*history_energy+em)/(1e-5**2)
            if curvature < -2e-3:raise ValueError("history point is a maximal or saddle stationary point")
        if 1 < len(used) <= 32:
            # Coordinate derivatives alone can miss a saddle with a negative
            # mixed direction.  Check the finite-difference Hessian spectrum
            # for every allowed ansatz size, not just the small histories.
            step=1e-4; p0=np.asarray(item["parameters"],dtype=float); hessian=np.zeros((len(used),len(used)))
            for a in range(len(used)):
                for b in range(a,len(used)):
                    if a == b:
                        pp=p0.copy(); pm=p0.copy(); pp[a]+=step; pm[a]-=step
                        hessian[a,a]=(float(np.vdot(state_for(used,pp),h@state_for(used,pp)).real)-2.0*history_energy+float(np.vdot(state_for(used,pm),h@state_for(used,pm)).real))/(step**2)
                    else:
                        vals=[]
                        for sa,sb in ((1,1),(1,-1),(-1,1),(-1,-1)):
                            px=p0.copy(); px[a]+=sa*step; px[b]+=sb*step
                            vals.append(float(np.vdot(state_for(used,px),h@state_for(used,px)).real))
                        hessian[a,b]=hessian[b,a]=(vals[0]-vals[1]-vals[2]+vals[3])/(4.0*step**2)
            if float(np.min(np.linalg.eigvalsh(hessian))) < -1e-2:
                raise ValueError("history point is a saddle or maximal stationary point")
    state=state_for(row["operator_sequence"],row["ansatz_parameters"])
    if used and any(abs(float(a)-float(b))>2e-7 for a,b in zip(row["ansatz_parameters"],row["adapt_history"][-1]["parameters"])):raise ValueError("final parameters differ from last optimisation")
    remaining=[x for x in _pool(payload) if x not in used]; max_final=max((abs(_grad(h,sec,state,x)) for x in remaining),default=0.)
    if len(used)<32 and max_final>1e-3:raise ValueError("ADAPT stopping rule not met")
    en=float(np.vdot(state,h@state).real); counts=np.full(len(sec),ne,dtype=float); ex=float(np.vdot(state,counts*state).real); var=float(np.vdot(state,(counts-ex)**2*state).real)
    return {"energy_ha":en,"number_expectation":ex,"number_variance":var}
def _mean(x): return sum(x)/len(x)
def score_payloads(visible,hidden,reference,context=None):
    hidden_runs=hidden if isinstance(hidden,list) else [hidden]
    reasons=[f"visible: {x}" for x in validate_payload(visible)]
    for i,payload in enumerate(hidden_runs): reasons += [f"hidden_{i+1}: {x}" for x in validate_payload(payload)]
    if reasons:return {"score":0.,"components":{},"reasons":reasons}
    raw={k:[] for k in ("visible_energy","unit_consistency","metadata","number_sector","ansatz","hidden_energy")}
    for tier,(mol,method) in CASES.items():
        r=visible[tier]; vr=reference["visible"][tier]
        ok=True
        if context is not None:
            try:
                checks=[(r,"visible")]+[(run[tier],f"hidden_{i+1}") for i,run in enumerate(hidden_runs)]
                for x,key in checks:
                    got=_reconstruct(x,context["payloads"][mol],context["configs"][key])
                    if abs(x["energy_ha"]-got["energy_ha"])>2e-7 or abs(x["number_expectation"]-got["number_expectation"])>2e-6 or abs(x["number_variance"]-got["number_variance"])>2e-6: raise ValueError("claimed values differ from reconstructed state")
            except Exception:ok=False
        raw["ansatz"].append(1. if ok else 0.)
        tol={"tier1":1e-4,"tier2":1.6e-3,"tier3":1.6e-3}[tier]
        raw["visible_energy"].append(1. if abs(r["energy_ha"]-vr["energy_ha"])<=tol else 0.)
        if context is None:
            all_rows=[(r, HA_TO_EV)] + [(run[tier], HA_TO_EV) for run in hidden_runs]
        else:
            all_rows=[(r, float(context["configs"]["visible"]["hartree_to_ev"]))] + [(run[tier], float(context["configs"][f"hidden_{i+1}"]["hartree_to_ev"])) for i,run in enumerate(hidden_runs)]
        raw["unit_consistency"].append(min(max(0.,1-abs(x["energy_ev"]-factor*x["energy_ha"])/.0002) for x,factor in all_rows))
        raw["metadata"].append(1. if all((x[0]["molecule"],x[0]["method"])==(mol,method) for x in all_rows) else 0.)
        if context is None:
            raw["number_sector"].append(0.)
        else:
            raw["number_sector"].append(1. if all(abs(x[0]["number_expectation"]-int(context["payloads"][mol]["n_electrons"]))<=.002 and 0<=x[0]["number_variance"]<=.002 for x in all_rows) else 0.)
        hidden_scores=[]
        for i,run in enumerate(hidden_runs): hidden_scores.append(1. if abs(run[tier]["energy_ha"]-reference[f"hidden_{i+1}"][tier]["energy_ha"])<=tol else 0.)
        raw["hidden_energy"].append(_mean(hidden_scores))
    weights={"visible_energy":.25,"unit_consistency":.10,"metadata":.05,"number_sector":.10,"ansatz":.20,"hidden_energy":.30}; components={k:weights[k]*_mean(v) for k,v in raw.items()}; score=sum(components.values())
    if components["ansatz"]<weights["ansatz"] or components["hidden_energy"]<weights["hidden_energy"]:score=min(score,.60)
    return {"score":min(1.,max(0.,score)),"components":components,"reasons":[]}
