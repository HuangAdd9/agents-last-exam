from __future__ import annotations
import json
from pathlib import Path
import numpy as np

def basis_states(nx=3, ny=4):
    n=nx*ny; k=n//2
    states=np.array([s for s in range(1<<n) if s.bit_count()==k],dtype=np.uint32)
    return states,{int(s):i for i,s in enumerate(states)}

def bonds(nx=3, ny=4):
    def site(x,y): return (y%ny)*nx+(x%nx)
    out=[]
    for y in range(ny):
      for x in range(nx):
        i=site(x,y)
        out += [(i,site(x+1,y),1.0),(i,site(x,y+1),1.0),
                (i,site(x+1,y+1),None),(i,site(x-1,y+1),None)]
    return out

def hamiltonian(j2: float, nx=3, ny=4):
    basis, idx=basis_states(nx,ny); d=len(basis); H=np.zeros((d,d),float)
    for ai,s in enumerate(basis):
      si=int(s)
      for i,j,c in bonds(nx,ny):
        c=j2 if c is None else c
        upi=(si>>i)&1; upj=(si>>j)&1
        H[ai,ai]+=c*(0.25 if upi==upj else -0.25)
        if upi!=upj:
          t=si^(1<<i)^(1<<j); H[idx[t],ai]+=0.5*c
    return H,basis,idx

def q_points(nx=3, ny=4):
    return np.array([[2*np.pi*ix/nx,2*np.pi*iy/ny] for iy in range(ny) for ix in range(nx)],float)

def solve_case(j2: float, nx=3, ny=4, n_lanczos=80, n_omega=240):
    H,basis,idx=hamiltonian(j2,nx,ny); vals,vecs=np.linalg.eigh(H)
    E0,E1=float(vals[0]),float(vals[1]); psi=vecs[:,0]
    if psi[np.argmax(np.abs(psi))]<0: psi=-psi
    n=nx*ny; qs=q_points(nx,ny); coords=np.array([(x,y) for y in range(ny) for x in range(nx)],float)
    corr=np.zeros((n,n),float)
    for ai,s in enumerate(basis):
      p=abs(psi[ai])**2; spins=np.array([0.5 if (int(s)>>i)&1 else -0.5 for i in range(n)])
      corr += p*np.outer(spins,spins)
      si=int(s)
      for i in range(n):
       for j in range(n):
        if i!=j and ((si>>i)&1)!=((si>>j)&1):
          t=si^(1<<i)^(1<<j); corr[i,j] += 0.5*np.real(np.conj(psi[ai])*psi[idx[t]])
    np.fill_diagonal(corr,0.75)
    sf=np.array([np.real(np.sum(corr*np.exp(1j*((coords[:,None,0]-coords[None,:,0])*q[0]+(coords[:,None,1]-coords[None,:,1])*q[1]))))/n for q in qs])
    eta=0.1; omega_max=float(vals[-1]-E0+1.0); omega=np.linspace(0,omega_max,n_omega); sqw=np.zeros((len(qs),n_omega)); coeff=[]
    for qi,q in enumerate(qs):
      phases=np.exp(-1j*(coords[:,0]*q[0]+coords[:,1]*q[1]))/np.sqrt(n)
      phi=np.zeros(len(basis),complex)
      for ai,s in enumerate(basis): phi[ai]=psi[ai]*sum(phases[i]*(0.5 if (int(s)>>i)&1 else -0.5) for i in range(n))
      weight=float(np.vdot(phi,phi).real)
      if weight>1e-14:
        overlaps=np.abs(vecs.conj().T@phi)**2
        gaps=vals-E0
        sqw[qi]=3*np.sum(overlaps[:,None]*(eta/np.pi)/((omega[None,:]-gaps[:,None])**2+eta**2),axis=0)
        v=phi/np.sqrt(weight); vs=[]; aa=[]; bb=[]; prev=np.zeros_like(v); bprev=0.0
        for step in range(n_lanczos):
          w=H@v
          if step: w-=bprev*prev
          a=float(np.vdot(v,w).real); w-=a*v
          for old in vs: w-=np.vdot(old,w)*old
          b=float(np.linalg.norm(w)); vs.append(v.copy()); aa.append(a)
          if step<n_lanczos-1:
            bb.append(b); prev,v=v,w/b if b>1e-14 else np.zeros_like(v); bprev=b
        coeff.append((np.array(aa),np.array(bb)))
      else: coeff.append((np.zeros(n_lanczos),np.zeros(n_lanczos-1)))
    return {'E0':E0,'E1':E1,'spin_gap':E1-E0,'ground_state_vector':psi,'correlation_matrix':corr,'q_points':qs,'structure_factor':sf,'omega_grid':omega,'S_q_omega':sqw,'eta':eta,'omega_max':omega_max,'coeff':coeff,'j2':j2,'nx':nx,'ny':ny}

def write_outputs(result, out: Path):
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'ground_state.npz',E0=result['E0'],E1=result['E1'],spin_gap=result['spin_gap'],ground_state_vector=result['ground_state_vector'])
    np.savez_compressed(out/'correlations.npz',correlation_matrix=result['correlation_matrix'],q_points=result['q_points'],structure_factor=result['structure_factor'])
    kw={'q_points':result['q_points'],'omega_grid':result['omega_grid'],'S_q_omega':result['S_q_omega'],'eta':result['eta'],'omega_max':result['omega_max']}
    for i,(a,b) in enumerate(result['coeff']): kw[f'a_{i}']=a; kw[f'b_{i}']=b
    np.savez_compressed(out/'dynamical_sf.npz',**kw)
    n=result['nx']*result['ny']; q=result['q_points']; sf=result['structure_factor']; om=result['omega_grid']; sqw=result['S_q_omega']
    pi_idx=int(np.argmin(np.linalg.norm(q-np.array([np.pi,np.pi]),axis=1))); z_idx=int(np.argmin(np.linalg.norm(q,axis=1)))
    integ=np.trapezoid(sqw,om,axis=1); err=float(np.max(np.abs(integ-sf)/np.maximum(np.abs(sf),1e-12)))
    results={'tier1':{'E0':result['E0'],'E1':result['E1'],'spin_gap':result['spin_gap'],'E0_per_site':result['E0']/n,'hilbert_dim':len(result['ground_state_vector']),'num_sites':n},'tier2':{'S_q_pi_pi':float(sf[pi_idx]),'S_q_0_0':float(sf[z_idx]),'correlation_sum_rule':float(sf.sum()),'C_nearest_neighbor':float(result['correlation_matrix'][0,1]),'C_next_nearest_neighbor':float(result['correlation_matrix'][0,1+result['nx']])},'tier3':{'omega_max':result['omega_max'],'eta':result['eta'],'num_lanczos_steps':len(result['coeff'][0][0]),'num_omega_points':len(om),'num_q_points':len(q),'sum_rule_max_error':err}}
    (out/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
