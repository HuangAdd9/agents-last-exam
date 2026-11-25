"""Adversarial coverage for protocol shortcuts rejected by the seed critic."""
from __future__ import annotations
import json, math, tempfile
from pathlib import Path
from oracle import metrics, EXPECTED_STAGES, EXPECTED_RUNTIME, EXPECTED_WORKFLOW_SHA256

MARK='# workflow=QE_SCf+NSCF; noncollinear=.true.; lspinorb=.true.; 2D_coulomb_truncation=z; k_sampling=16x16x1; gw_bands=2500; cutoff_Ry=25; bse_grid=32x32x1\n'
AMARK='# workflow=BerkeleyGW epsilon sigma kernel absorption; electron-hole interaction=on; noncollinear=.true.; lspinorb=.true.; 2D_coulomb_truncation=z; source=absorption_eh.dat\n'

def manifest(p):
 import hashlib
 srcroot=p.parent/'sources'; srcroot.mkdir(exist_ok=True)
 for n in ('MoSe2_bands.dat.gnu','absorption_eh.dat'):
  (srcroot/n).write_bytes((p/n).read_bytes())
 decks={}; stages=[]
 for stage in EXPECTED_STAGES:
  d=p/f'{stage}.in'; evidence={'scf':"calculation='scf' noncolin=.true. lspinorb=.true. assume_isolated='2D'",'nscf':"calculation='nscf' noncolin=.true. lspinorb=.true.",'bands':"calculation='bands' noncolin=.true. lspinorb=.true.",'bands_post':'filband=', 'pw2bgw':'pw2bgw', 'epsilon':'cell_slab_truncation use_coulomb_truncation epsilon_cutoff', 'sigma':'cell_slab_truncation use_coulomb_truncation screened_coulomb_cutoff', 'kernel':'cell_slab_truncation use_coulomb_truncation bare_coulomb_cutoff', 'absorption':'cell_slab_truncation use_coulomb_truncation energy_range'}[stage]; d.write_text(f'! executable stage {stage}\n{evidence}\n')
  h=hashlib.sha256(d.read_bytes()).hexdigest(); path=str(d.resolve()); decks[stage]=path
  exe={'scf':'pw.x','nscf':'pw.x','bands':'pw.x','bands_post':'bands.x','pw2bgw':'pw2bgw.x','epsilon':'epsilon.cplx.x','sigma':'sigma.cplx.x','kernel':'kernel.cplx.x','absorption':'absorption.cplx.x'}[stage]
  stages.append({'stage':stage,'executable':exe,'command':['mpirun','-np','4',exe],'deck_path':path,'deck_sha256':h,'return_code':0})
 if True:
  for s,n in ((next(x for x in stages if x['stage']=='bands_post'),'MoSe2_bands.dat.gnu'),(next(x for x in stages if x['stage']=='absorption'),'absorption_eh.dat')):
   q=srcroot/n; s['output_path']=str(q.resolve()); s['output_sha256']=hashlib.sha256(q.read_bytes()).hexdigest()
 m={'schema_version':2,'runtime_image':EXPECTED_RUNTIME,'workflow_sha256':EXPECTED_WORKFLOW_SHA256,'executables':['mpirun','pw.x','bands.x','pw2bgw.x','epsilon.cplx.x','sigma.cplx.x','kernel.cplx.x','absorption.cplx.x'],'decks':decks,'stages':stages,'settings':{'noncollinear':True,'lspinorb':True,'coulomb_truncation':'2D slab z','k_sampling':'16x16x1','gw_summation_bands':2500,'screened_cutoff_Ry':25,'bse_fine_grid':'32x32x1'},'input_sha256':{'MoSe2.vasp':'131349e81094e74a2583b07a157f0e4235d534d94ef56e75ef4a859de1a26e8a','Mo.soc.upf':'b5a929dee3378934d30b33a8f510c8a715450d22b4f1a3469276569ac47ab29d','Se.soc.upf':'e18e8a18f12ddd61382b6051c5d6938b55159cdcb688ed3762502ed8549672e1'},'output_hashes':{},'table_provenance':{'bands':{'stage':'bands_post','source_path':str((srcroot/'MoSe2_bands.dat.gnu').resolve()),'source_sha256':hashlib.sha256((srcroot/'MoSe2_bands.dat.gnu').read_bytes()).hexdigest()},'absorption':{'stage':'absorption','source_path':str((srcroot/'absorption_eh.dat').resolve()),'source_sha256':hashlib.sha256((srcroot/'absorption_eh.dat').read_bytes()).hexdigest()}}}
 for n in ('MoSe2_bands.dat.gnu','MoSe2_bands.png','absorption_eh.dat','exciton_absorption_spectra_avg.png'): m['output_hashes'][n]=hashlib.sha256((p/n).read_bytes()).hexdigest()
 canonical={k:m[k] for k in ('schema_version','runtime_image','workflow_sha256','stages','settings','input_sha256','output_hashes','table_provenance')}
 m['attestation_sha256']=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 (p/'run_manifest.json').write_text(json.dumps(m)); return p/'run_manifest.json'
def good(p):
 p.mkdir();
 with (p/'MoSe2_bands.dat.gnu').open('w') as f:
  f.write(MARK)
  for b,off in enumerate((-0.10,-0.25,1.45,1.62)):
   for i in range(81):
    k=i/80; e=off+((-0.35 if b<2 else 0.30)*k*k); f.write(f'{k:.8f} {e:.8f}\n')
   f.write('\n')
 with (p/'absorption_eh.dat').open('w') as f:
  f.write(AMARK)
  for i in range(701):
   e=1.0+3.5*i/700; y=0.03+1.8*math.exp(-((e-1.52)/.045)**2)+1.1*math.exp(-((e-1.88)/.065)**2); f.write(f'{e:.8f} {y:.8f}\n')
 sig=b'\x89PNG\r\n\x1a\n'
 (p/'MoSe2_bands.png').write_bytes(sig+b'band'); (p/'exciton_absorption_spectra_avg.png').write_bytes(sig+b'spec')
 return manifest(p)

def main():
 with tempfile.TemporaryDirectory() as td:
  root=Path(td); out=root/'good'; mf=good(out); assert metrics(out,mf)['score']==1.0
  # Missing stages: evaluator must not accept a band/spectrum pair alone.
  missing=json.loads(mf.read_text()); missing['stages']=missing['stages'][:-1]; mf.write_text(json.dumps(missing)); assert metrics(out,mf)['score']==0.0
  mf=good(root/'protocol')
  # Omitted SOC/noncollinearity metadata is a protocol failure.
  q=(root/'protocol'/'MoSe2_bands.dat.gnu'); q.write_text(q.read_text().replace('noncollinear=.true.','noncollinear=.false.')); q=(root/'protocol'/'absorption_eh.dat'); q.write_text(q.read_text().replace('noncollinear=.true.','noncollinear=.false.')); assert metrics(root/'protocol',mf)['score']==0.0
  mf=good(root/'truncation'); q=(root/'truncation'/'absorption_eh.dat'); q.write_text(q.read_text().replace('2D_coulomb_truncation=z','')); q=(root/'truncation'/'MoSe2_bands.dat.gnu'); q.write_text(q.read_text().replace('2D_coulomb_truncation=z','')); assert metrics(root/'truncation',mf)['score']==0.0
  mf=good(root/'static'); q=(root/'static'/'absorption_eh.dat'); lines=q.read_text().splitlines(); q.write_text(lines[0]+'\n'+'\n'.join(f'{1+i/700:.8f} 1.0' for i in range(701))+'\n'); assert metrics(root/'static',mf)['score']==0.0
  # Valid markers, plausible bands, and two fabricated peaks must still fail
  # the normalized reference-shape comparison.
  fabricated_shape=root/'fabricated_shape'; mf=good(fabricated_shape)
  q=fabricated_shape/'absorption_eh.dat'; q.write_text(AMARK+'\n'.join(f'{1+i*3.5/700:.8f} {0.03 + 1.8*math.exp(-((1+i*3.5/700-2.35)/.04)**2) + 1.1*math.exp(-((1+i*3.5/700-3.25)/.05)**2):.8f}' for i in range(701))+'\n'); mf=manifest(fabricated_shape); assert metrics(fabricated_shape,mf)['score']==0.0
  # Direct diagonalization is a forbidden kernel shortcut.
  diagonal=root/'diagonal'; mf=good(diagonal); kd=diagonal/'kernel.in'; kd.write_text(kd.read_text()+'use_diagonalization true\n'); assert metrics(diagonal,mf)['score']==0.0
  mf=good(root/'fabricated')
  q=(root/'fabricated'/'MoSe2_bands.dat.gnu'); q.write_text("\n".join(line for line in q.read_text().splitlines() if not line.startswith('#'))+'\n')
  q=(root/'fabricated'/'absorption_eh.dat'); q.write_text("\n".join(line for line in q.read_text().splitlines() if not line.startswith('#'))+'\n')
  assert metrics(root/'fabricated',mf)['score']==0.0
  mf=good(root/'return_code'); m=json.loads(mf.read_text()); m['stages'][0]['return_code']=7; mf.write_text(json.dumps(m)); assert metrics(root/'return_code',mf)['score']==0.0
  mf=good(root/'forged'); m=json.loads(mf.read_text()); m['workflow_sha256']='forged'; mf.write_text(json.dumps(m)); assert metrics(root/'forged',mf)['score']==0.0
  # Provenance cannot self-attest a submitted output or a staged input.
  mf=good(root/'provenance_output'); m=json.loads(mf.read_text());
  m['table_provenance']['absorption']['source_path']=str((root/'provenance_output'/'absorption_eh.dat').resolve())
  m['stages'][-1]['output_path']=m['table_provenance']['absorption']['source_path']
  m['table_provenance']['absorption']['source_sha256']=__import__('hashlib').sha256((root/'provenance_output'/'absorption_eh.dat').read_bytes()).hexdigest()
  m['stages'][-1]['output_sha256']=m['table_provenance']['absorption']['source_sha256']
  m['attestation_sha256']=__import__('hashlib').sha256(json.dumps({k:m[k] for k in ('schema_version','runtime_image','workflow_sha256','stages','settings','input_sha256','output_hashes','table_provenance')},sort_keys=True,separators=(',',':')).encode()).hexdigest()
  mf.write_text(json.dumps(m)); assert metrics(root/'provenance_output',mf)['score']==0.0
  mf=good(root/'input_hash'); m=json.loads(mf.read_text()); m['input_sha256']['MoSe2.vasp']='0'*64; mf.write_text(json.dumps(m)); assert metrics(root/'input_hash',mf)['score']==0.0
  # A mutated staged driver cannot be hidden behind a freshly forged output.
  (root/'forged'/'workflow.py').write_text('print("fabricated")\n'); assert metrics(root/'forged',mf)['score']==0.0
 print('adversarial checks passed')
if __name__=='__main__': main()
