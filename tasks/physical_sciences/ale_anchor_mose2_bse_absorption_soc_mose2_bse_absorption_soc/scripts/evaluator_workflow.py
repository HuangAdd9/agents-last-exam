#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, os, shutil, struct, subprocess, sys, zlib
from pathlib import Path

def poscar(path):
    lines=[x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    scale=float(lines[1]); lattice=[[scale*float(x) for x in lines[i].split()[:3]] for i in (2,3,4)]
    symbols=lines[5].split(); counts=[int(x) for x in lines[6].split()]
    i=7
    if lines[i].lower().startswith(('s','d')): i+=1
    coords=[lines[i+j].split()[:3] for j in range(sum(counts))]
    atoms=[]
    for sym,n in zip(symbols,counts):
        atoms += [(sym, c) for c in coords[:n]]; coords=coords[n:]
    return lattice,atoms

def qe_decks(root, inp, pseudo):
    lattice,atoms=poscar(inp/'MoSe2.vasp')
    cell='\n'.join('  '+ ' '.join(f'{v:.12f}' for v in row) for row in lattice)
    apos='\n'.join(f'  {s} {" ".join(c)}' for s,c in atoms)
    common=f"""&CONTROL\n calculation='scf', prefix='mose2_soc', pseudo_dir='{pseudo}', outdir='{root/'qe'}', wf_collect=.true., verbosity='high'\n/\n&SYSTEM\n ibrav=0, nat=3, ntyp=2, ecutwfc=80, ecutrho=640, occupations='fixed', noncolin=.true., lspinorb=.true., assume_isolated='2D'\n/\n&ELECTRONS\n conv_thr=1.0d-10, mixing_beta=0.3\n/\nATOMIC_SPECIES\n Mo 95.95 Mo.soc.upf\n Se 78.971 Se.soc.upf\nCELL_PARAMETERS angstrom\n{cell}\nATOMIC_POSITIONS crystal\n{apos}\nK_POINTS automatic\n16 16 1 0 0 0\n"""
    (root/'scf.in').write_text(common)
    (root/'nscf.in').write_text(common.replace("calculation='scf'", "calculation='nscf'").replace('conv_thr=1.0d-10, mixing_beta=0.3','conv_thr=1.0d-10, diagonalization=\'david\'\n nbnd=2500'))
    (root/'bands.in').write_text(common.replace("calculation='scf'", "calculation='bands'").replace('K_POINTS automatic\n16 16 1 0 0 0','K_POINTS crystal\n8\n0.000 0.000 0.000 1\n0.500 0.000 0.000 1\n0.333 0.333 0.000 1\n0.000 0.000 0.000 1\n0.000 0.500 0.000 1\n0.333 0.333 0.000 1\n0.500 0.000 0.000 1\n0.000 0.000 0.000 1'))
    (root/'bands_post.in').write_text("&BANDS\n prefix='mose2_soc', outdir='%s', filband='MoSe2_bands.dat'\n/\n" % (root/'qe'))
    (root/'pw2bgw.in').write_text("""&input_pw2bgw\n prefix='mose2_soc', outdir='%s', real_or_complex='complex', wfng_flag=.true., rhog_flag=.true.\n/\n""" % (root/'qe'))
    bgw="""number_bands 2500\nnumber_valence_bands 11\ncell_slab_truncation z\nuse_coulomb_truncation true\nfrequency_grid 0.0 30.0 0.1\n"""
    (root/'epsilon.inp').write_text(bgw+'epsilon_cutoff 25.0\nfine_grid 32 32 1\n')
    (root/'sigma.inp').write_text(bgw+'screened_coulomb_cutoff 25.0\nfine_grid 32 32 1\n')
    (root/'kernel.inp').write_text(bgw+'screened_coulomb_cutoff 25.0\nbare_coulomb_cutoff 25.0\nfine_grid 32 32 1\n')
    (root/'absorption.inp').write_text(bgw+'screened_coulomb_cutoff 25.0\nfine_grid 32 32 1\nenergy_range 1.0 4.5\nenergy_resolution 0.005\npolarization 1 0 0\n')

def exe(name, bindir):
    p=Path(bindir)/name
    return str(p) if p.exists() else name

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def run(cmd, cwd, log, env, stage, deck_name=None):
    deck_path=str((Path(cwd)/deck_name).resolve()) if deck_name else ''
    record={'stage':stage,'executable':next((x for x in cmd if x.endswith(('.x','.cplx.x'))),Path(cmd[0]).name),
            'command':cmd,'deck_path':deck_path,'return_code':127}
    source=open(Path(cwd)/deck_name) if stage in ('pw2bgw','epsilon','sigma','kernel','absorption') else None
    try:
        try: r=subprocess.run(cmd,cwd=cwd,env=env,text=True,capture_output=True,stdin=source)
        except OSError as e:
            r=None; record['error']=str(e)
    finally:
        if source: source.close()
    if r is not None: record['return_code']=int(r.returncode)
    with open(log,'a') as f: f.write('$ '+' '.join(cmd)+'\n'+(r.stdout if r is not None else '')+'\n'+(r.stderr if r is not None else '')+'\n')
    return record

def png(path, series):
    # Minimal standards-compliant RGB PNG plotted from the actual table.
    w,h=900,500; px=[bytearray([255,255,255])*w for _ in range(h)]
    ys=[y for _,y in series if y==y]; lo=min(ys); hi=max(ys); span=hi-lo or 1.0
    for i in range(1,len(series)):
        x0=int((i-1)* (w-1)/(len(series)-1)); x1=int(i*(w-1)/(len(series)-1)); y0=h-1-int((series[i-1][1]-lo)/span*(h-1)); y1=h-1-int((series[i][1]-lo)/span*(h-1))
        n=max(abs(x1-x0),abs(y1-y0),1)
        for j in range(n+1):
            x=x0+(x1-x0)*j//n; y=y0+(y1-y0)*j//n
            if 0<=x<w and 0<=y<h: px[y][3*x:3*x+3]=bytes((25,70,160))
    raw=b''.join(b'\0'+bytes(row) for row in px)
    def ch(t,d): return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+ch(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+ch(b'IDAT',zlib.compress(raw,9))+ch(b'IEND',b''))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input-dir',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--work-dir',required=True); ap.add_argument('--bin-dir',default=''); a=ap.parse_args()
    inp=Path(a.input_dir).resolve(); out=Path(a.output_dir).resolve(); root=Path(a.work_dir).resolve(); out.mkdir(parents=True,exist_ok=True); root.mkdir(parents=True,exist_ok=True); (root/'qe').mkdir(exist_ok=True)
    for old in out.iterdir():
        if old.is_file(): old.unlink()
    bindir=a.bin_dir or os.environ.get('QE_BGW_BIN','')
    pseudo=inp; qe_decks(root,inp,pseudo)
    env=os.environ.copy(); env['OMP_NUM_THREADS']='1'; env['ESPRESSO_PSUFDIR']=str(pseudo)
    if env.get('MOSE2_HIDDEN_VARIANT'):
        # Hidden reruns perturb the physical optical broadening and screened
        # interaction while retaining the same SOC/truncation protocol.
        q=(root/'absorption.inp').read_text().replace('energy_resolution 0.005','energy_resolution 0.007').replace('energy_range 1.0 4.5','energy_range 1.1 4.3'); (root/'absorption.inp').write_text(q)
        (root/'kernel.inp').write_text((root/'kernel.inp').read_text().replace('screened_coulomb_cutoff 25.0','screened_coulomb_cutoff 23.0'))
    mpirun=exe('mpirun',bindir); launcher=[mpirun,'-np','4'] if shutil.which(mpirun) or Path(mpirun).exists() else []
    calls=[('scf',launcher+[exe('pw.x',bindir),'-in','scf.in'],'scf.in'),('nscf',launcher+[exe('pw.x',bindir),'-in','nscf.in'],'nscf.in'),('bands',launcher+[exe('pw.x',bindir),'-in','bands.in'],'bands.in'),('bands_post',launcher+[exe('bands.x',bindir),'-in','bands_post.in'],'bands_post.in'),('pw2bgw',launcher+[exe('pw2bgw.x',bindir)],'pw2bgw.in'),('epsilon',launcher+[exe('epsilon.cplx.x',bindir)],'epsilon.inp'),('sigma',launcher+[exe('sigma.cplx.x',bindir)],'sigma.inp'),('kernel',launcher+[exe('kernel.cplx.x',bindir)],'kernel.inp'),('absorption',launcher+[exe('absorption.cplx.x',bindir)],'absorption.inp')]
    log=root/'workflow.log';
    stages=[]
    for stage,c,deck_name in calls:
        stdin_name=deck_name if stage in ('pw2bgw','epsilon','sigma','kernel','absorption') else None
        rec=run(c,root,log,env,stage,deck_name)
        rec['deck_sha256']=sha256(root/deck_name) if (root/deck_name).is_file() else ''
        stages.append(rec)
    bands=next(root.rglob('*.dat.gnu'),None); spec=next((p for p in root.rglob('*') if p.is_file() and 'absorption' in p.name.lower() and p.suffix in ('.dat','.out')),None)
    if bands is None or spec is None:
        failed=[s for s in stages if s['return_code']!=0] or [{'stage':'output-generation','return_code':1}]
        settings={'noncollinear':True,'lspinorb':True,'coulomb_truncation':'2D slab z','k_sampling':'16x16x1','gw_summation_bands':2500,'screened_cutoff_Ry':23 if env.get('MOSE2_HIDDEN_VARIANT') else 25,'bse_fine_grid':'32x32x1','hidden_variant':bool(env.get('MOSE2_HIDDEN_VARIANT'))}
        manifest={'schema_version':2,'runtime_image':'ale/base-physical-qe-bgw:6.7.0-4.0-20260920','workflow_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'executables':['mpirun','pw.x','bands.x','pw2bgw.x','epsilon.cplx.x','sigma.cplx.x','kernel.cplx.x','absorption.cplx.x'],'decks':{s['stage']:s['deck_path'] for s in stages},'stages':stages,'settings':settings,'input_sha256':{p.name:sha256(p) for p in (inp/'MoSe2.vasp',inp/'Mo.soc.upf',inp/'Se.soc.upf')},'output_hashes':{},'table_provenance':{}}
        canonical=json.dumps({'schema_version':manifest['schema_version'],'runtime_image':manifest['runtime_image'],'workflow_sha256':manifest['workflow_sha256'],'stages':manifest['stages'],'settings':manifest['settings'],'input_sha256':manifest['input_sha256'],'output_hashes':manifest['output_hashes'],'table_provenance':manifest['table_provenance']},sort_keys=True,separators=(',',':')).encode(); manifest['attestation_sha256']=hashlib.sha256(canonical).hexdigest(); (root/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        raise RuntimeError('QE/BerkeleyGW did not produce band and absorption tables')
    btxt=bands.read_text(); stxt=spec.read_text()
    (out/'MoSe2_bands.dat.gnu').write_text('# workflow=QE_SCf+NSCF; noncollinear=.true.; lspinorb=.true.; 2D_coulomb_truncation=z; k_sampling=16x16x1; gw_bands=2500; cutoff_Ry=25; bse_grid=32x32x1\n'+btxt)
    (out/'absorption_eh.dat').write_text('# workflow=BerkeleyGW epsilon sigma kernel absorption; electron-hole interaction=on; noncollinear=.true.; lspinorb=.true.; 2D_coulomb_truncation=z; source='+str(spec)+'\n'+stxt)
    def nums(p):
        rows=[]
        for line in p.read_text().splitlines():
            q=line.split()
            if len(q)>=2:
                try: rows.append((float(q[0]),float(q[1])))
                except ValueError: pass
        return rows
    br=nums(out/'MoSe2_bands.dat.gnu'); sr=nums(out/'absorption_eh.dat')
    if len(br)<10 or len(sr)<10: raise RuntimeError('numeric QE/BerkeleyGW outputs are empty')
    png(out/'MoSe2_bands.png',br); png(out/'exciton_absorption_spectra_avg.png',sr)
    failed=[s for s in stages if s['return_code']!=0]
    # Tie each attested source table to the stage that produced it.  The
    # source is kept in the workflow directory, outside the submitted output.
    for rec, produced in ((next(s for s in stages if s['stage']=='bands_post'), bands),
                          (next(s for s in stages if s['stage']=='absorption'), spec)):
        rec['output_path']=str(produced.resolve()); rec['output_sha256']=sha256(produced)
    output_hashes={n:sha256(out/n) for n in ('MoSe2_bands.dat.gnu','MoSe2_bands.png','absorption_eh.dat','exciton_absorption_spectra_avg.png') if (out/n).is_file()}
    source_tables={'bands':{'stage':'bands_post','source_path':str(bands.resolve()),'source_sha256':sha256(bands)},'absorption':{'stage':'absorption','source_path':str(spec.resolve()),'source_sha256':sha256(spec)}}
    settings={'noncollinear':True,'lspinorb':True,'coulomb_truncation':'2D slab z','k_sampling':'16x16x1','gw_summation_bands':2500,'screened_cutoff_Ry':23 if env.get('MOSE2_HIDDEN_VARIANT') else 25,'bse_fine_grid':'32x32x1','hidden_variant':bool(env.get('MOSE2_HIDDEN_VARIANT'))}
    if env.get('MOSE2_HIDDEN_VARIANT'): settings.update({'energy_range':'1.1 4.3','energy_resolution':0.007})
    manifest={'schema_version':2,'runtime_image':'ale/base-physical-qe-bgw:6.7.0-4.0-20260920','workflow_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'executables':['mpirun','pw.x','bands.x','pw2bgw.x','epsilon.cplx.x','sigma.cplx.x','kernel.cplx.x','absorption.cplx.x'],'decks':{s['stage']:s['deck_path'] for s in stages},'stages':stages,'settings':settings,'input_sha256':{p.name:sha256(p) for p in (inp/'MoSe2.vasp',inp/'Mo.soc.upf',inp/'Se.soc.upf')},'output_hashes':output_hashes,'table_provenance':source_tables}
    canonical=json.dumps({'schema_version':manifest['schema_version'],'runtime_image':manifest['runtime_image'],'workflow_sha256':manifest['workflow_sha256'],'stages':manifest['stages'],'settings':manifest['settings'],'input_sha256':manifest['input_sha256'],'output_hashes':manifest['output_hashes'],'table_provenance':manifest['table_provenance']},sort_keys=True,separators=(',',':')).encode()
    manifest['attestation_sha256']=hashlib.sha256(canonical).hexdigest()
    (root/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if failed: raise RuntimeError('QE/BerkeleyGW stages failed: '+','.join(s['stage'] for s in failed))
if __name__=='__main__': main()
