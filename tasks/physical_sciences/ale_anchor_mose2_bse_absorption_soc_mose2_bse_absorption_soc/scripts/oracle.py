"""Independent output oracle for the QE/BerkeleyGW MoSe2 task.

The oracle never regenerates a spectrum from a guessed Hamiltonian.  It parses
band and absorption tables, verifies workflow provenance, and recomputes the
observables (edge coincidence, SOC separation, peak positions, and normalized
spectrum) from those tables.
"""
from __future__ import annotations
import csv, hashlib, json, math, struct, zlib
from pathlib import Path

OUTPUTS=('MoSe2_bands.dat.gnu','MoSe2_bands.png','absorption_eh.dat','exciton_absorption_spectra_avg.png')
REQUIRED_MARKERS=('noncollinear=.true.','lspinorb=.true.','2d_coulomb_truncation','16x16x1','2500','25','32x32x1')
EXPECTED_RUNTIME='ale/base-physical-qe-bgw:6.7.0-4.0-20260920'
PACKAGE_ROOT=Path(__file__).resolve().parents[1]
def _file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
EXPECTED_INPUT_SHA256={name:_file_sha256(PACKAGE_ROOT/'input'/name) for name in ('MoSe2.vasp','Mo.soc.upf','Se.soc.upf')}
EXPECTED_STAGES=('scf','nscf','bands','bands_post','pw2bgw','epsilon','sigma','kernel','absorption')
EXPECTED_EXECUTABLES={'scf':'pw.x','nscf':'pw.x','bands':'pw.x','bands_post':'bands.x','pw2bgw':'pw2bgw.x','epsilon':'epsilon.cplx.x','sigma':'sigma.cplx.x','kernel':'kernel.cplx.x','absorption':'absorption.cplx.x'}
DECK_EVIDENCE={'scf':("calculation='scf'",'noncolin=.true.','lspinorb=.true.',"assume_isolated='2d'"),'nscf':("calculation='nscf'",'noncolin=.true.','lspinorb=.true.'),'bands':("calculation='bands'",'noncolin=.true.','lspinorb=.true.'),'bands_post':('filband=',),'pw2bgw':('pw2bgw',),'epsilon':('cell_slab_truncation','use_coulomb_truncation','epsilon_cutoff'),'sigma':('cell_slab_truncation','use_coulomb_truncation','screened_coulomb_cutoff'),'kernel':('cell_slab_truncation','use_coulomb_truncation','bare_coulomb_cutoff'),'absorption':('cell_slab_truncation','use_coulomb_truncation','energy_range')}
# Updated together with main.py's embedded driver after the final edit.
EXPECTED_WORKFLOW_SHA256='d7ca36997bb1a548609aceb88508a69ecd122622bbc6903b454cf6fd67fba8f7'

def _rows(path):
    blocks=[]; cur=[]
    for raw in Path(path).read_text(errors='replace').splitlines():
        s=raw.strip()
        if not s:
            if cur: blocks.append(cur); cur=[]
            continue
        if s.startswith('#') or s.startswith('!'): continue
        q=s.replace(',',' ').split()
        try:
            a,b=float(q[0]),float(q[1])
            if math.isfinite(a) and math.isfinite(b): cur.append((a,b))
        except (ValueError,IndexError): pass
    if cur: blocks.append(cur)
    return blocks

def spectrum(path):
    out=[]
    for raw in Path(path).read_text(errors='replace').splitlines():
        s=raw.strip()
        if not s or s.startswith(('#','!')): continue
        q=s.replace(',',' ').split()
        try:
            e=float(q[0]); y=float(q[1]);
            if math.isfinite(e) and math.isfinite(y): out.append((e,y))
        except (ValueError,IndexError): pass
    return out

def _peaks(rows):
    peaks=[]
    for i in range(1,len(rows)-1):
        if rows[i][1]>=rows[i-1][1] and rows[i][1]>=rows[i+1][1] and rows[i][1]>0:
            peaks.append(rows[i])
    return sorted(peaks,key=lambda x:x[1],reverse=True)

def _sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def _manifest_checks(manifest, output_dir):
    failures=[]
    try: m=json.loads(Path(manifest).read_text())
    except Exception: return ['manifest-invalid']
    if m.get('schema_version')!=2: failures.append('manifest-schema')
    if m.get('runtime_image')!=EXPECTED_RUNTIME: failures.append('manifest-runtime-image')
    if m.get('workflow_sha256')!=EXPECTED_WORKFLOW_SHA256: failures.append('manifest-workflow-hash')
    expected_exe=['mpirun','pw.x','bands.x','pw2bgw.x','epsilon.cplx.x','sigma.cplx.x','kernel.cplx.x','absorption.cplx.x']
    if m.get('executables')!=expected_exe: failures.append('manifest-executables')
    stages=m.get('stages')
    if not isinstance(stages,list) or {x.get('stage') for x in stages if isinstance(x,dict)}!=set(EXPECTED_STAGES) or len(stages)!=len(EXPECTED_STAGES):
        failures.append('manifest-stage-set'); stages=[]
    by_stage={x.get('stage'):x for x in stages if isinstance(x,dict)}
    decks=m.get('decks')
    if not isinstance(decks,dict) or set(decks)!=set(EXPECTED_STAGES): failures.append('manifest-deck-map'); decks={}
    for name in EXPECTED_STAGES:
        s=by_stage.get(name,{})
        if s.get('return_code')!=0: failures.append('manifest-nonzero-return:'+name)
        path=s.get('deck_path')
        if not isinstance(path,str) or not Path(path).is_absolute() or not Path(path).is_file(): failures.append('manifest-deck-missing:'+name)
        else:
            if decks.get(name)!=path: failures.append('manifest-deck-mismatch:'+name)
            if s.get('deck_sha256')!=_sha256(path): failures.append('manifest-deck-hash:'+name)
        if s.get('executable')!=EXPECTED_EXECUTABLES[name]: failures.append('manifest-executable:'+name)
        command=s.get('command',[])
        if not isinstance(command,list) or 'mpirun' not in ' '.join(str(x) for x in command): failures.append('manifest-mpirun:'+name)
        if isinstance(path,str) and Path(path).is_file():
            deck=Path(path).read_text(errors='replace').lower()
            for token in DECK_EVIDENCE[name]:
                if token.lower() not in deck: failures.append('deck-missing-setting:'+name+':'+token)
            if name=='kernel' and ('use_diagonalization' in deck or 'diagonalization' in deck): failures.append('kernel-direct-diagonalization')
        if name in ('bands_post','absorption'):
            produced=s.get('output_path'); produced_hash=s.get('output_sha256')
            if not isinstance(produced,str) or not Path(produced).is_absolute() or not Path(produced).is_file(): failures.append('manifest-stage-output:'+name)
            elif produced_hash!=_sha256(produced): failures.append('manifest-stage-output-hash:'+name)
    out_hashes=m.get('output_hashes')
    if not isinstance(out_hashes,dict) or set(out_hashes)!=set(OUTPUTS): failures.append('manifest-output-hashes')
    else:
        for n in OUTPUTS:
            f=Path(output_dir)/n
            if out_hashes.get(n)!=_sha256(f): failures.append('manifest-output-hash:'+n)
    settings=m.get('settings',{})
    if settings.get('noncollinear') is not True or settings.get('lspinorb') is not True or '2d' not in str(settings.get('coulomb_truncation','')).lower(): failures.append('manifest-protocol')
    if settings.get('k_sampling')!='16x16x1' or int(settings.get('gw_summation_bands',0))<2000 or float(settings.get('screened_cutoff_Ry',0))<20 or settings.get('bse_fine_grid')!='32x32x1': failures.append('manifest-settings')
    prov=m.get('table_provenance')
    out_root=Path(output_dir).resolve()
    expected_sources={'bands':('bands_post','bands_post'),'absorption':('absorption','absorption')}
    if not isinstance(prov,dict): failures.append('manifest-table-provenance')
    else:
        by_stage={x.get('stage'):x for x in stages if isinstance(x,dict)}
        for key,(stage_name,_) in expected_sources.items():
            info=prov.get(key,{})
            source=Path(info.get('source_path','')) if isinstance(info,dict) else Path('')
            if info.get('stage')!=stage_name or not source.is_absolute() or not source.is_file(): failures.append('manifest-provenance-source:'+key); continue
            try:
                inside_output=source.resolve().is_relative_to(out_root)
            except AttributeError:
                inside_output=str(source.resolve()).startswith(str(out_root)+str(Path('/')))
            if inside_output: failures.append('manifest-provenance-inside-output:'+key)
            # A stage attestation must point to a runtime-produced table, not
            # an input, reference fixture, or arbitrary host file.  The stage
            # record is required to carry the identical absolute path and hash.
            staged_inputs={str((PACKAGE_ROOT/'input'/n).resolve()) for n in ('MoSe2.vasp','Mo.soc.upf','Se.soc.upf')}
            if str(source.resolve()) in staged_inputs: failures.append('manifest-provenance-input:'+key)
            if str(source.resolve()).startswith(('/etc/','/proc/','/sys/','/dev/')):
                failures.append('manifest-provenance-host-file:'+key)
            if info.get('source_sha256')!=_sha256(source): failures.append('manifest-provenance-hash:'+key)
            rec=by_stage.get(stage_name,{})
            if rec.get('output_path')!=str(source.resolve()) or rec.get('output_sha256')!=info.get('source_sha256'):
                failures.append('manifest-provenance-stage-mismatch:'+key)
    inputs=m.get('input_sha256',{})
    if inputs!=EXPECTED_INPUT_SHA256: failures.append('manifest-input-hashes')
    canonical={k:m.get(k) for k in ('schema_version','runtime_image','workflow_sha256','stages','settings','input_sha256','output_hashes','table_provenance')}
    expected=hashlib.sha256(json.dumps(canonical,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if m.get('attestation_sha256')!=expected: failures.append('manifest-attestation')
    return failures

def metrics(output_dir, manifest=None):
    p=Path(output_dir); failures=[]
    for n in OUTPUTS:
        if not (p/n).is_file() or (p/n).stat().st_size==0: failures.append('missing:'+n)
    if failures: return {'score':0.0,'failures':failures}
    for image in ('MoSe2_bands.png','exciton_absorption_spectra_avg.png'):
        if not (p/image).read_bytes().startswith(b'\x89PNG\r\n\x1a\n'): failures.append('invalid-png:'+image)
    bblocks=_rows(p/'MoSe2_bands.dat.gnu'); sr=spectrum(p/'absorption_eh.dat')
    for table in ('MoSe2_bands.dat.gnu','absorption_eh.dat'):
        for line_no, raw in enumerate((p/table).read_text(errors='replace').splitlines(),1):
            q=raw.strip()
            if not q or q.startswith(('#','!')): continue
            try: float(q.replace(',',' ').split()[0]); float(q.replace(',',' ').split()[1])
            except (ValueError,IndexError): failures.append('malformed:'+table+':'+str(line_no))
    comments=(p/'MoSe2_bands.dat.gnu').read_text(errors='replace').splitlines()[:12]+(p/'absorption_eh.dat').read_text(errors='replace').splitlines()[:12]
    text=' '.join(comments).lower()
    for marker in REQUIRED_MARKERS:
        if marker.lower() not in text: failures.append('missing-workflow-marker:'+marker)
    if len(bblocks)<4 or sum(map(len,bblocks))<50: failures.append('bands-too-small')
    if len(sr)<100: failures.append('spectrum-too-small')
    if any(sr[i][0]>=sr[i+1][0] for i in range(len(sr)-1)): failures.append('energy-grid-not-increasing')
    if sr and (max(y for _,y in sr)<=0 or max(e for e,_ in sr)-min(e for e,_ in sr)<1.0): failures.append('spectrum-no-range')
    # The edge calculation is based on the two highest-energy valence blocks and
    # two lowest-energy conduction blocks, with no assumed grid or row count.
    bmetrics={}
    if len(bblocks)>=4:
        ext=[(max(y for _,y in b),min(y for _,y in b),b) for b in bblocks if b]
        ext.sort(key=lambda x:x[0])
        v=sorted([x for x in ext if x[0] <= 0.5],key=lambda x:x[0],reverse=True)
        c=sorted([x for x in ext if x[1] > 0.5],key=lambda x:x[1])
        if len(v)>=2 and c:
            bmetrics['vbm_eV']=v[0][0]; bmetrics['vbm2_eV']=v[1][0]; bmetrics['soc_splitting_meV']=1000*(v[0][0]-v[1][0]); bmetrics['cbm_eV']=c[0][1]; bmetrics['direct_gap_eV']=bmetrics['cbm_eV']-bmetrics['vbm_eV']
            bmetrics['edge_k_v']=v[0][2][max(range(len(v[0][2])),key=lambda i:v[0][2][i][1])][0]; bmetrics['edge_k_c']=c[0][2][min(range(len(c[0][2])),key=lambda i:c[0][2][i][1])][0]
            if bmetrics['soc_splitting_meV']<20 or bmetrics['soc_splitting_meV']>500: failures.append('unresolved-soc-splitting')
            if bmetrics['direct_gap_eV']<0.3 or bmetrics['direct_gap_eV']>4.0: failures.append('unphysical-gap')
            if abs(bmetrics['edge_k_v']-bmetrics['edge_k_c'])>0.15: failures.append('band-edges-not-coincident')
        else: failures.append('cannot-identify-valence-conduction')
    peaks=_peaks(sr)
    if len(peaks)<2: failures.append('missing-excitonic-features')
    elif abs(peaks[0][0]-peaks[1][0])<0.03: failures.append('unresolved-excitonic-separation')
    # Compare the normalized absorption curve with the released reference after
    # interpolation.  Peak markers alone cannot satisfy this physical check.
    ys=[y for _,y in sr]; variation=max(ys)-min(ys)
    def trap(rows):
        return sum(0.5*(max(0.0,rows[i-1][1])+max(0.0,rows[i][1]))*(rows[i][0]-rows[i-1][0]) for i in range(1,len(rows)))
    # Absorption is a non-negative response; normalize by its integrated area
    # so vertical marker tricks and arbitrary baselines cannot pass.
    if any(y<0 for _,y in sr): failures.append('negative-absorption')
    area=trap(sr)
    if area<=0 or variation<=0.02*max(ys): failures.append('static-spectrum')
    else:
        settings={}
        try: settings=json.loads(Path(manifest).read_text()).get('settings',{}) if manifest is not None else {}
        except Exception: settings={}
        ref_name='hidden_case' if settings.get('hidden_variant') is True else 'visible'
        ref_path=Path(__file__).resolve().parents[1]/'reference'/ref_name/'absorption_eh.dat'
        try:
            ref=spectrum(ref_path)
            def interp(rows,x):
                if x<rows[0][0] or x>rows[-1][0]: return None
                lo,hi=0,len(rows)-1
                while hi-lo>1:
                    mid=(lo+hi)//2
                    if rows[mid][0]<=x: lo=mid
                    else: hi=mid
                x0,y0=rows[lo]; x1,y1=rows[hi]
                return y0+(y1-y0)*(x-x0)/(x1-x0) if x1!=x0 else y0
            ref_area=trap(ref); cand_area=trap(sr)
            if ref_area<=0 or cand_area<=0: failures.append('spectrum-normalization')
            else:
                errs=[]; covered=0
                for x,y in ref:
                    yi=interp(sr,x)
                    if yi is None: continue
                    covered+=1; errs.append((yi/cand_area-y/ref_area)**2)
                if covered<0.95*len(ref) or not errs: failures.append('spectrum-grid-coverage')
                elif math.sqrt(sum(errs)/len(errs))>0.08: failures.append('spectrum-shape-mismatch')
        except Exception: failures.append('spectrum-reference-unavailable')
    if manifest is not None: failures.extend(_manifest_checks(manifest,p))
    else: failures.append('manifest-required')
    score=0.0 if failures else 1.0
    return {'score':score,'failures':failures,'metrics':bmetrics,'peak_energies_eV':[x[0] for x in peaks[:3]],'spectrum_points':len(sr)}

def validate(output_dir, manifest=None): return metrics(output_dir,manifest)
def solve(output_dir, manifest=None):
    """Compatibility entry point used by local harnesses."""
    return metrics(output_dir, manifest)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('output_dir'); ap.add_argument('--manifest'); a=ap.parse_args()
    print(json.dumps(metrics(a.output_dir,a.manifest),sort_keys=True))
