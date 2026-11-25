from __future__ import annotations
import argparse,csv,json,math,statistics
from pathlib import Path

SRC=['source_id','x','y','ra_deg','dec_deg','flux_e_s','mag_ab','snr','sharpness','flags']
ALI=['exposure_id','dx_pix','dy_pix','rms_pix','matched_sources']
FILES={'drizzled_image.csv','source_catalog.csv','alignment_solution.csv','photometry_qc.json','reduction_report.md'}
def rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def image_rows(p):
    with p.open(newline='') as f:
        return [[float(cell) for cell in row] for row in csv.reader(f) if row]
def visit_dirs(root):
    if not root.exists(): return []
    direct=[p for p in root.iterdir() if p.is_dir() and (p/'source_catalog.csv').exists()]
    if direct:return sorted(direct)
    out=[]
    for split in ('visible','hidden'):
        d=root/split
        if d.exists():out.extend(p for p in d.iterdir() if p.is_dir())
    return sorted(out)
def score_visit(out,ref):
    if not out.is_dir() or any(not (out/f).is_file() for f in FILES): return 0.0,['missing required artifact']
    try:
        s=rows(out/'source_catalog.csv'); sr=rows(ref/'source_catalog.csv'); a=rows(out/'alignment_solution.csv'); ar=rows(ref/'alignment_solution.csv')
        im=image_rows(out/'drizzled_image.csv'); ir=image_rows(ref/'drizzled_image.csv')
        q=json.loads((out/'photometry_qc.json').read_text()); qr=json.loads((ref/'photometry_qc.json').read_text()); report=(out/'reduction_report.md').read_text().lower()
    except Exception as e:return 0.0,[f'parse failure: {e}']
    score=0.; notes=[]
    if s and list(s[0])==SRC: score+=.05
    if a and list(a[0])==ALI: score+=.05
    if (len(im)==len(ir)==72 and all(len(row)==72 for row in im+ir)
            and all(math.isfinite(value) for row in im for value in row)):
        squared_errors=[(x-y)**2 for row,ref_row in zip(im,ir) for x,y in zip(row,ref_row)]
        rmse=math.sqrt(statistics.mean(squared_errors)); score += .25 if rmse<=0.5 else .12 if rmse<=2.0 else 0.;
        if rmse>2:notes.append(f'image RMSE {rmse:.3f}')
    pairs=[]; used=set()
    for rr in sr:
        best=None
        for i,x in enumerate(s):
            if i in used:continue
            d=math.hypot(float(x['x'])-float(rr['x']),float(x['y'])-float(rr['y']))
            if best is None or d<best[0]:best=(d,i,x)
        if best and best[0]<=0.4:used.add(best[1]);pairs.append((best[2],rr,best[0]))
    comp=len(pairs)/max(1,len(sr)); score += .15*comp
    if pairs:
        ferr=statistics.median([abs(float(x['flux_e_s'])-float(y['flux_e_s']))/max(abs(float(y['flux_e_s'])),1e-9) for x,y,_ in pairs]); merr=statistics.median([abs(float(x['mag_ab'])-float(y['mag_ab'])) for x,y,_ in pairs]); serr=statistics.median([math.hypot((float(x['ra_deg'])-float(y['ra_deg']))*3600,(float(x['dec_deg'])-float(y['dec_deg']))*3600) for x,y,_ in pairs]); score += .15*(1 if ferr<=.08 else .5 if ferr<=.16 else 0); score += .10*(1 if merr<=.06 else .5 if merr<=.12 else 0); score += .10*(1 if serr<=.04 else .5 if serr<=.08 else 0)
    if len(a)==len(ar) and len(a)>0:
        e=max([abs(float(x[k])-float(y[k])) for x,y in zip(a,ar) for k in ('dx_pix','dy_pix')]); score += .10 if e<=.05 else .05 if e<=.12 else 0
    qscore=0
    for k in ('cosmic_ray_pixels_masked','hot_pixels_masked','num_sources'):
        if q.get(k)==qr.get(k):qscore+=.02
    if abs(float(q.get('astrometric_rms_pix',9))-float(qr.get('astrometric_rms_pix',0)))<=.04:qscore+=.02
    if float(q.get('aperture_radius_pix',0))==3.0 and float(q.get('pixfrac',0))==.8:qscore+=.02
    score+=qscore
    score+=.0125*sum(p in report for p in ('calacs-style','astrodrizzle-style','astrometric rms','cosmic ray'))
    return min(1.,score),notes
def evaluate(output,reference):
    refs=visit_dirs(reference); vals=[]; notes=[]
    for r in refs:
        o=output/r.name
        if r.parent.name in ('visible','hidden') and not o.exists(): o=output/r.parent.name/r.name
        if not o.exists() and (output/'results'/r.name).exists():o=output/'results'/r.name
        v,n=score_visit(o,r); vals.append(v); notes += [f'{r.name}: {x}' for x in n]
    return {'score':float(statistics.mean(vals)) if vals else 0.0,'visit_scores':vals,'passed':bool(vals and min(vals)>=.8),'notes':notes[:20]}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--reference-dir',type=Path,required=True);a=p.parse_args();print(json.dumps(evaluate(a.output_dir,a.reference_dir),sort_keys=True))
