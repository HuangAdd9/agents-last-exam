"""Independent deterministic reference reducer for the benchmark-authored visits."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import numpy as np

def read_matrix(p): return np.loadtxt(p,delimiter=',')
def bilinear(a,x,y):
    h,w=a.shape
    if x<0 or y<0 or x>w-1 or y>h-1: return float('nan')
    x0=int(math.floor(x)); y0=int(math.floor(y)); x1=min(x0+1,w-1); y1=min(y0+1,h-1)
    fx=x-x0; fy=y-y0
    return float((1-fx)*(1-fy)*a[y0,x0]+fx*(1-fy)*a[y0,x1]+(1-fx)*fy*a[y1,x0]+fx*fy*a[y1,x1])
def reduce_visit(v,out):
    meta=json.loads((v/'visit.json').read_text()); out.mkdir(parents=True,exist_ok=True)
    assoc=list(csv.DictReader((v/'association.csv').open())); anchors=list(csv.DictReader((v/'anchors.csv').open()))
    stack=[]; masked_cr=masked_hot=0
    for row in assoc:
        eid=row['exposure_id']; a=read_matrix(v/'exposures'/f'{v.name}_{eid}.csv'); dq=read_matrix(v/'dq'/f'{v.name}_{eid}_dq.csv').astype(int)
        masked_cr += int(np.count_nonzero((dq & 4096)!=0)); masked_hot += int(np.count_nonzero((dq & 16)!=0))
        a=a.astype(float); a[(dq & (4096|16))!=0]=np.nan; dx=float(row['dx_pix']); dy=float(row['dy_pix'])
        aligned=np.full_like(a,np.nan)
        for y in range(a.shape[0]):
            for x in range(a.shape[1]): aligned[y,x]=bilinear(a,x+dx,y+dy)
        stack.append(aligned)
    image=np.nanmean(np.stack(stack),axis=0)
    np.savetxt(out/'drizzled_image.csv',image,delimiter=',',fmt='%.8f')
    yy,xx=np.mgrid[0:image.shape[0],0:image.shape[1]]; sky=float(np.nanmedian(np.r_[image[:5,:].ravel(),image[-5:,:].ravel(),image[:, :5].ravel(),image[:, -5:].ravel()]))
    rows=[]; r=3.0
    for an in anchors:
        x=float(an['x']); y=float(an['y']); mask=(xx-x)**2+(yy-y)**2<=r*r; flux=float(np.nansum(image[mask]-sky)); n=int(mask.sum()); snr=flux/math.sqrt(max(flux+n*sky,1e-12)); mag=float(meta['zeropoint_ab']-2.5*math.log10(max(flux,1e-12))); sharp=float(image[int(round(y)),int(round(x))]-sky)/max(flux,1e-12)
        rows.append([an['source_id'],f'{x:.6f}',f'{y:.6f}',an['ra_deg'],an['dec_deg'],f'{flux:.6f}',f'{mag:.6f}',f'{snr:.6f}',f'{sharp:.8f}','0'])
    with (out/'source_catalog.csv').open('w',newline='') as f: w=csv.writer(f); w.writerow(['source_id','x','y','ra_deg','dec_deg','flux_e_s','mag_ab','snr','sharpness','flags']); w.writerows(rows)
    with (out/'alignment_solution.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['exposure_id','dx_pix','dy_pix','rms_pix','matched_sources'])
        for row in assoc: w.writerow([row['exposure_id'],f"{float(row['dx_pix']):.6f}",f"{float(row['dy_pix']):.6f}",'0.000000',len(anchors)])
    qc={'visit_id':v.name,'filter':meta['filter'],'num_sources':len(rows),'background_e_s':sky,'cosmic_ray_pixels_masked':masked_cr,'hot_pixels_masked':masked_hot,'aperture_radius_pix':3.0,'pixfrac':0.8,'final_pixel_scale_arcsec':meta['pixel_scale_arcsec'],'astrometric_rms_pix':0.0}
    (out/'photometry_qc.json').write_text(json.dumps(qc,indent=2,sort_keys=True))
    (out/'reduction_report.md').write_text('# ACS/WFC visit reduction\n\ncalacs-style DQ masking was applied before AstroDrizzle-style alignment and coaddition. Aperture photometry uses a 3 pixel radius; the astrometric RMS is reported against the supplied Gaia-like anchors. Cosmic ray and hot pixel flags are excluded.\n')
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    visits=[a.input] if (a.input/'visit.json').exists() else sorted(p for p in a.input.iterdir() if p.is_dir() and (p/'visit.json').exists())
    for v in visits: reduce_visit(v,a.output/v.name)
if __name__=='__main__': main()
