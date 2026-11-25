from pathlib import Path
import tempfile,csv,json,shutil,sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from scripts.oracle import write_reference,capture,compute
from scripts.grader import score_dirs

ROOT=Path(__file__).parents[1]; hidden=ROOT/'reference/base/materials_hidden.csv'
with tempfile.TemporaryDirectory() as td:
 t=Path(td); ref=t/'ref'; write_reference(hidden,ref)
 perfect=t/'perfect'; write_reference(hidden,perfect); assert score_dirs(perfect,ref)['score']>0.999
 empty=t/'empty'; empty.mkdir(); assert score_dirs(empty,ref)['score']==0.0
 assert capture(-1.0)==0.0 and capture(0.0005)==0.0 and capture(8.001)==0.0
 tie=[{'material_id':'t1','A':'Cs','B':'Ge','X':'Br','a_A':'5.8','c_A':'5.8','tilt_deg':'8'},{'material_id':'t2','A':'Cs','B':'Ge','X':'Br','a_A':'5.8','c_A':'5.8','tilt_deg':'8'}]
 assert compute(tie)['summary']['best_material_id']=='t1'
 wrong=t/'wrong'; shutil.copytree(perfect,wrong); p=wrong/'band_gaps.csv'; rows=list(csv.DictReader(p.open())); rows[0]['band_gap_eV']='nan'
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['material_id','band_gap_eV','soc_applied']); w.writeheader(); w.writerows(rows)
 assert score_dirs(wrong,ref)['score']==0.0
 reordered=t/'reordered'; shutil.copytree(perfect,reordered); p=reordered/'band_gaps.csv'; rows=list(csv.DictReader(p.open())); rows[0],rows[1]=rows[1],rows[0]
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 assert score_dirs(reordered,ref)['score']==0.0
 falseinv=t/'falseinv'; shutil.copytree(perfect,falseinv); ip=falseinv/'invariants.json'; d=json.loads(ip.read_text()); d['soc_only_for_pb']=False; ip.write_text(json.dumps(d))
 assert score_dirs(falseinv,ref)['score']<1.0
 badrow=t/'badrow'; shutil.copytree(perfect,badrow); p=badrow/'spectral_capture.csv'; txt=p.read_text().rstrip()+'\nextra,1,unexpected\n'; p.write_text(txt); assert score_dirs(badrow,ref)['score']==0.0
print('adversarial checks passed')
