#!/usr/bin/env python3
"""Independent deterministic oracle for the authored lake calibration fixture."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
DEPTHS=(0.0,2.0,5.0,10.0,15.0)
TRUE={'k_mix':0.235,'light_extinction':0.165,'air_exchange':0.041,'sediment_flux':0.014,'initial_temp':5.65}

def forcing(n,variant):
    rows=[]
    for d in range(n):
        phase=0.17 if variant=='visible' else 0.34
        air=4.8+13.5*math.sin(2*math.pi*(d+phase*365)/365)+1.7*math.cos(2*math.pi*d/31)
        sw=max(0.0,205*math.sin(2*math.pi*(d-28+phase*30)/365))
        if variant=='hidden' and 82<=d<=89: air-=7.0; sw*=0.72
        wind=2.8+1.3*math.sin(2*math.pi*(d+11)/19)
        rows.append((d,air,sw,wind))
    return rows

def run(p,forcing_rows):
    state=[p['initial_temp']-0.025*z for z in DEPTHS]; rows=[]
    for d,air,sw,wind in forcing_rows:
        for z,t in zip(DEPTHS,state): rows.append((d,z,t))
        nxt=[]
        for i,(z,t) in enumerate(zip(DEPTHS,state)):
            a=state[i-1] if i else t; b=state[i+1] if i+1<len(state) else t
            optical=math.exp(-p['light_extinction']*z)
            nxt.append(t + p['k_mix']*(a-2*t+b)/4*(1+0.12*wind) + 0.00072*sw*optical + p['air_exchange']*(air-t)*math.exp(-0.11*z) + p['sediment_flux']*math.exp(-z/12)*(5.5-t)/10)
        state=nxt
    return rows

def write_forcing(path,rows):
    with path.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(('day','air_temp_c','shortwave_w_m2','wind_m_s'))
        for d,a,s,wnd in rows: w.writerow((d,f'{a:.9f}',f'{s:.9f}',f'{wnd:.9f}'))
def write_obs(path,rows):
    with path.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(('day','depth_m','temp_c'))
        for d,z,t in rows:
            bias=0.018*math.sin(0.37*d+0.19*z)+0.006*math.cos(0.11*d-0.07*z)
            w.writerow((d,f'{z:.3f}',f'{t+bias:.9f}'))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('input_dir'); ap.add_argument('reference_dir'); a=ap.parse_args(); inp=Path(a.input_dir); ref=Path(a.reference_dir); ref.mkdir(parents=True,exist_ok=True)
    vis=forcing(120,'visible'); hid=forcing(147,'hidden'); vr=run(TRUE,vis); hr=run(TRUE,hid)
    write_forcing(inp/'forcing_visible.csv',vis); write_obs(inp/'observations_visible.csv',vr)
    write_forcing(ref/'forcing_hidden.csv',hid); write_obs(ref/'observations_hidden.csv',hr)
    (ref/'oracle_parameters.json').write_text(json.dumps({'parameter_names':list(TRUE),'generation':'benchmark-authored deterministic equations in oracle.py'},sort_keys=True,indent=2)+'\n')
    (ref/'oracle_diagnostics.json').write_text(json.dumps({'visible_rows':len(vr),'hidden_rows':len(hr),'visible_days':120,'hidden_days':147,'depths_m':DEPTHS},sort_keys=True,indent=2)+'\n')
if __name__=='__main__': main()
