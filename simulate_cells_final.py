#!/usr/bin/env python3

import os, argparse, logging
import numpy as np, pandas as pd
from brian2 import *
from sklearn.utils import resample

# ──────────────────────────────────────────────────────────
# Configuración general
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
prefs.codegen.target = "numpy"
# ──────────────────────────────────────────────────────────

def diagnosticar_dataset(df, name):
    if df is None or df.empty:
        logging.warning(f"Dataset «{name}» vacío.")
        return
    logging.info(f"Dataset «{name}»: {len(df)} filas.")
    if 'spike' in df:
        vc = df['spike'].value_counts(normalize=True)*100
        logging.info(f"  % clases: {{0: {vc.get(0,0):.1f}%, 1: {vc.get(1,0):.1f}%}}")
    for col in ['voltage_mV','input_current_nA']:
        if col in df:
            s = df[col].describe()
            logging.info(f"  {col}: mean={s['mean']:.2f}, std={s['std']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}")

def procesar_multi_neurona(name, mon, spikes, I_global, dt_ms, out_dir,
                           peri_spike_ms=5.0, peri_ratio=0.5,
                           no_spike_factor=10, max_no_spikes=15000):
    t = (mon.t/ms).astype(float)
    neuron_indices = list(spikes.spike_trains().keys()) or list(range(len(mon.v)))
    dfs = []
    for idx in neuron_indices:
        v = (mon.v[idx]/mV).astype(float)
        st = spikes.spike_trains().get(idx,[])/ms
        bins = np.round(st/dt_ms).astype(int)
        mask = np.zeros_like(t, bool)
        mask[np.clip(bins,0,len(t)-1)] = True

        spikes_i = np.where(mask)[0]
        nospikes_i = np.where(~mask)[0]

        # Muestreo peri-spike
        w = int(peri_spike_ms/dt_ms)
        peri = np.unique(np.concatenate([np.arange(max(0,i-w),i) for i in spikes_i])) if spikes_i.size else np.array([],int)
        peri = np.intersect1d(peri, nospikes_i)

        # Cuántos no-spikes tomar
        n_keep = min(len(spikes_i)*no_spike_factor, max_no_spikes)
        k_peri = int(n_keep*peri_ratio)
        k_rest = n_keep - k_peri

        sel_peri = np.random.choice(peri, min(k_peri,len(peri)), replace=False) if peri.size else np.array([],int)
        rest = np.setdiff1d(nospikes_i, peri)
        sel_rest = np.random.choice(rest, min(k_rest,len(rest)), replace=False) if rest.size else np.array([],int)

        final_idx = np.sort(np.concatenate([spikes_i, sel_peri, sel_rest])).astype(int)
        df = pd.DataFrame({
            'time_ms': t[final_idx],
            'voltage_mV': v[final_idx],
            'input_current_nA': I_global[final_idx],
            'spike': mask[final_idx].astype(int)
        })
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)
    diagnosticar_dataset(df_all, name)

    # Balanceo 1:1
    pos = df_all[df_all.spike==1]
    neg = df_all[df_all.spike==0]
    if len(pos)==0 or len(neg)==0:
        df_bal = df_all.sample(frac=1, random_state=42)
    else:
        if len(pos)<len(neg):
            pos = resample(pos, replace=True, n_samples=len(neg), random_state=42)
        else:
            neg = resample(neg, replace=True, n_samples=len(pos), random_state=42)
        df_bal = pd.concat([pos,neg]).sample(frac=1, random_state=42)

    diagnosticar_dataset(df_bal, name+"_balanced")

    os.makedirs(out_dir, exist_ok=True)
    df_bal.to_csv(f"{out_dir}/{name}_kan_ready.csv", index=False)
    n_light = min(100_000, len(df_bal))
    df_bal.sample(n=n_light, random_state=42).to_csv(f"{out_dir}/{name}_light.csv", index=False)
    logging.info(f"Guardados CSV para «{name}» en «{out_dir}»")

def simular_lif(name, params, args):
    start_scope()
    defaultclock.dt = args.dt*ms

    eqs = '''
    dv/dt = (-g_L*(v-EL) + I_syn)/C : volt
    dI_syn/dt = (-I_syn + I0)/tau + sigma*xi/sqrt(tau) : amp
    g_L : siemens
    C : farad
    EL : volt
    I0 : amp
    tau : second
    sigma : amp
    '''
    G = NeuronGroup(
        args.n, eqs,
        threshold='v>Vth', reset='v=Vres',
        method='euler',
        namespace={'Vth': params['Vth'], 'Vres': params['Vres']}
    )
    for p in ['g_L','C']:
        base = params[p]
        noise = 1 + args.het*(np.random.rand(args.n)-0.5)
        setattr(G, p, base * noise)
    for pname in ['EL','I0','tau','sigma']:
        setattr(G, pname, params[pname])
    G.v = params['EL']; G.I_syn = params['I0']

    mon = StateMonitor(G, ['v','I_syn'], record=True, dt=args.dt*ms)
    spk = SpikeMonitor(G)
    logging.info(f"Simulando LIF «{name}» por {args.T} ms...")
    run(args.T*ms)

    I_glob = (mon.I_syn[0]/nA).astype(float)
    procesar_multi_neurona(name, mon, spk, I_glob, args.dt, args.out)

def simular_purkinje(name, args):
    start_scope()
    dt = 0.01
    defaultclock.dt = dt*ms

    # Estímulo ruidoso
    steps = int(args.T/dt)
    noise = np.random.randn(steps)
    w = max(1, int(1/(args.f*dt))) if args.f>0 else 1
    smooth = np.convolve(noise, np.ones(w)/w, mode='same')
    Iarr = args.I0 + args.Istd * smooth
    I_t = TimedArray(Iarr * uA/cm**2, dt=dt*ms)

    eqs = '''
    dv/dt = (I_t(t)
             - gNa*m**3*h*(v-ENa)
             - gK*n**4*(v-EK)
             - gL*(v-EL)) / Cm : volt

    dm/dt = alpha_m*(1-m) - beta_m*m : 1
    dh/dt = alpha_h*(1-h) - beta_h*h : 1
    dn/dt = alpha_n*(1-n) - beta_n*n : 1

    alpha_m = 0.1*((25*mV - v)/mV)/(exp((25*mV - v)/(10*mV)) - 1)/ms : Hz
    beta_m  = 4*exp(-v/(18*mV))/ms : Hz

    alpha_h = 0.07*exp(-v/(20*mV))/ms : Hz
    beta_h  = 1/(exp((30*mV - v)/(10*mV))+1)/ms : Hz

    alpha_n = 0.01*((10*mV - v)/mV)/(exp((10*mV - v)/(10*mV)) - 1)/ms : Hz
    beta_n  = 0.125*exp(-v/(80*mV))/ms : Hz
    '''
    G = NeuronGroup(
        args.n, eqs,
        threshold='v>-40*mV', reset='v=-65*mV',
        method='exponential_euler',
        namespace={
            'Cm':1*uF/cm**2, 'gNa':120*msiemens/cm**2,
            'gK':36*msiemens/cm**2, 'gL':0.3*msiemens/cm**2,
            'ENa':50*mV,   'EK':-77*mV,      'EL':-54.4*mV,
            'I_t': I_t
        }
    )
    G.v, G.m, G.h, G.n = -65*mV, 0.05, 0.6, 0.32

    # **Registrar todas las neuronas**
    mon = StateMonitor(G, 'v', record=True, dt=dt*ms)
    spk = SpikeMonitor(G)
    logging.info(f"Simulando HH «{name}» por {args.T} ms...")
    run(args.T*ms)

    # Convertir Iarr (uA/cm2) a nA para compatibilidad
    I_glob = (Iarr * uA/cm**2 / nA).astype(float)
    procesar_multi_neurona(name, mon, spk, I_glob, dt, args.out)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='dataset_cerebelo')
    p.add_argument('--dt',  type=float, default=0.1)
    p.add_argument('--T',   type=float, default=2000)
    p.add_argument('--n',   type=int,   default=100)
    p.add_argument('--het', type=float, default=0.05)
    p.add_argument('--I0',  type=float, default=20.0)
    p.add_argument('--Istd',type=float, default=10.0)
    p.add_argument('--f',   type=float, default=0.2)
    args = p.parse_args()

    lifs = {
      'granule_lif':    {'g_L':5*nS,'C':100*pF,'EL':-70*mV,'Vth':-50*mV,'Vres':-65*mV,'I0':0.8*nA,'tau':5*ms,'sigma':0.2*nA},
      'golgi_lif':      {'g_L':10*nS,'C':200*pF,'EL':-68*mV,'Vth':-52*mV,'Vres':-65*mV,'I0':1.0*nA,'tau':5*ms,'sigma':0.2*nA},
      'basket_lif':     {'g_L':12*nS,'C':150*pF,'EL':-67*mV,'Vth':-50*mV,'Vres':-65*mV,'I0':1.0*nA,'tau':5*ms,'sigma':0.2*nA},
      'stellate_lif':   {'g_L':10*nS,'C':180*pF,'EL':-68*mV,'Vth':-52*mV,'Vres':-66*mV,'I0':0.8*nA,'tau':5*ms,'sigma':0.2*nA},
      'deep_nuclei_lif':{'g_L':15*nS,'C':300*pF,'EL':-65*mV,'Vth':-50*mV,'Vres':-66*mV,'I0':1.1*nA,'tau':5*ms,'sigma':0.2*nA},
      'mossy_fiber':    {'g_L':10*nS,'C':150*pF,'EL':-70*mV,'Vth':-50*mV,'Vres':-70*mV,'I0':0.8*nA,'tau':5*ms,'sigma':0.2*nA},
      'climbing_fiber': {'g_L':8*nS, 'C':100*pF,'EL':-65*mV,'Vth':-45*mV,'Vres':-68*mV,'I0':0.5*nA,'tau':5*ms,'sigma':0.2*nA},
    }

    os.makedirs(args.out, exist_ok=True)
    for name, params in lifs.items():
        simular_lif(name, params, args)
    simular_purkinje('purkinje_hh_dinamico', args)

if __name__=="__main__":
    main()
