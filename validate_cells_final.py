#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from brian2 import (
    prefs,
    start_scope,
    defaultclock,
    run,
    ms, mV, nA, nS, pF,
    TimedArray,
    NeuronGroup,
    StateMonitor,
    SpikeMonitor,
)

import torch

from Cerebellar_Class import NeuronaCerebelarKAN, BASE_MODEL_DIR, DEVICE

# ──────────────────────────────────────────────────────────────────────────────
# Brian2: usar siempre numpy target para permitir múltiples run()
prefs.codegen.target = 'numpy'

# Suprimir warnings de KAN (división inválida en escala de splines)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore",
    message="std\\(\\): degrees of freedom is <= 0\\.",
    module="kan.MultKAN"
)
# ──────────────────────────────────────────────────────────────────────────────
speedup_records = []
def generar_estimulos_prueba(duracion_ms, dt_ms):
    pasos = int(duracion_ms / dt_ms)
    t = np.arange(pasos) * dt_ms
    pulso = np.zeros(pasos)
    pulso[int(100/dt_ms):int(400/dt_ms)] = 0.6
    rampa = np.linspace(0, 0.8, pasos)
    sinuso = 0.4 + 0.3 * np.sin(2 * np.pi * 5 * t / 1000)
    cortos = np.zeros(pasos)
    for s in [100, 200, 300, 400]:
        cortos[int(s/dt_ms):int((s+20)/dt_ms)] = 0.9
    return {"Pulso": pulso, "Rampa": rampa, "Sinusoidal": sinuso, "Cortos": cortos}, t

def sim_brian2_lif(params, corriente, T_ms, dt_ms):
    start_scope()
    defaultclock.dt = dt_ms * ms
    Iext = TimedArray(corriente * nA, dt=defaultclock.dt)
    eqs = '''
    dv/dt = (-g_L*(v-EL) + Iext(t))/C : volt
    g_L : siemens
    C   : farad
    EL  : volt
    '''
    G = NeuronGroup(1, eqs,
                    threshold='v>V_th', reset='v=V_res',
                    method='euler',
                    namespace={'V_th': params['V_th_lif'],
                               'V_res': params['V_res_lif']})
    G.g_L = params['g_L']
    G.C   = params['C']
    G.EL  = params['EL_lif']
    mon_v = StateMonitor(G, 'v', record=0)
    mon_sp = SpikeMonitor(G)
    run(T_ms * ms)
    t = mon_v.t / ms
    v = mon_v.v[0] / mV
    spikes = mon_sp.t / ms
    return t, v, spikes

def sim_kan_bucle(neurona, params, corriente, t_ms, dt_ms):
    pasos = len(t_ms)
    v_kan = np.zeros(pasos)
    v_kan[0] = params['EL_lif'] / mV
    spikes = []
    for i in range(pasos - 1):
        df = pd.DataFrame({
            'time_ms': [t_ms[i]],
            'voltage_mV': [v_kan[i]],
            'input_current_nA': [corriente[i]]
        })
        _, pred = neurona.predecir(df)
        if pred is not None and pred[0][0] == 1:
            v_kan[i+1] = params['V_res_lif'] / mV
            spikes.append(t_ms[i])
        else:
            dv = ((-params['g_L'] * (v_kan[i]*mV - params['EL_lif'])) +
                  corriente[i]*nA) / params['C']
            v_kan[i+1] = (v_kan[i]*mV + dv * (dt_ms*ms)) / mV
    return v_kan, np.array(spikes)

def validar_generalizacion(name, params, dt_ms):
    print(f"\n--- Generalización: {name} ---")
    neur = NeuronaCerebelarKAN(name)
    if not neur.cargar_modelo(): return
    estim, t = generar_estimulos_prueba(500, dt_ms)
    fig, axes = plt.subplots(len(estim), 1, figsize=(8, 3*len(estim)), sharex=True)
    for ax, (key, cur) in zip(axes, estim.items()):
        # Brian2
        t_gt, v_gt, sp_gt = sim_brian2_lif(params, cur, 500, dt_ms)
        # KAN
        v_kan, sp_kan = sim_kan_bucle(neur, params, cur, t, dt_ms)
        ax.plot(t, v_gt, label='Brian2')
        ax.plot(t, v_kan, '--', label='KAN')
        # Aquí la corrección: y debe ser un array
        if len(sp_gt)>0:
            y0 = np.full_like(sp_gt, np.min(v_gt)-5)
            ax.plot(sp_gt, y0, '|', label='spk GT')
        if len(sp_kan)>0:
            y1 = np.full_like(sp_kan, np.min(v_gt)-8)
            ax.plot(sp_kan, y1, 'x', label='spk KAN')
        ax.set_title(key)
        ax.legend()
        ax.grid(True)
    axes[-1].set_xlabel("Tiempo (ms)")
    fig.suptitle(f"Generalización: {name}", y=1.02)
    plt.tight_layout()
    out = os.path.join(BASE_MODEL_DIR, f"gen_{name}.png")
    fig.savefig(out); plt.close(fig)
    print("→", out)

def validar_rendimiento(name, params, dt_ms):
    print(f"\n--- Rendimiento: {name} ---")
    neur = NeuronaCerebelarKAN(name)
    if not neur.cargar_modelo(): return
    estim, _ = generar_estimulos_prueba(2000, dt_ms)
    sin = estim['Sinusoidal']
    t1 = time.time()
    sim_brian2_lif(params, sin, 2000, dt_ms)
    t1 = time.time() - t1
    df = pd.DataFrame({
        'time_ms': np.arange(len(sin))*dt_ms,
        'voltage_mV': np.zeros_like(sin),
        'input_current_nA': sin
    })
    t2 = time.time()
    neur.predecir(df)
    t2 = time.time() - t2
    print(f"Brian2: {t1:.3f}s  |  KAN: {t2:.3f}s  →  {t1/t2:.1f}×")
    speedup = t1 / t2 if t2>0 else np.nan
    speedup_records.append({
        "cell": name,
        "brian2_time_s": t1,
        "kan_time_s": t2,
        "speedup": speedup
    })

def validar_interpretabilidad(name):
    print(f"\n--- Interpretabilidad: {name} ---")
    neur = NeuronaCerebelarKAN(name)
    if not neur.cargar_modelo(): return
    df0 = pd.read_csv(neur.ruta_datos_csv, nrows=100)
    neur.predecir(df0)
    try:
        neur.modelo_kan_cargado.plot(beta=10)
        fig = plt.gcf()
        fig.suptitle(f"Splines: {name}")
        out = os.path.join(BASE_MODEL_DIR, f"spline_{name}.png")
        fig.savefig(out); plt.close(fig)
        print("→", out)
    except Exception as e:
        print("⚠ interpretabilidad falló:", e)

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    CONFIG = {
        "granule_lif":      {"params": {"g_L":5*nS,   "C":100*pF, "EL_lif":-70*mV, "V_th_lif":-50*mV, "V_res_lif":-65*mV}, "dt_ms":0.1},
        "golgi_lif":        {"params": {"g_L":10*nS,  "C":200*pF, "EL_lif":-68*mV, "V_th_lif":-52*mV, "V_res_lif":-65*mV}, "dt_ms":0.1},
        "basket_lif":       {"params": {"g_L":12*nS,  "C":150*pF, "EL_lif":-67*mV, "V_th_lif":-50*mV, "V_res_lif":-65*mV}, "dt_ms":0.1},
        "stellate_lif":     {"params": {"g_L":10*nS,  "C":180*pF, "EL_lif":-68*mV, "V_th_lif":-52*mV, "V_res_lif":-66*mV}, "dt_ms":0.1},
        "deep_nuclei_lif":  {"params": {"g_L":15*nS,  "C":300*pF, "EL_lif":-65*mV, "V_th_lif":-50*mV, "V_res_lif":-66*mV}, "dt_ms":0.1},
        "mossy_fiber":      {"params": {"g_L":10*nS,  "C":150*pF, "EL_lif":-70*mV, "V_th_lif":-50*mV, "V_res_lif":-70*mV}, "dt_ms":0.1},
        "climbing_fiber":   {"params": {"g_L":8*nS,   "C":100*pF, "EL_lif":-65*mV, "V_th_lif":-45*mV, "V_res_lif":-68*mV}, "dt_ms":0.1},
        # Si tienes Purkinje HH, añade aquí su entrada y adapta sim_brian2_hh si la creas.
    }
    cells = list(CONFIG.keys())
    n_cells = len(cells)
    start_all = time.time()
    print("==== INICIANDO VALIDACIÓN COMPLETA ====")
for name in tqdm(cells, desc="Validando células", unit="cel"):
    cell_start = time.time()

    # 1) Generalización
    validar_generalizacion(name,
                                     CONFIG[name]["params"],
                                     CONFIG[name]["dt_ms"])
    # 2) Rendimiento
    validar_rendimiento(name,
                                      CONFIG[name]["params"],
                                      CONFIG[name]["dt_ms"])
    # 3) Interpretabilidad
    validar_interpretabilidad(name)

    elapsed = time.time() - cell_start
    # Si quieres el ETA manual, podrías calcularlo así:
    # times_per_cell.append(elapsed)
    # avg = sum(times_per_cell)/len(times_per_cell)
    # remaining = n_cells - (len(times_per_cell))
    # print(f"✔ {name} en {elapsed:.1f}s — ETA {avg*remaining:.1f}s")
    su_df = pd.DataFrame(speedup_records)
    out_speedup = os.path.join(BASE_MODEL_DIR, "summary_speedup.csv")
    su_df.to_csv(out_speedup, index=False)
    print(f"▶️ Speedup summary guardado en {out_speedup}")
    print("==== VALIDACIÓN FINALIZADA ====")
