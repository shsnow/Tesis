#!/usr/bin/env python3
# generate_results.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from brian2 import (
    prefs, start_scope, defaultclock, ms, mV, nA,nS, pF, set_device,
    TimedArray, NeuronGroup, StateMonitor, SpikeMonitor, run
)
import torch
from Cerebellar_Class import NeuronaCerebelarKAN, BASE_MODEL_DIR, DEVICE

# Asegurarse de usar numpy codegen en Brian2
prefs.codegen.target = 'numpy'
os.makedirs(BASE_MODEL_DIR, exist_ok=True)
set_device('runtime', build_on_run=False)  # limpia cualquier build previo

# --- Configuración de pruebas (mismos parámetros que validate_cells_final.py) ---
CELL_CONFIG = {
    "granule_lif":      {"params": {"g_L":5*nS,   "C":100*pF, "EL_lif":-70*mV, "V_th_lif":-50*mV, "V_res_lif":-65*mV}, "dt_ms":0.1},
    "golgi_lif":        {"params": {"g_L":10*nS,  "C":200*pF, "EL_lif":-68*mV, "V_th_lif":-52*mV, "V_res_lif":-65*mV}, "dt_ms":0.1},
    "basket_lif":       {"params": {"g_L":12*nS,  "C":150*pF, "EL_lif":-67*mV, "V_th_lif":-50*mV, "V_res_lif":-65*mV}, "dt_ms":0.1},
    "stellate_lif":     {"params": {"g_L":10*nS,  "C":180*pF, "EL_lif":-68*mV, "V_th_lif":-52*mV, "V_res_lif":-66*mV}, "dt_ms":0.1},
    "deep_nuclei_lif":  {"params": {"g_L":15*nS,  "C":300*pF, "EL_lif":-65*mV, "V_th_lif":-50*mV, "V_res_lif":-66*mV}, "dt_ms":0.1},
    "mossy_fiber":      {"params": {"g_L":10*nS,  "C":150*pF, "EL_lif":-70*mV, "V_th_lif":-50*mV, "V_res_lif":-70*mV}, "dt_ms":0.1},
    "climbing_fiber":   {"params": {"g_L":8*nS,   "C":100*pF, "EL_lif":-65*mV, "V_th_lif":-45*mV, "V_res_lif":-68*mV}, "dt_ms":0.1},
}

# --- Funciones comunes ---
def sim_brian2_lif(params, corriente, T_ms, dt_ms):
    """Simula un LIF en Brian2 y devuelve t (ms), v (mV), spikes (ms)."""
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
    t      = mon_v.t/ms
    v      = mon_v.v[0]/mV
    spikes = mon_sp.t/ms
    return t, v, spikes

def sim_kan_closed_loop(neur, params, corriente, t_ms, dt_ms):
    """Simula en bucle cerrado la KAN: la predicción en t alimenta t+1."""
    pasos = len(t_ms)
    v_kan = np.zeros(pasos)
    v_kan[0] = params['EL_lif']/mV
    sp_kan = []
    for i in range(pasos-1):
        df = pd.DataFrame({
            'time_ms': [t_ms[i]],
            'voltage_mV': [v_kan[i]],
            'input_current_nA': [corriente[i]]
        })
        _, pred = neur.predecir(df)
        if pred is not None and pred[0][0]==1:
            v_kan[i+1] = params['V_res_lif']/mV
            sp_kan.append(t_ms[i])
        else:
            dv = ((-params['g_L']*(v_kan[i]*mV-params['EL_lif']))+
                  corriente[i]*nA)/params['C']
            v_kan[i+1] = (v_kan[i]*mV + dv*(dt_ms*ms))/mV
    return v_kan, np.array(sp_kan)

def generar_estimulos_prueba(T_ms, dt_ms):
    """Genera cuatro estímulos: pulso, rampa, sinusoidal y pulsos cortos."""
    pasos = int(T_ms/dt_ms)
    t = np.arange(pasos)*dt_ms
    pulso = np.zeros(pasos); pulso[int(100/dt_ms):int(400/dt_ms)] = 0.6
    rampa = np.linspace(0,0.8,pasos)
    sinuso = 0.4 + 0.3*np.sin(2*np.pi*5*t/1000)
    cortos = np.zeros(pasos)
    for s in [100,200,300,400]:
        cortos[int(s/dt_ms):int((s+20)/dt_ms)] = 0.9
    return {"Pulso":pulso, "Rampa":rampa, "Sinusoidal":sinuso, "Cortos":cortos}, t

# --- Generación de figuras ---
if __name__=="__main__":
    for cell, cfg in CELL_CONFIG.items():
        params = cfg["params"]; dt=cfg["dt_ms"]
        print(f"\n🔍 Procesando '{cell}'")

        # Cargar modelo KAN
        neur = NeuronaCerebelarKAN(cell)
        if not neur.cargar_modelo(): 
            continue

        # --- 1) CLOSED-LOOP EMULATION ---
        print(" • Generando closed-loop emulation plot …")
        # Usamos estímulo sinusoidal para el closed-loop
        estim, t = generar_estimulos_prueba(500, dt)
        sin = estim["Sinusoidal"]
        t_gt, v_gt, sp_gt = sim_brian2_lif(params, sin, 500, dt)
        v_kan, sp_kan  = sim_kan_closed_loop(neur, params, sin, t, dt)

        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(t, v_gt,    label="Brian2", linewidth=1)
        ax.plot(t, v_kan, '--', label="KAN",    linewidth=1)
        # marcar spikes
        if len(sp_gt)>0:
            y0 = np.min(v_gt) - 5
            ys = np.full_like(sp_gt, y0)          # array de la misma longitud que sp_gt
            ax.plot(sp_gt, ys, '|', color='blue', label='spikes GT')        
        if len(sp_kan)>0:
            y1 = np.min(v_gt) - 8
            ys1 = np.full_like(sp_kan, y1)
            ax.plot(sp_kan, ys1, 'x', color='orange', label='spikes KAN')

        ax.set_title(f"Closed-loop: {cell}")
        ax.set_xlabel("Time (ms)"); ax.set_ylabel("Voltage (mV)")
        ax.legend(loc='upper right'); ax.grid(True)
        out1 = os.path.join(BASE_MODEL_DIR, f"closed_loop_{cell}.png")
        fig.savefig(out1, dpi=300, bbox_inches='tight'); plt.close(fig)
        print("   →", out1)

        # --- 2) GENERALIZATION ---
        print(" • Generando generalization plot …")
        estim, t = generar_estimulos_prueba(500, dt)
        fig, axes = plt.subplots(len(estim), 1, figsize=(6,2.5*len(estim)), sharex=True)
        for ax, (name_signal, sig) in zip(axes, estim.items()):
            # Brian2 vs KAN
            t_gt, v_gt, sp_gt = sim_brian2_lif(params, sig, 500, dt)
            v_kan, sp_kan    = sim_kan_closed_loop(neur, params, sig, t, dt)
            ax.plot(t, v_gt,    label="Brian2")
            ax.plot(t, v_kan, '--', label="KAN")
            # marcar spikes de Brian2
            if len(sp_gt) > 0:
                y0  = np.min(v_gt) - 5
                ys0 = np.full_like(sp_gt, y0)    # array con un valor repetido
                ax.plot(sp_gt, ys0, '|', color='blue')
            # marcar spikes de KAN
            if len(sp_kan) > 0:
                y1  = np.min(v_gt) - 8
                ys1 = np.full_like(sp_kan, y1)
                ax.plot(sp_kan, ys1, 'x', color='orange')
            ax.set_ylabel(name_signal)
            ax.grid(True)
        axes[-1].set_xlabel("Time (ms)")
        fig.suptitle(f"Generalization: {cell}", y=1.02)
        fig.legend(["Brian2","KAN"], loc='lower center', ncol=2)
        plt.tight_layout()
        out2 = os.path.join(BASE_MODEL_DIR, f"generalization_{cell}.png")
        fig.savefig(out2, dpi=300, bbox_inches='tight'); plt.close(fig)
        print("   →", out2)

    print("\n✅ Todas las figuras generadas en", BASE_MODEL_DIR)
