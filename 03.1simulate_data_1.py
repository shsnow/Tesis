# ===============================
# 1. Archivo: 03_generate_data_allcells_1.py
# ===============================
from brian2 import *
import pandas as pd
import numpy as np
import os

def guardar_csv(mon, spikes, nombre, incluir_corriente=True):
    t_array = mon.t / ms
    idx = int(spikes.i[0]) if len(spikes.i) > 0 else 0
    v_array = mon.v[idx] / mV
    spike_array = np.zeros_like(t_array)

    for st in spikes.spike_trains()[idx]/ms:
        spike_array[np.isclose(t_array, st, atol=0.05)] = 1

    if incluir_corriente:
        I_array = mon.I[idx] / nA
        df = pd.DataFrame({
            'time_ms': t_array,
            'voltage_mV': v_array,
            'input_current_nA': I_array,
            'spike': spike_array
        })
    else:
        df = pd.DataFrame({
            'time_ms': t_array,
            'voltage_mV': v_array,
            'spike': spike_array
        })

    os.makedirs("dataset", exist_ok=True)
    df.to_csv(f"dataset/{nombre}_kan_ready.csv", index=False)
    print(f"✅ {nombre} guardado en dataset/{nombre}_kan_ready.csv")

def simular_adex(nombre, parametros, n_neuronas=300, duracion=1000*ms):
    start_scope()
    defaultclock.dt = 0.1*ms

    eqs = '''
    dv/dt = (-g_L*(v - EL) + g_L*Delta_T*exp((v - VT)/Delta_T) - w + I)/C : volt
    dw/dt = (a*(v - EL) - w)/tau_w : amp
    dI/dt = (-I + I0)/tau_syn + (sigma*xi)/sqrt(tau_syn) : amp
    g_L : siemens (shared)
    C : farad (shared)
    EL : volt (shared)
    VT : volt (shared)
    Delta_T : volt (shared)
    a : siemens (shared)
    tau_w : second (shared)
    I0 : amp (shared)
    sigma : amp (shared)
    tau_syn : second (shared)
    V_reset : volt (shared)
    V_spike : volt (shared)
    '''

    G = NeuronGroup(n_neuronas, eqs, threshold='v > V_spike',
                    reset='v = V_reset; w += 0.2*nA', method='euler')
    for param, value in parametros.items():
        setattr(G, param, value)

    G.v = parametros["EL"]
    G.w = 0*pA
    G.I = 0*nA

    mon = StateMonitor(G, ['v', 'I'], record=True)
    spikes = SpikeMonitor(G)
    run(duracion)
    guardar_csv(mon, spikes, nombre)

def simular_poisson(nombre, rate=20*Hz, duracion=1000*ms):
    start_scope()
    defaultclock.dt = 0.1*ms

    eqs = '''
    dv/dt = (I - g_L*(v - EL))/C : volt
    dI/dt = (-I + I0)/tau_syn + (sigma*xi)/sqrt(tau_syn) : amp
    g_L : siemens (shared)
    C : farad (shared)
    EL : volt (shared)
    I0 : amp (shared)
    sigma : amp (shared)
    tau_syn : second (shared)
    '''

    G = NeuronGroup(300, eqs, threshold='v > -40*mV', reset='v = -65*mV', method='euler')
    G.g_L = 30*nS
    G.C = 200*pF
    G.EL = -70*mV
    G.I0 = 1.0*nA
    G.sigma = 0.2*nA
    G.tau_syn = 5*ms

    G.v = -70*mV
    G.I = 0*nA

    mon = StateMonitor(G, ['v', 'I'], record=True)
    spikes = SpikeMonitor(G)
    run(duracion)
    guardar_csv(mon, spikes, nombre)

def simular_todas():
    print("=== Simulando células cerebelares ===")
    simular_adex("granule", {
        "g_L": 30*nS, "C": 200*pF, "EL": -70*mV, "VT": -50*mV,
        "Delta_T": 2*mV, "a": 4*nS, "tau_w": 100*ms,
        "I0": 1.0*nA, "sigma": 0.2*nA, "tau_syn": 5*ms,
        "V_reset": -65*mV, "V_spike": -40*mV
    })

    # Golgi (AdEx)
    simular_adex("golgi", {
        "g_L": 20*nS, "C": 250*pF, "EL": -68*mV, "VT": -52*mV,
        "Delta_T": 1.8*mV, "a": 3*nS, "tau_w": 120*ms,
        "I0": 1.0*nA, "sigma": 0.2*nA, "tau_syn": 5*ms,
        "V_reset": -65*mV, "V_spike": -40*mV
    })

    # Basket (AdEx)
    simular_adex("basket", {
        "g_L": 25*nS, "C": 150*pF, "EL": -67*mV, "VT": -50*mV,
        "Delta_T": 2*mV, "a": 1*nS, "tau_w": 80*ms,
        "I0": 1.0*nA, "sigma": 0.2*nA, "tau_syn": 5*ms,
        "V_reset": -65*mV, "V_spike": -40*mV
    })

    # Stellate (AdEx)
    simular_adex("stellate", {
        "g_L": 20*nS, "C": 180*pF, "EL": -68*mV, "VT": -52*mV,
        "Delta_T": 1.5*mV, "a": 2*nS, "tau_w": 90*ms,
        "I0": 0.8*nA, "sigma": 0.2*nA, "tau_syn": 5*ms,
        "V_reset": -66*mV, "V_spike": -40*mV
    })

    # Núcleo profundo (AdEx)
    simular_adex("nuclei", {
        "g_L": 30*nS, "C": 300*pF, "EL": -65*mV, "VT": -50*mV,
        "Delta_T": 2*mV, "a": 3*nS, "tau_w": 120*ms,
        "I0": 1.1*nA, "sigma": 0.2*nA, "tau_syn": 5*ms,
        "V_reset": -66*mV, "V_spike": -40*mV
    })



    simular_poisson("mossy")
    simular_poisson("climbing")

simular_todas()
