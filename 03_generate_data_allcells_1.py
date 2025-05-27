# =======================================
# Simulador de neuronas cerebelares para KAN
# =======================================
from brian2 import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def guardar_csv(mon, spikes, nombre, duracion):
    t_array = mon.t / ms
    if len(spikes.i) == 0:
        print(f"❌ Ninguna neurona disparó en {nombre}.")
        return

    idx = int(spikes.i[0])
    v_array = mon.v[idx] / mV
    I_array = mon.I[idx] / nA
    spike_array = np.zeros_like(t_array)

    for st in spikes.spike_trains()[idx]/ms:
        spike_array[np.isclose(t_array, st, atol=0.05)] = 1

    df = pd.DataFrame({
        'time_ms': t_array,
        'voltage_mV': v_array,
        'input_current_nA': I_array,
        'spike': spike_array
    })
    df.to_csv(f"dataset/{nombre}_kan_ready.csv", index=False)
    print(f"✅ dataset/{nombre} guardado en {nombre}_kan_ready.csv")

def simular_adex(nombre, parametros, n_neuronas=100, duracion=1000*ms):
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

    G.v = parametros["EL"]; G.w = 0*pA; G.I = 0*nA
    mon = StateMonitor(G, ['v', 'I'], record=True)
    spikes = SpikeMonitor(G)
    run(duracion)
    guardar_csv(mon, spikes, nombre, duracion)


def simular_purkinje(n_neuronas=300, duracion=2000*ms):
    start_scope()
    defaultclock.dt = 0.01*ms

    eqs = '''
    dv/dt = (I_ext - gNa*(m**3)*h*(v - ENa) - gK*(n**4)*(v - EK) - gL*(v - EL)) / Cm : volt
    dm/dt = alpha_m*(1 - m) - beta_m*m : 1
    dh/dt = alpha_h*(1 - h) - beta_h*h : 1
    dn/dt = alpha_n*(1 - n) - beta_n*n : 1

    I_ext : amp/meter**2  # Corriente de entrada

    alpha_m = 0.1 * (25 - v/mV) / (exp((25 - v/mV) / 10) - 1) / ms : Hz
    beta_m = 4 * exp(-v / (18*mV)) / ms : Hz
    alpha_h = 0.07 * exp(-v / (20*mV)) / ms : Hz
    beta_h = 1 / (exp((30 - v/mV) / 10) + 1) / ms : Hz
    alpha_n = 0.01 * (10 - v/mV) / (exp((10 - v/mV) / 10) - 1) / ms : Hz
    beta_n = 0.125 * exp(-v / (80*mV)) / ms : Hz
    '''

    G = NeuronGroup(n_neuronas, eqs, threshold='v > -40*mV', reset='v = -65*mV',
                    method='exponential_euler',
                    namespace={
                        'Cm': 1*uF/cm**2, 'gNa': 120*msiemens/cm**2,
                        'gK': 36*msiemens/cm**2, 'gL': 0.3*msiemens/cm**2,
                        'ENa': 50*mV, 'EK': -77*mV, 'EL': -54.4*mV
                    })

    G.v = -65*mV; G.m = 0.05; G.h = 0.6; G.n = 0.32
    G.I_ext = 30*uA/cm**2

    mon = StateMonitor(G, variables=['v', 'I_ext'], record=True)
    spikes = SpikeMonitor(G)
    run(duracion)

    t_array = mon.t / ms
    idx = int(spikes.i[0]) if len(spikes.i) > 0 else 0
    v_array = mon.v[idx] / mV
    I_array = (mon.I_ext[idx] * cm**2) / nA  # Resultado en unidades de nA
    spike_array = np.zeros_like(t_array)

    for st in spikes.spike_trains()[idx]/ms:
        spike_array[np.isclose(t_array, st, atol=0.05)] = 1

    df = pd.DataFrame({
        'time_ms': t_array,
        'voltage_mV': v_array,
        'input_current_nA': I_array,
        'spike': spike_array
    })
    df.to_csv("dataset/purkinje_kan_ready.csv", index=False)
    print("✅ Purkinje guardado en dataset/purkinje_kan_ready.csv")


def simular_poisson(nombre, rate=50*Hz, duracion=1000*ms):
    start_scope()
    G = PoissonGroup(100, rates=rate)
    spikes = SpikeMonitor(G)
    run(duracion)

    t_array = np.arange(0, duracion/ms, defaultclock.dt/ms)
    spike_array = np.zeros_like(t_array)
    st = spikes.spike_trains()[0]/ms if 0 in spikes.spike_trains() else []
    for spike_time in st:
        spike_array[np.isclose(t_array, spike_time, atol=0.05)] = 1

    df = pd.DataFrame({
        'time_ms': t_array,
        'spike': spike_array
    })
    df.to_csv(f"dataset/{nombre}_kan_ready.csv", index=False)
    print(f"✅ {nombre} guardado en {nombre}_kan_ready.csv")


def simular_fibra(nombre="mossy", rate=40*Hz, n_neuronas=100, duracion=1000*ms):
    start_scope()
    defaultclock.dt = 0.1*ms

    eqs = '''
    dv/dt = (-g_L*(v - EL) + I)/C : volt
    dI/dt = (-I + I0)/tau_syn + (sigma*xi)/sqrt(tau_syn) : amp
    g_L : siemens (shared)
    C : farad (shared)
    EL : volt (shared)
    I0 : amp (shared)
    sigma : amp (shared)
    tau_syn : second (shared)
    V_reset : volt (shared)
    V_spike : volt (shared)
    '''

    G = NeuronGroup(n_neuronas, eqs,
                    threshold='v > V_spike',
                    reset='v = V_reset',
                    method='euler')

    G.g_L = 30*nS
    G.C = 200*pF
    G.EL = -70*mV
    G.I0 = 1.4*nA if nombre == "mossy" else 0.5*nA
    G.sigma = 0.2*nA
    G.tau_syn = 5*ms
    G.V_reset = -65*mV
    G.V_spike = -40*mV
    G.v = G.EL
    G.I = 0*nA

    mon = StateMonitor(G, ['v', 'I'], record=True)
    spikes = SpikeMonitor(G)
    run(duracion)

    t_array = mon.t / ms
    if len(spikes.i) == 0:
        print(f"❌ Ninguna neurona disparó en {nombre}.")
        return

    idx = int(spikes.i[0])
    v_array = mon.v[idx] / mV
    I_array = mon.I[idx] / nA
    spike_array = np.zeros_like(t_array)
    for st in spikes.spike_trains()[idx]/ms:
        spike_array[np.isclose(t_array, st, atol=0.05)] = 1

    df = pd.DataFrame({
        'time_ms': t_array,
        'voltage_mV': v_array,
        'input_current_nA': I_array,
        'spike': spike_array
    })
    df.to_csv(f"{nombre}_kan_ready.csv", index=False)
    print(f"✅ {nombre} guardado en {nombre}_kan_ready.csv")





def simular_todas():
    # Granulosa (AdEx)
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


    # Fibras musgosas (Poisson)
    simular_fibra("mossy", rate=40*Hz)

    # Fibras trepadoras (Poisson)
    simular_fibra("climbing", rate=1*Hz)

    # Purkinje (HH)
    simular_purkinje()



# Ejecutar todas
simular_todas()
#simular_purkinje()