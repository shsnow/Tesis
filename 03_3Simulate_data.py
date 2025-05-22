from brian2 import *
import pandas as pd
import numpy as np
import os

output_folder = "dataset"
os.makedirs(output_folder, exist_ok=True)

defaultclock.dt = 0.1*ms

def guardar_csv(t, v, I, spikes, nombre):
    spike_array = np.zeros_like(t)
    for st in spikes:
        spike_array[np.isclose(t, st, atol=0.05)] = 1
    df = pd.DataFrame({
        "time_ms": t,
        "voltage_mV": v,
        "input_current_nA": I,
        "spike": spike_array
    })
    df.to_csv(f"{output_folder}/{nombre}_kan_ready.csv", index=False)
    print(f"✅ Guardado {nombre}_kan_ready.csv")

def simular_adex(nombre, params, duracion=500*ms, n=100):
    start_scope()
    eqs = '''
    dv/dt = (-g_L*(v - EL) + g_L*Delta_T*exp((v - VT)/Delta_T) - w + I) / C : volt
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
    G = NeuronGroup(n, eqs, threshold='v > V_spike',
                    reset='v = V_reset; w += 0.2*nA', method='euler')
    for k, v in params.items():
        setattr(G, k, v)
    G.v = params["EL"]
    G.w = 0*pA
    G.I = 0*nA

    mon = StateMonitor(G, ['v', 'I'], record=True)
    spikes = SpikeMonitor(G)
    run(duracion)

    if len(spikes.i) == 0:
        print(f"❌ {nombre}: ninguna neurona disparó.")
        return
    idx = int(spikes.i[0])
    guardar_csv(mon.t/ms, mon.v[idx]/mV, mon.I[idx]/nA, spikes.spike_trains()[idx]/ms, nombre)

def simular_purkinje(duracion=500*ms, n=100):
    start_scope()
    eqs = '''
    dv/dt = (I_ext - gNa*(m**3)*h*(v - ENa) - gK*(n**4)*(v - EK) - gL*(v - EL)) / Cm : volt
    dm/dt = alpha_m*(1 - m) - beta_m*m : 1
    dh/dt = alpha_h*(1 - h) - beta_h*h : 1
    dn/dt = alpha_n*(1 - n) - beta_n*n : 1
    I_ext : amp/meter**2

    alpha_m = 0.1*(25 - v/mV) / (exp((25 - v/mV)/10) - 1)/ms : Hz
    beta_m = 4 * exp(-v / (18*mV)) / ms : Hz
    alpha_h = 0.07 * exp(-v / (20*mV)) / ms : Hz
    beta_h = 1 / (exp((30 - v/mV)/10) + 1)/ms : Hz
    alpha_n = 0.01 * (10 - v/mV)/(exp((10 - v/mV)/10) - 1)/ms : Hz
    beta_n = 0.125 * exp(-v / (80*mV)) / ms : Hz
    '''
    G = NeuronGroup(n, eqs, threshold='v > -40*mV', reset='v = -65*mV', method='exponential_euler',
                    namespace={
                        'Cm': 1*uF/cm**2, 'gNa': 120*msiemens/cm**2,
                        'gK': 36*msiemens/cm**2, 'gL': 0.3*msiemens/cm**2,
                        'ENa': 50*mV, 'EK': -77*mV, 'EL': -54.4*mV
                    })
    G.v = -65*mV
    G.m = 0.05
    G.h = 0.6
    G.n = 0.32
    G.I_ext = 10*uA/cm**2

    mon = StateMonitor(G, 'v', record=True)
    spikes = SpikeMonitor(G)
    run(duracion)
    idx = int(spikes.i[0]) if len(spikes.i) > 0 else 0
    I = np.ones_like(mon.t/ms) * (G.I_ext[0] * cm**2 / nA)
    guardar_csv(mon.t/ms, mon.v[idx]/mV, I, spikes.spike_trains()[idx]/ms, "purkinje")

def simular_poisson(nombre, rate, duracion=500*ms):
    start_scope()
    n = 1
    G = PoissonGroup(n, rates=rate)
    S = SpikeMonitor(G)
    run(duracion)

    t_array = np.arange(0, float(duracion/ms), float(defaultclock.dt/ms))
    I_array = np.random.normal(loc=0.5, scale=0.1, size=len(t_array))  # ficticio
    V_array = -70 + 5*np.sin(0.1*t_array) + np.random.normal(0, 0.2, len(t_array))  # ficticio
    guardar_csv(t_array, V_array, I_array, S.spike_trains()[0]/ms if len(S.i) > 0 else [], nombre)

def simular_todas():
    simular_adex("granule", {
        "g_L": 30*nS, "C": 200*pF, "EL": -70*mV, "VT": -50*mV, "Delta_T": 2*mV,
        "a": 4*nS, "tau_w": 100*ms, "I0": 1.0*nA, "sigma": 0.2*nA,
        "tau_syn": 5*ms, "V_reset": -65*mV, "V_spike": -40*mV
    })

    simular_adex("golgi", {
        "g_L": 20*nS, "C": 250*pF, "EL": -68*mV, "VT": -52*mV, "Delta_T": 1.8*mV,
        "a": 3*nS, "tau_w": 120*ms, "I0": 1.0*nA, "sigma": 0.2*nA,
        "tau_syn": 5*ms, "V_reset": -65*mV, "V_spike": -40*mV
    })

    simular_adex("basket", {
        "g_L": 25*nS, "C": 150*pF, "EL": -67*mV, "VT": -50*mV, "Delta_T": 2*mV,
        "a": 1*nS, "tau_w": 80*ms, "I0": 1.0*nA, "sigma": 0.2*nA,
        "tau_syn": 5*ms, "V_reset": -65*mV, "V_spike": -40*mV
    })

    simular_adex("stellate", {
        "g_L": 20*nS, "C": 180*pF, "EL": -68*mV, "VT": -52*mV, "Delta_T": 1.5*mV,
        "a": 2*nS, "tau_w": 90*ms, "I0": 0.8*nA, "sigma": 0.2*nA,
        "tau_syn": 5*ms, "V_reset": -66*mV, "V_spike": -40*mV
    })

    simular_adex("nuclei", {
        "g_L": 30*nS, "C": 300*pF, "EL": -65*mV, "VT": -50*mV, "Delta_T": 2*mV,
        "a": 3*nS, "tau_w": 120*ms, "I0": 1.1*nA, "sigma": 0.2*nA,
        "tau_syn": 5*ms, "V_reset": -66*mV, "V_spike": -40*mV
    })

    simular_poisson("mossy", rate=40*Hz)
    simular_poisson("climbing", rate=1*Hz)

    simular_purkinje()

if __name__ == "__main__":
    simular_todas()
