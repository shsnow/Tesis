# =========================
# 03.1simulate_data_1.py
# Generador mejorado para Purkinje
# =========================

from brian2 import *
import pandas as pd
import numpy as np

def simular_purkinje_mejorado(n_neuronas=200, duracion=1000*ms):
    start_scope()
    defaultclock.dt = 0.01*ms

    eqs = '''
    dv/dt = (I_ext - gNa*(m**3)*h*(v - ENa) - gK*(n**4)*(v - EK) - gL*(v - EL)) / Cm : volt
    dm/dt = alpha_m*(1 - m) - beta_m*m : 1
    dh/dt = alpha_h*(1 - h) - beta_h*h : 1
    dn/dt = alpha_n*(1 - n) - beta_n*n : 1

    I_ext : amp/meter**2

    alpha_m = 0.1 * (25 - v/mV) / (exp((25 - v/mV) / 10) - 1) / ms : Hz
    beta_m = 4 * exp(-v / (18*mV)) / ms : Hz

    alpha_h = 0.07 * exp(-v / (20*mV)) / ms : Hz
    beta_h = 1 / (exp((30 - v/mV) / 10) + 1) / ms : Hz

    alpha_n = 0.01 * (10 - v/mV) / (exp((10 - v/mV) / 10) - 1) / ms : Hz
    beta_n = 0.125 * exp(-v / (80*mV)) / ms : Hz
    '''

    G = NeuronGroup(n_neuronas, eqs, threshold='v > -40*mV', reset='v = -65*mV', method='exponential_euler',
                    namespace={
                        'Cm': 1*uF/cm**2, 'gNa': 120*msiemens/cm**2,
                        'gK': 36*msiemens/cm**2, 'gL': 0.3*msiemens/cm**2,
                        'ENa': 50*mV, 'EK': -77*mV, 'EL': -54.4*mV
                    })

    G.v = -65*mV
    G.m = 0.05
    G.h = 0.6
    G.n = 0.32
    G.I_ext = 20*uA/cm**2

    mon = StateMonitor(G, 'v', record=True)
    spikes = SpikeMonitor(G)

    run(duracion)

    idx = int(spikes.i[0]) if len(spikes.i) > 0 else 0
    t_array = mon.t / ms
    v_array = mon.v[idx] / mV
    I_array = np.ones_like(t_array) * float(G.I_ext[0] * cm**2 / nA)  # constante

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
    print("✅ Nuevo Purkinje guardado en dataset/purkinje_kan_ready.csv")

if __name__ == "__main__":
    simular_purkinje_mejorado()
