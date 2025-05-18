from brian2 import *
import pandas as pd
import numpy as np

def simular_granulosas(n_neuronas=100, duracion=500*ms, guardar=True):
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

    parametros = {
        "g_L": 30*nS,
        "C": 200*pF,
        "EL": -70*mV,
        "VT": -50*mV,
        "Delta_T": 2*mV,
        "a": 4*nS,
        "tau_w": 100*ms,
        "I0": 1.0*nA,
        "sigma": 0.2*nA,
        "tau_syn": 5*ms,
        "V_reset": -65*mV,
        "V_spike": -40*mV
    }

    G = NeuronGroup(n_neuronas, model=eqs,
                    threshold='v > V_spike',
                    reset='v = V_reset; w += 0.2*nA',
                    method='euler')

    for param, value in parametros.items():
        setattr(G, param, value)

    G.v = parametros["EL"]
    G.w = 0*pA
    G.I = 0*nA

    mon_v = StateMonitor(G, ['v', 'I'], record=True)
    spikes = SpikeMonitor(G)

    run(duracion)

    if guardar:
        t_array = mon_v.t / ms
        v_array = mon_v.v[0] / mV
        I_array = mon_v.I[0] / nA
        spike_times = spikes.spike_trains()[0] / ms
        spike_binary = np.isin(t_array, spike_times).astype(int)

        df = pd.DataFrame({
            'time_ms': t_array,
            'voltage_mV': v_array,
            'input_current_nA': I_array,
            'spike': spike_binary
        })

        df.to_csv("granule_kan_ready.csv", index=False)
        print("✅ Datos guardados en granule_kan_ready.csv")

    plt.plot(t_array, v_array)
    plt.title("Voltaje de membrana de la célula 0")
    plt.xlabel("Tiempo (ms)")
    plt.ylabel("Voltaje (mV)")
    plt.grid()
    plt.show()

    print("Simulación exitosa!")

simular_granulosas()

