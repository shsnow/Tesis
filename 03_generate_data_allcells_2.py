from brian2 import *
import numpy as np
import pandas as pd
import os
from sklearn.utils import resample

prefs.codegen.target = "numpy"
os.makedirs("dataset", exist_ok=True)

def guardar_csv(mon, spikes, nombre, duracion, max_neuronas=10):
    t_array = mon.t / ms
    if len(spikes.i) == 0:
        print(f"❌ Ninguna neurona disparó en {nombre}.")
        return

    disparadoras = [i for i in np.unique(spikes.i) if len(spikes.spike_trains()[i]) > 2][:max_neuronas]
    if len(disparadoras) == 0:
        print(f"⚠️ Neuronas disparadoras insuficientes en {nombre}. Skipping...")
        return

    dfs = []
    for idx in disparadoras:
        v_array = mon.v[idx] / mV
        I_array = mon.I[idx] / nA if hasattr(mon, 'I') else np.full_like(v_array, 1.0)
        spike_array = np.zeros_like(t_array)
        for st in spikes.spike_trains()[idx]/ms:
            spike_array[np.isclose(t_array, st, atol=0.05)] = 1

        df = pd.DataFrame({
            'time_ms': t_array,
            'voltage_mV': v_array,
            'input_current_nA': I_array,
            'spike': spike_array
        })
        dfs.append(df)

    df_final = pd.concat(dfs, ignore_index=True)

    # Balancear clase 0 y 1 con 1:1
    df_spike = df_final[df_final['spike'] == 1]
    df_nospike = df_final[df_final['spike'] == 0]
    if len(df_spike) == 0:
        print(f"⚠️ Ningún spike útil en {nombre}. Skipping...")
        return

    # Oversample spike (clase 1) y undersample clase 0
    df_spike_upsampled = resample(df_spike, replace=True, n_samples=len(df_nospike), random_state=42)
    df_bal = pd.concat([df_spike_upsampled, df_nospike]).sample(frac=1).reset_index(drop=True)

    df_bal.to_csv(f"dataset/{nombre}_kan_ready.csv", index=False)
    print(f"✅ dataset/{nombre} guardado en {nombre}_kan_ready.csv con balance aproximado 1:1")

def simular_adex(nombre, duracion=1000, N=100, I0=1.0, gL=10*nS, C=200*pF, EL=-70*mV):
    eqs = '''
    dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) - w + I) / C : volt
    dw/dt = (a*(v - EL) - w) / tau_w : amp
    I : amp
    '''
    G = NeuronGroup(N, eqs, threshold='v > -40*mV', reset='v = EL; w += b', method='euler')
    G.v = EL
    G.w = 0*pA
    G.I = I0 * nA

    G.namespace.update({
        'C': C, 'gL': gL, 'EL': EL, 'VT': -50*mV, 'DeltaT': 2*mV,
        'a': 2*nS, 'tau_w': 100*ms, 'b': 0.02*nA
    })

    mon = StateMonitor(G, ['v', 'I'], record=True)
    spikes = SpikeMonitor(G)
    run(duracion * ms)
    guardar_csv(mon, spikes, nombre, duracion)

def simular_izhikevich(nombre, duracion=1000, N=100, I0=10.0):
    eqs = '''
    dv/dt = (0.04*v**2 + 5*v + 140 - u + I)/ms : 1
    du/dt = a*(b*v - u)/ms : 1
    I : 1
    '''
    G = NeuronGroup(N, eqs, threshold='v > 30', reset='v = c; u += d', method='euler')
    G.v = -65
    G.u = G.v * 0
    G.I = I0

    G.namespace.update({'a': 0.02, 'b': 0.2, 'c': -65, 'd': 6})

    mon = StateMonitor(G, ['v', 'I'], record=True)
    spikes = SpikeMonitor(G)
    run(duracion * ms)
    guardar_csv(mon, spikes, nombre, duracion)

def simular_hodgkin_huxley(nombre, duracion=1000, N=100, I0=20.0):
    eqs = '''
    dv/dt = (I - gNa*(m**3)*h*(v - ENa) - gK*(n**4)*(v - EK) - gL*(v - EL)) / Cm : volt
    dm/dt = alpha_m*(1 - m) - beta_m*m : 1
    dh/dt = alpha_h*(1 - h) - beta_h*h : 1
    dn/dt = alpha_n*(1 - n) - beta_n*n : 1
    I : amp
    alpha_m = 0.1*(25 - v/mV)/ (exp((25 - v/mV)/10) - 1)/ms : Hz
    beta_m = 4*exp(-v/(18*mV))/ms : Hz
    alpha_h = 0.07*exp(-v/(20*mV))/ms : Hz
    beta_h = 1 / (exp((30 - v/mV)/10) + 1)/ms : Hz
    alpha_n = 0.01*(10 - v/mV)/ (exp((10 - v/mV)/10) - 1)/ms : Hz
    beta_n = 0.125*exp(-v/(80*mV))/ms : Hz
    '''
    G = NeuronGroup(N, eqs, method='exponential_euler', threshold='v > -40*mV', reset='v = -65*mV')
    G.v = -65*mV; G.m = 0.05; G.h = 0.6; G.n = 0.32
    G.I = I0 * namp

    G.namespace.update({
        'Cm': 1*uF/cm**2, 'gNa': 120*msiemens/cm**2, 'gK': 36*msiemens/cm**2,
        'gL': 0.3*msiemens/cm**2, 'ENa': 50*mV, 'EK': -77*mV, 'EL': -54.4*mV
    })

    mon = StateMonitor(G, ['v', 'I'], record=True)
    spikes = SpikeMonitor(G)
    run(duracion * ms)
    guardar_csv(mon, spikes, nombre, duracion)

def simular_fibra(nombre, duracion=1000, N=100, rate=40):
    G = PoissonGroup(N, rates=rate * Hz)
    mon = StateMonitor(G, 'rates', record=True)
    spikes = SpikeMonitor(G)
    run(duracion * ms)

    t_array = mon.t / ms
    dfs = []
    for idx in range(min(N, 10)):
        spike_array = np.zeros_like(t_array)
        for st in spikes.spike_trains()[idx]/ms:
            spike_array[np.isclose(t_array, st, atol=0.05)] = 1
        df = pd.DataFrame({
            'time_ms': t_array,
            'voltage_mV': np.zeros_like(t_array),
            'input_current_nA': np.ones_like(t_array),
            'spike': spike_array
        })
        dfs.append(df)
    df_final = pd.concat(dfs, ignore_index=True)

    df_spike = df_final[df_final['spike'] == 1]
    df_nospike = df_final[df_final['spike'] == 0]
    if len(df_spike) == 0:
        print(f"⚠️ Ningún spike útil en {nombre}. Skipping...")
        return

    df_spike_upsampled = resample(df_spike, replace=True, n_samples=len(df_nospike), random_state=42)
    df_bal = pd.concat([df_spike_upsampled, df_nospike]).sample(frac=1).reset_index(drop=True)

    df_bal.to_csv(f"dataset/{nombre}_kan_ready.csv", index=False)
    print(f"✅ dataset/{nombre} guardado en {nombre}_kan_ready.csv con balance aproximado 1:1")

def main():
    #simular_adex("basket", duracion=2000, I0=1.0)
    #simular_adex("granule", duracion=2000, I0=1.0)
    #simular_izhikevich("stellate", duracion=2000, I0=10.0)
    #simular_adex("golgi", duracion=2000, I0=1.2)
    simular_hodgkin_huxley("purkinje", duracion=2000, I0=20.0)
    #simular_fibra("climbing", duracion=2000, rate=20)
    #simular_fibra("mossy", duracion=2000, rate=40)
    #simular_adex("nuclei", duracion=2000, I0=1.4)

if __name__ == "__main__":
    main()
