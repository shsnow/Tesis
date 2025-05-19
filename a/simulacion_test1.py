from brian2 import *
import numpy as np
import pandas as pd
import os

prefs.codegen.target = 'numpy'
start_scope()
os.makedirs("cerebellar_datasets", exist_ok=True)

duration = 3000*ms
num_neurons = 100

# Poisson Inputs
mossy = PoissonGroup(num_neurons, rates=100*Hz)
climbing = PoissonGroup(num_neurons, rates=10*Hz)

mossy_mon = SpikeMonitor(mossy)
climbing_mon = SpikeMonitor(climbing)

# Granule Cell (AdEx)
adex_eqs = '''
dv/dt = (-gL*(v - EL) + gL*DeltaT*exp((v - VT)/DeltaT) - w + I_syn + I_inj)/C : volt
dw/dt = (a*(v - EL) - w)/tau_w : amp
I_syn : amp
I_inj : amp
'''
granule = NeuronGroup(num_neurons, adex_eqs, threshold='v > -50*mV',
    reset='v = EL; w += b', method='euler',
    namespace=dict(C=200*pF, gL=10*nS, EL=-70*mV, VT=-50*mV,
                   DeltaT=2*mV, tau_w=30*ms, a=2*nS, b=0.05*nA))
granule.v = -70*mV
granule.w = 0*pA
granule.I_inj = 150*pA
granule_mon = SpikeMonitor(granule)

# Purkinje Cell (HH con corrección de unidades)
hh_eqs = '''
dv/dt = (I - INa - IK - IL + I_syn/(200*umetre**2)) / Cm : volt
INa = gNa * m**3 * h * (ENa - v) : amp/meter**2
IK = gK * n**4 * (EK - v) : amp/meter**2
IL = gL * (EL - v) : amp/meter**2
dm/dt = alpha_m*(1 - m) - beta_m*m : 1
dn/dt = alpha_n*(1 - n) - beta_n*n : 1
dh/dt = alpha_h*(1 - h) - beta_h*h : 1
I_syn : amp
I : amp/meter**2
alpha_m = 0.1/mV*(25*mV - v)/(exp((25*mV - v)/(10*mV)) - 1)/ms : Hz
beta_m = 4*exp(-v/(18*mV))/ms : Hz
alpha_h = 0.07*exp(-v/(20*mV))/ms : Hz
beta_h = 1/(exp((30*mV - v)/(10*mV)) + 1)/ms : Hz
alpha_n = 0.01/mV*(10*mV - v)/(exp((10*mV - v)/(10*mV)) - 1)/ms : Hz
beta_n = 0.125*exp(-v/(80*mV))/ms : Hz
'''
purkinje = NeuronGroup(num_neurons, hh_eqs, threshold='v > -45*mV',
    reset='v = -65*mV', method='exponential_euler',
    namespace=dict(Cm=1*uF/cm**2, gNa=120*msiemens/cm**2,
                   gK=36*msiemens/cm**2, gL=0.3*msiemens/cm**2,
                   ENa=50*mV, EK=-77*mV, EL=-54.4*mV))
purkinje.v = -65*mV
purkinje.m = 0.05
purkinje.h = 0.6
purkinje.n = 0.32
purkinje.I = 25 * uA / cm**2
purkinje_mon = SpikeMonitor(purkinje)

# Sinapsis (conectadas y guardadas)
connect_p = 0.1
s1 = Synapses(mossy, granule, on_pre='I_syn += 600*pA'); s1.connect(p=connect_p)
s2 = Synapses(granule, purkinje, on_pre='I_syn += 800*pA'); s2.connect(p=connect_p)
s3 = Synapses(climbing, purkinje, on_pre='I_syn += 1000*pA'); s3.connect(p=connect_p)

# Simulación
run(duration)

# Guardar CSV
def save_spikes(spikemon, name):
    t = spikemon.t/ms
    i = spikemon.i
    df = pd.DataFrame({'neuron_id': i, 'spike_time': t})
    df_grouped = df.groupby('neuron_id')['spike_time'].apply(list)
    df_out = pd.DataFrame(index=range(num_neurons))
    df_out['spike_times'] = df_grouped.reindex(df_out.index).apply(lambda x: x if isinstance(x, list) else [])
    df_out['spike_count'] = df_out['spike_times'].apply(len)
    df_out['mean_rate'] = df_out['spike_count'] / (duration/ms) * 1000
    df_out.to_csv(f'cerebellar_datasets/kan_{name}_dataset.csv', index=False)

save_spikes(granule_mon, 'granule')
save_spikes(purkinje_mon, 'purkinje')
save_spikes(mossy_mon, 'mossy')
save_spikes(climbing_mon, 'climbing')

print("✅ Datos generados y guardados correctamente.")
