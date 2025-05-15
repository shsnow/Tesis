
from brian2 import *
import numpy as np
import pandas as pd
import os

prefs.codegen.target = 'numpy'
start_scope()
os.makedirs("cerebellar_datasets", exist_ok=True)

duration = 3000*ms
sample_interval = 1*ms
num_neurons = 1000

# Input Groups con monitoreo explícito
mossy = PoissonGroup(num_neurons, rates=30*Hz)
climbing = PoissonGroup(num_neurons, rates=2*Hz)

# Monitores desde el principio
mossy_mon = SpikeMonitor(mossy)
climbing_mon = SpikeMonitor(climbing)

# Granule Cells (AdEx)
granule_eqs = '''
dv/dt = ( -gL*(v - EL) + gL*DeltaT*exp((v - VT)/DeltaT) - w + I_syn + I_inj ) / C : volt
dw/dt = (a*(v - EL) - w)/tau_w : amp
I_syn : amp
I_inj : amp
'''
granule = NeuronGroup(num_neurons, granule_eqs, threshold='v > -50*mV',
                      reset='v = EL; w += b', method='euler',
                      namespace={
                          'C': 200*pF, 'gL': 10*nS, 'EL': -70*mV,
                          'VT': -50*mV, 'DeltaT': 2*mV, 'tau_w': 30*ms,
                          'a': 2*nS, 'b': 0.02*nA
                      })
granule.v = -70*mV
granule.w = 0*pA
granule.I_inj = '80*pA + 40*pA*rand()'
granule_mon = SpikeMonitor(granule)

# Golgi Cells (AdEx)
golgi = NeuronGroup(num_neurons, granule_eqs, threshold='v > -52*mV',
                    reset='v = EL; w += b', method='euler',
                    namespace={
                        'C': 200*pF, 'gL': 10*nS, 'EL': -65*mV,
                        'VT': -52*mV, 'DeltaT': 2*mV, 'tau_w': 100*ms,
                        'a': 4*nS, 'b': 0.1*nA
                    })
golgi.v = -65*mV
golgi.w = 0*pA
golgi.I_inj = '70*pA + 50*pA*rand()'
golgi_mon = SpikeMonitor(golgi)

# Basket Cells
basket = NeuronGroup(num_neurons, granule_eqs, threshold='v > -52*mV',
                     reset='v = EL; w += b', method='euler',
                     namespace={
                         'C': 150*pF, 'gL': 10*nS, 'EL': -65*mV,
                         'VT': -52*mV, 'DeltaT': 1*mV, 'tau_w': 10*ms,
                         'a': 0*nS, 'b': 0.05*nA
                     })
basket.v = -65*mV
basket.w = 0*pA
basket.I_inj = '70*pA + 30*pA*rand()'
basket_mon = SpikeMonitor(basket)

# Stellate Cells (Izhikevich corrected)
stellate_eqs = '''
dv/dt = (0.04*(v/mV)**2 * mV + 5*v + 140*mV - u*mV + I_syn/(1*nS)) / ms : volt
du/dt = a*(b*(v/mV) - u) : 1
I_syn : amp
a : 1/second
b : 1
'''
stellate = NeuronGroup(num_neurons, stellate_eqs, threshold='v > 30*mV',
                       reset='v = -65*mV; u += 8', method='euler',
                       namespace={'a': 0.02/ms, 'b': 0.2})
stellate.v = -65*mV
stellate.u = -14
stellate_mon = SpikeMonitor(stellate)

# Purkinje (HH)
purkinje_eqs = '''
dv/dt = (I - gNa*(m**3)*h*(v - ENa) - gK*(n**4)*(v - EK) - gL*(v - EL) + I_syn/cm**2)/Cm : volt
dm/dt = alpha_m*(1 - m) - beta_m*m : 1
dn/dt = alpha_n*(1 - n) - beta_n*n : 1
dh/dt = alpha_h*(1 - h) - beta_h*h : 1
I_syn : amp
I : amp/meter**2

alpha_m = (0.1*(25*mV - v)/mV)/(exp((25*mV - v)/(10*mV)) - 1)/ms : Hz
beta_m = 4*exp(-v/(18*mV))/ms : Hz
alpha_h = 0.07*exp(-v/(20*mV))/ms : Hz
beta_h = 1/(exp((30*mV - v)/(10*mV)) + 1)/ms : Hz
alpha_n = (0.01*(10*mV - v)/mV)/(exp((10*mV - v)/(10*mV)) - 1)/ms : Hz
beta_n = 0.125*exp(-v/(80*mV))/ms : Hz
'''
purkinje = NeuronGroup(num_neurons, purkinje_eqs, threshold='v > -40*mV',
                       reset='v = -65*mV', method='rk4',
                       namespace={
                           'Cm': 1*uF/cm**2, 'gNa': 120*msiemens/cm**2,
                           'gK': 36*msiemens/cm**2, 'gL': 0.3*msiemens/cm**2,
                           'ENa': 50*mV, 'EK': -77*mV, 'EL': -54.4*mV
                       })
purkinje.v = -65*mV
purkinje.m = 0.05
purkinje.h = 0.6
purkinje.n = 0.32
#purkinje.I = 10 * uA / cm**2  # Valor inicial para evitar warnings
purkinje.I = 20 * uA / cm**2

purkinje.run_regularly('I = (10 + 3*sin(2*pi*3*Hz*t)) * uA/cm**2', dt=defaultclock.dt)
purkinje_mon = SpikeMonitor(purkinje)

# Nuclei (AdEx)
nuclei = NeuronGroup(num_neurons, granule_eqs, threshold='v > -50*mV',
                     reset='v = EL; w += b', method='euler',
                     namespace={
                         'C': 250*pF, 'gL': 10*nS, 'EL': -68*mV,
                         'VT': -50*mV, 'DeltaT': 2*mV, 'tau_w': 100*ms,
                         'a': 0*nS, 'b': 0.15*nA
                     })
nuclei.v = -68*mV
nuclei.w = 0*pA
nuclei.I_inj = '100*pA + 60*pA*rand()'
nuclei_mon = SpikeMonitor(nuclei)

# Sinapsis nombradas
s1 = Synapses(mossy, granule, on_pre='I_syn += 200*pA'); s1.connect(j='i%num_neurons')
s2 = Synapses(mossy, nuclei, on_pre='I_syn += 200*pA'); s2.connect(j='i%num_neurons')
s3 = Synapses(granule, purkinje, on_pre='I_syn += 200*pA'); s3.connect(j='i%num_neurons')
s4 = Synapses(granule, golgi, on_pre='I_syn += 150*pA'); s4.connect(j='i%num_neurons')
s5 = Synapses(golgi, granule, on_pre='I_syn -= 200*pA'); s5.connect(j='i%num_neurons')
s6 = Synapses(granule, basket, on_pre='I_syn += 180*pA'); s6.connect(j='i%num_neurons')
s7 = Synapses(granule, stellate, on_pre='I_syn += 180*pA'); s7.connect(j='i%num_neurons')
s8 = Synapses(basket, purkinje, on_pre='I_syn -= 200*pA'); s8.connect(j='i%num_neurons')
s9 = Synapses(stellate, purkinje, on_pre='I_syn -= 200*pA'); s9.connect(j='i%num_neurons')
s10 = Synapses(climbing, purkinje, on_pre='I_syn += 250*pA'); s10.connect(j='i%num_neurons')
s11 = Synapses(purkinje, nuclei, on_pre='I_syn -= 300*pA'); s11.connect(j='i%num_neurons')

# Run simulation
run(duration)

# Save CSVs
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

# Save all
monitors = {
    'granule': granule_mon,
    'golgi': golgi_mon,
    'basket': basket_mon,
    'stellate': stellate_mon,
    'purkinje': purkinje_mon,
    'nuclei': nuclei_mon,
    'mossy': mossy_mon,
    'climbing': climbing_mon
}
for name, mon in monitors.items():
    save_spikes(mon, name)

print("✅ Simulación completada y datos guardados.")
