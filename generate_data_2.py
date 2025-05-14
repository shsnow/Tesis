from brian2 import *
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

prefs.codegen.target = 'numpy'
start_scope()
os.makedirs("cerebellar_datasets", exist_ok=True)

duration = 2000*ms
sample_interval = 1*ms
num_neurons = 1000  # por tipo


# Fibras musgosas
mossy = PoissonGroup(num_neurons, rates=20*Hz)
mossy_mon = SpikeMonitor(mossy)

# Fibras trepadoras
climbing = PoissonGroup(num_neurons, rates=1*Hz)
climbing_mon = SpikeMonitor(climbing)


granule_eqs = '''
dv/dt = ( -gL*(v - EL) + gL*DeltaT*exp((v - VT)/DeltaT) - w + I_syn ) / C : volt
dw/dt = (a*(v - EL) - w)/tau_w : amp
I_syn : amp
'''
granule = NeuronGroup(num_neurons, granule_eqs, threshold='v > -40*mV',
                      reset='v = EL; w += b', method='euler',
                      namespace={
                          'C': 200*pF, 'gL': 10*nS, 'EL': -70*mV,
                          'VT': -40*mV, 'DeltaT': 2*mV, 'tau_w': 30*ms,
                          'a': 2*nS, 'b': 0.02*nA
                      })
granule.v = -70*mV
granule.w = 0*pA
granule_mon = SpikeMonitor(granule)


golgi_eqs = granule_eqs  # Reutiliza AdEx

golgi = NeuronGroup(num_neurons, golgi_eqs, threshold='v > -40*mV',
                    reset='v = EL; w += b', method='euler',
                    namespace={
                        'C': 200*pF, 'gL': 10*nS, 'EL': -60*mV,
                        'VT': -50*mV, 'DeltaT': 2*mV, 'tau_w': 100*ms,
                        'a': 4*nS, 'b': 0.1*nA
                    })
golgi.v = -60*mV
golgi.w = 0*pA
golgi_mon = SpikeMonitor(golgi)


# Basket (AdEx rápido)
basket = NeuronGroup(num_neurons, granule_eqs, threshold='v > -40*mV',
                     reset='v = EL; w += b', method='euler',
                     namespace={
                         'C': 150*pF, 'gL': 10*nS, 'EL': -65*mV,
                         'VT': -52*mV, 'DeltaT': 0.5*mV, 'tau_w': 10*ms,
                         'a': 0*nS, 'b': 0.05*nA
                     })
basket.v = -65*mV
basket.w = 0*pA
basket_mon = SpikeMonitor(basket)

# Stellate (Izhikevich)
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
                       reset='v = -65*mV', method='exponential_euler',
                       namespace={
                           'Cm': 1*uF/cm**2, 'gNa': 120*msiemens/cm**2,
                           'gK': 36*msiemens/cm**2, 'gL': 0.3*msiemens/cm**2,
                           'ENa': 50*mV, 'EK': -77*mV, 'EL': -54.4*mV
                       })
purkinje.v = -65*mV
purkinje.m = 0.05
purkinje.h = 0.6
purkinje.n = 0.32
purkinje.I = '(10 + 3*sin(2*pi*3*Hz*t)) * uA/cm**2'

purkinje_mon = SpikeMonitor(purkinje)
purkinje_state = StateMonitor(purkinje, 'v', record=True, dt=sample_interval)


nuclei = NeuronGroup(num_neurons, granule_eqs, threshold='v > -40*mV',
                     reset='v = EL; w += b', method='euler',
                     namespace={
                         'C': 250*pF, 'gL': 10*nS, 'EL': -68*mV,
                         'VT': -50*mV, 'DeltaT': 2*mV, 'tau_w': 100*ms,
                         'a': 0*nS, 'b': 0.15*nA
                     })
nuclei.v = -68*mV
nuclei.w = 0*pA
nuclei_mon = SpikeMonitor(nuclei)
nuclei_state = StateMonitor(nuclei, 'v', record=True, dt=sample_interval)


#Synapses(mossy, granule, on_pre='I_syn += 100*pA').connect(j='i%num_neurons')
mossy_granule_syn = Synapses(mossy, granule, on_pre='I_syn += 100*pA')
mossy_granule_syn.connect(j='i%num_neurons')

#Synapses(mossy, nuclei, on_pre='I_syn += 150*pA').connect(j='i%num_neurons')
mossy_nuclei_syn = Synapses(mossy, nuclei, on_pre='I_syn += 150*pA')
mossy_nuclei_syn.connect(j='i%num_neurons')

#Synapses(granule, purkinje, on_pre='I_syn += 80*pA').connect(j='i%num_neurons')
granule_purkinje_syn = Synapses(granule, purkinje, on_pre='I_syn += 80*pA')
granule_purkinje_syn.connect(j='i%num_neurons')

#Synapses(granule, golgi, on_pre='I_syn += 80*pA').connect(j='i%num_neurons')
granule_golgi_syn = Synapses(granule, golgi, on_pre='I_syn += 80*pA')
granule_golgi_syn.connect(j='i%num_neurons')

#Synapses(golgi, granule, on_pre='I_syn -= 150*pA').connect(j='i%num_neurons')
golgi_granule_syn = Synapses(golgi, granule, on_pre='I_syn -= 150*pA')
golgi_granule_syn.connect(j='i%num_neurons')

#Synapses(granule, basket, on_pre='I_syn += 120*pA').connect(j='i%num_neurons')
granule_basket_syn = Synapses(granule, basket, on_pre='I_syn += 120*pA')
granule_basket_syn.connect(j='i%num_neurons')

#Synapses(granule, stellate, on_pre='I_syn += 120*pA').connect(j='i%num_neurons')
granule_stellate_syn = Synapses(granule, stellate, on_pre='I_syn += 120*pA')
granule_stellate_syn.connect(j='i%num_neurons')

#Synapses(basket, purkinje, on_pre='I_syn -= 120*pA').connect(j='i%num_neurons')
basket_purkinje_syn = Synapses(basket, purkinje, on_pre='I_syn -= 120*pA')
basket_purkinje_syn.connect(j='i%num_neurons')

#Synapses(stellate, purkinje, on_pre='I_syn -= 120*pA').connect(j='i%num_neurons')
stellate_purkinje_syn = Synapses(stellate, purkinje, on_pre='I_syn -= 120*pA')
stellate_purkinje_syn.connect(j='i%num_neurons')

#Synapses(climbing, purkinje, on_pre='I_syn += 200*pA').connect(j='i%num_neurons')
climbing_purkinje_syn = Synapses(climbing, purkinje, on_pre='I_syn += 200*pA')
climbing_purkinje_syn.connect(j='i%num_neurons')


#Synapses(purkinje, nuclei, on_pre='I_syn -= 300*pA').connect(j='i%num_neurons')
purkinje_nuclei_syn = Synapses(purkinje, nuclei, on_pre='I_syn -= 300*pA')
purkinje_nuclei_syn.connect(j='i%num_neurons')

run(duration)

def generate_csv(spikemon, statemon=None, name="neuron"):
    t = spikemon.t/ms
    i = spikemon.i
    df = pd.DataFrame({'neuron_id': i, 'spike_time': t})
    df_grouped = df.groupby('neuron_id')['spike_time'].apply(list)

    df_out = pd.DataFrame(index=range(num_neurons))
    # Solución robusta
    df_out['spike_times'] = df_grouped.reindex(df_out.index).apply(lambda x: x if isinstance(x, list) else [])

    # Características básicas
    df_out['spike_count'] = df_out['spike_times'].apply(len)
    df_out['mean_rate'] = df_out['spike_count'] / (duration/ms) * 1000

    if statemon is not None:
        vs = statemon.v / mV
        df_out['voltage_mean'] = np.mean(vs, axis=1)
        df_out['voltage_std'] = np.std(vs, axis=1)

    df_out.to_csv(f'cerebellar_datasets/kan_{name}_dataset.csv', index=False)

# Guardar todos
generate_csv(mossy_mon, name='mossy')
generate_csv(climbing_mon, name='climbing')
generate_csv(granule_mon, name='granule')
generate_csv(golgi_mon, name='golgi')
generate_csv(basket_mon, name='basket')
generate_csv(stellate_mon, name='stellate')
generate_csv(purkinje_mon, purkinje_state, name='purkinje')
generate_csv(nuclei_mon, nuclei_state, name='nuclei')


