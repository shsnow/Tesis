from brian2 import *
import numpy as np
import pandas as pd
import os

start_scope()
# prefs.codegen.target = 'numpy'  # NO USAR si da problemas

duration = 2000*ms
num_neurons = 1000

# Crear carpeta para guardar resultados
os.makedirs("cerebellar_datasets", exist_ok=True)

# Poisson inputs
mossy = PoissonGroup(num_neurons, rates=30*Hz)
climbing = PoissonGroup(num_neurons, rates=2*Hz)

# Leaky Integrate-and-Fire genérico con I_inj
base_eqs = '''
dv/dt = (-gL*(v - EL) + I_syn + I_inj)/C : volt
I_syn : amp
I_inj : amp
'''

def create_LIF_group(n, EL, inj_range=(100*pA, 200*pA), threshold='v > -50*mV'):
    group = NeuronGroup(n, base_eqs, threshold=threshold, reset='v = EL', method='euler',
                        namespace={'C': 200*pF, 'gL': 10*nS, 'EL': EL})
    group.v = EL
    group.I_inj = '({}*pA) + ({}*pA) * rand()'.format(
    float(inj_range[0]/pA),
    float((inj_range[1] - inj_range[0])/pA)
)
    return group

# Crear grupos neuronales
granule = create_LIF_group(num_neurons, EL=-70*mV)
golgi = create_LIF_group(num_neurons, EL=-65*mV)
basket = create_LIF_group(num_neurons, EL=-65*mV)
stellate = create_LIF_group(num_neurons, EL=-65*mV)
nuclei = create_LIF_group(num_neurons, EL=-68*mV)

# Purkinje especial (HH)
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

# Sinapsis
Synapses(mossy, granule, on_pre='I_syn += 200*pA').connect(j='i%num_neurons')
Synapses(mossy, nuclei, on_pre='I_syn += 200*pA').connect(j='i%num_neurons')
Synapses(granule, purkinje, on_pre='I_syn += 200*pA').connect(j='i%num_neurons')
Synapses(granule, golgi, on_pre='I_syn += 150*pA').connect(j='i%num_neurons')
Synapses(golgi, granule, on_pre='I_syn -= 200*pA').connect(j='i%num_neurons')
Synapses(granule, basket, on_pre='I_syn += 180*pA').connect(j='i%num_neurons')
Synapses(granule, stellate, on_pre='I_syn += 180*pA').connect(j='i%num_neurons')
Synapses(basket, purkinje, on_pre='I_syn -= 200*pA').connect(j='i%num_neurons')
Synapses(stellate, purkinje, on_pre='I_syn -= 200*pA').connect(j='i%num_neurons')
Synapses(climbing, purkinje, on_pre='I_syn += 250*pA').connect(j='i%num_neurons')
Synapses(purkinje, nuclei, on_pre='I_syn -= 300*pA').connect(j='i%num_neurons')

# Monitores
monitors = {
    'mossy': SpikeMonitor(mossy),
    'climbing': SpikeMonitor(climbing),
    'granule': SpikeMonitor(granule),
    'golgi': SpikeMonitor(golgi),
    'basket': SpikeMonitor(basket),
    'stellate': SpikeMonitor(stellate),
    'purkinje': SpikeMonitor(purkinje),
    'nuclei': SpikeMonitor(nuclei)
}

# Ejecutar simulación
run(duration)

# Guardar resultados
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

for name, mon in monitors.items():
    save_spikes(mon, name)
    print(f"{name}: Total spikes = {np.sum(mon.count)}")

print("✅ Simulación finalizada y archivos guardados.")
