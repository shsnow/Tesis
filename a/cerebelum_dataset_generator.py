# Guarda este script como "cerebellum_simulator.py"
# Asegúrate de tener instalados: brian2, pandas, numpy

from brian2 import *
import pandas as pd
import os

prefs.codegen.target = 'numpy'
start_scope()
os.makedirs("cerebellar_datasets", exist_ok=True)

duration = 3000*ms
num_neurons = 100
connect_p = 0.1  # conectividad

# Poisson Inputs
mossy = PoissonGroup(num_neurons, rates=150*Hz)
climbing = PoissonGroup(num_neurons, rates=15*Hz)

# Monitores
mossy_mon = SpikeMonitor(mossy)
climbing_mon = SpikeMonitor(climbing)

# -------- AdEx común --------
adex_eqs = '''
dv/dt = (-gL*(v - EL) + gL*DeltaT*exp((v - VT)/DeltaT) - w + I_syn + I_inj)/C : volt
dw/dt = (a*(v - EL) - w)/tau_w : amp
I_syn : amp
I_inj : amp
'''

# Granule
granule = NeuronGroup(num_neurons, adex_eqs, threshold='v > -50*mV', reset='v = EL; w += b', method='euler',
    namespace={'C': 200*pF, 'gL': 10*nS, 'EL': -70*mV, 'VT': -48*mV, 'DeltaT': 2*mV, 'tau_w': 30*ms, 'a': 2*nS, 'b': 0.05*nA})
granule.v = -70*mV; granule.w = 0*pA; granule.I_inj = 300*pA
granule_mon = SpikeMonitor(granule)

# Golgi
golgi = NeuronGroup(num_neurons, adex_eqs, threshold='v > -50*mV', reset='v = EL; w += b', method='euler',
    namespace={'C': 200*pF, 'gL': 10*nS, 'EL': -65*mV, 'VT': -52*mV, 'DeltaT': 2*mV, 'tau_w': 100*ms, 'a': 4*nS, 'b': 0.1*nA})
golgi.v = -65*mV; golgi.w = 0*pA; golgi.I_inj = 300*pA
golgi_mon = SpikeMonitor(golgi)

# Basket
basket = NeuronGroup(num_neurons, adex_eqs, threshold='v > -50*mV', reset='v = EL; w += b', method='euler',
    namespace={'C': 150*pF, 'gL': 10*nS, 'EL': -65*mV, 'VT': -50*mV, 'DeltaT': 1.5*mV, 'tau_w': 10*ms, 'a': 0*nS, 'b': 0.05*nA})
basket.v = -65*mV; basket.w = 0*pA; basket.I_inj = 300*pA
basket_mon = SpikeMonitor(basket)

# Stellate (Izhikevich)
stellate_eqs = '''
dv/dt = (0.04*(v/mV)**2*mV + 5*v + 140*mV - u*mV + I_syn/(1*nS))/ms : volt
du/dt = a*(b*(v/mV) - u) : 1
I_syn : amp
a : 1/second
b : 1
'''
stellate = NeuronGroup(num_neurons, stellate_eqs, threshold='v > 30*mV', reset='v = -65*mV; u += 8', method='euler',
    namespace={'a': 0.02/ms, 'b': 0.2})
stellate.v = -65*mV; stellate.u = -14
stellate_mon = SpikeMonitor(stellate)

# Purkinje (HH corregido)
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
purkinje = NeuronGroup(num_neurons, hh_eqs, threshold='v > -45*mV', reset='v = -65*mV', method='exponential_euler',
    namespace={'Cm': 1*uF/cm**2, 'gNa': 120*msiemens/cm**2, 'gK': 36*msiemens/cm**2,
               'gL': 0.3*msiemens/cm**2, 'ENa': 50*mV, 'EK': -77*mV, 'EL': -54.4*mV})
purkinje.v = -65*mV; purkinje.m = 0.05; purkinje.h = 0.6; purkinje.n = 0.32; purkinje.I = 25 * uA / cm**2
purkinje_mon = SpikeMonitor(purkinje)

# Nuclei
nuclei = NeuronGroup(num_neurons, adex_eqs, threshold='v > -50*mV', reset='v = EL; w += b', method='euler',
    namespace={'C': 250*pF, 'gL': 10*nS, 'EL': -68*mV, 'VT': -50*mV, 'DeltaT': 2*mV, 'tau_w': 100*ms, 'a': 0*nS, 'b': 0.15*nA})
nuclei.v = -68*mV; nuclei.w = 0*pA; nuclei.I_inj = 300*pA
nuclei_mon = SpikeMonitor(nuclei)

# -------- Sinapsis --------
s1 = Synapses(mossy, granule, on_pre='I_syn += 800*pA'); s1.connect(p=connect_p)
s2 = Synapses(granule, purkinje, on_pre='I_syn += 1*nA'); s2.connect(p=connect_p)
s3 = Synapses(granule, golgi, on_pre='I_syn += 1*nA'); s3.connect(p=connect_p)
s4 = Synapses(golgi, granule, on_pre='I_syn -= 1*nA'); s4.connect(p=connect_p)
s5 = Synapses(granule, basket, on_pre='I_syn += 800*pA'); s5.connect(p=connect_p)
s6 = Synapses(granule, stellate, on_pre='I_syn += 800*pA'); s6.connect(p=connect_p)
s7 = Synapses(basket, purkinje, on_pre='I_syn -= 1*nA'); s7.connect(p=connect_p)
s8 = Synapses(stellate, purkinje, on_pre='I_syn -= 1*nA'); s8.connect(p=connect_p)
s9 = Synapses(climbing, purkinje, on_pre='I_syn += 2*nA'); s9.connect(p=connect_p)
s10 = Synapses(purkinje, nuclei, on_pre='I_syn -= 2*nA'); s10.connect(p=connect_p)
s11 = Synapses(mossy, nuclei, on_pre='I_syn += 1*nA'); s11.connect(p=connect_p)

# -------- Ejecutar --------
run(duration)

# -------- Guardar CSV --------
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

# Guardar todos
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

print("✅ Simulación terminada y datasets guardados.")
