from brian2 import *
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats
import pickle

# Configuración inicial
prefs.codegen.target = 'numpy'
start_scope()

# Directorio para datasets
os.makedirs("cerebellar_datasets", exist_ok=True)

# Duración de la simulación (2 segundos)
duration = 2000*ms
sample_interval = 1*ms  # Intervalo de muestreo para los StateMonitors
bin_size = 50*ms        # Tamaño del bin para características de tasa de disparo

# =============================================
# 1. Mossy Fibers (Entrada sensorial/motora)
# =============================================
num_mossy = 100
mossy_rates = 20*Hz
mossy_poisson = PoissonGroup(num_mossy, rates=mossy_rates)
mossy_spikemon = SpikeMonitor(mossy_poisson)

# =============================================
# 2. Granule Cells (Transformación de señal)
# =============================================
num_granule = 2000
granule_eqs = '''
dv/dt = ( -gL*(v - EL) + gL*DeltaT*exp((v - VT)/DeltaT) - w + I_syn ) / C : volt
dw/dt = (a*(v - EL) - w)/tau_w : amp
I_syn : amp
'''

granule_cells = NeuronGroup(num_granule, granule_eqs, threshold='v > -40*mV', 
                          reset='v = EL; w += b', method='euler',
                          namespace={
                              'C': 200*pF,
                              'gL': 10*nS,
                              'EL': -70*mV,
                              'VT': -40*mV,
                              'DeltaT': 2*mV,
                              'tau_w': 30*ms,
                              'a': 2*nS,
                              'b': 0.02*nA
                          })
granule_cells.v = -70*mV
granule_cells.w = 0*pA

mossy_granule_syn = Synapses(mossy_poisson, granule_cells, 'w_syn : 1', on_pre='I_syn += 100*pA')
mossy_granule_syn.connect(j='i*20')
granule_spikemon = SpikeMonitor(granule_cells)

# =============================================
# 3. Golgi Cells (Inhibición feedback)
# =============================================
num_golgi = 2000
golgi_eqs = '''
dv/dt = ( -gL*(v - EL) + gL*DeltaT*exp((v - VT)/DeltaT) - w + I_syn ) / C : volt
dw/dt = (a*(v - EL) - w)/tau_w : amp
I_syn : amp
'''

golgi_cells = NeuronGroup(num_golgi, golgi_eqs, threshold='v > -40*mV',
                        reset='v = EL; w += b', method='euler',
                        namespace={
                            'C': 200*pF,
                            'gL': 10*nS,
                            'EL': -60*mV,
                            'VT': -50*mV,
                            'DeltaT': 2*mV,
                            'tau_w': 100*ms,
                            'a': 4*nS,
                            'b': 0.1*nA
                        })
golgi_cells.v = -60*mV
golgi_cells.w = 0*pA

granule_golgi_syn = Synapses(granule_cells, golgi_cells, 'w_syn : 1', on_pre='I_syn += 80*pA')
granule_golgi_syn.connect(j='i%40')

golgi_granule_syn = Synapses(golgi_cells, granule_cells, 'w_syn : 1', on_pre='I_syn -= 150*pA')
golgi_granule_syn.connect(j='i*40')
golgi_spikemon = SpikeMonitor(golgi_cells)

# =============================================
# 4. Basket & Stellate Cells (Inhibición lateral)
# =============================================
num_basket = 2000
num_stellate = 2000

# Basket Cells (Fast-spiking)
basket_eqs = '''
dv/dt = ( -gL*(v - EL) + gL*DeltaT*exp((v - VT)/DeltaT) - w + I_syn ) / C : volt
dw/dt = (a*(v - EL) - w)/tau_w : amp
I_syn : amp
'''

basket_cells = NeuronGroup(num_basket, basket_eqs, threshold='v > -40*mV',
                         reset='v = EL; w += b', method='euler',
                         namespace={
                             'C': 150*pF,
                             'gL': 10*nS,
                             'EL': -65*mV,
                             'VT': -52*mV,
                             'DeltaT': 0.5*mV,
                             'tau_w': 10*ms,
                             'a': 0*nS,
                             'b': 0.05*nA
                         })
basket_cells.v = -65*mV
basket_cells.w = 0*pA

# Stellate Cells (Regular spiking)
stellate_eqs = '''
dv/dt = (0.04*(v**2/mV) + 5*v + 140*mV - u*mV + I_syn/(1*nS)) / ms : volt
du/dt = a*(b*(v/mV) - u) : 1
I_syn : amp
a : 1/second
b : 1
'''

stellate_cells = NeuronGroup(num_stellate, stellate_eqs, threshold='v > 30*mV',
                           reset='v = -65*mV; u += 8', method='euler',
                           namespace={
                               'a': 0.02/ms,
                               'b': 0.2
                           })
stellate_cells.v = -65*mV
stellate_cells.u = -14

granule_basket_syn = Synapses(granule_cells, basket_cells, 'w_syn : 1', on_pre='I_syn += 120*pA')
granule_basket_syn.connect(j='i%20')

granule_stellate_syn = Synapses(granule_cells, stellate_cells, 'w_syn : 1', on_pre='I_syn += 120*pA')
granule_stellate_syn.connect(j='i%20')

basket_spikemon = SpikeMonitor(basket_cells)
stellate_spikemon = SpikeMonitor(stellate_cells)

# =============================================
# 5. Climbing Fibers (Señal de error)
# =============================================
num_climbing = 2000
climbing_rates = 1*Hz
climbing_poisson = PoissonGroup(num_climbing, rates=climbing_rates)
climbing_spikemon = SpikeMonitor(climbing_poisson)

# =============================================
# 6. Purkinje Cells (Salida principal) - VERSIÓN CORREGIDA
# =============================================
num_purkinje = 2000

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
beta_h = 1.0/(exp((30*mV - v)/(10*mV)) + 1)/ms : Hz
alpha_n = (0.01*(10*mV - v)/mV)/(exp((10*mV - v)/(10*mV)) - 1)/ms : Hz
beta_n = 0.125*exp(-v/(80*mV))/ms : Hz
'''

purkinje_cells = NeuronGroup(num_purkinje, purkinje_eqs, 
                           threshold='v > -40*mV',
                           reset='v = -65*mV',
                           method='exponential_euler',
                           namespace={
                               'Cm': 1*uF/cm**2,
                               'gNa': 120*msiemens/cm**2,
                               'gK': 36*msiemens/cm**2,
                               'gL': 0.3*msiemens/cm**2,
                               'ENa': 50*mV,
                               'EK': -77*mV,
                               'EL': -54.4*mV
                           })
purkinje_cells.v = -65*mV
purkinje_cells.m = 0.05
purkinje_cells.h = 0.6
purkinje_cells.n = 0.32
purkinje_cells.I = '(10 + 3*sin(2*pi*3*Hz*t)) * uA/cm**2'

purkinje_spikemon = SpikeMonitor(purkinje_cells)
purkinje_statemon = StateMonitor(purkinje_cells, ['v', 'm', 'h', 'n'], record=True, dt=sample_interval)

# =============================================
# 7. Cerebellar Deep Nuclei (Salida final)
# =============================================
num_nuclei = 10

nuclei_eqs = '''
dv/dt = ( -gL*(v - EL) + gL*DeltaT*exp((v - VT)/DeltaT) - w + I_syn ) / C : volt
dw/dt = (a*(v - EL) - w)/tau_w : amp
I_syn : amp
'''

deep_nuclei = NeuronGroup(num_nuclei, nuclei_eqs, threshold='v > -40*mV',
                        reset='v = EL; w += b', method='euler',
                        namespace={
                            'C': 250*pF,
                            'gL': 10*nS,
                            'EL': -68*mV,
                            'VT': -50*mV,
                            'DeltaT': 2*mV,
                            'tau_w': 100*ms,
                            'a': 0*nS,
                            'b': 0.15*nA
                        })
deep_nuclei.v = -68*mV
deep_nuclei.w = 0*pA

purkinje_nuclei_syn = Synapses(purkinje_cells, deep_nuclei, 'w_syn : 1', on_pre='I_syn -= 300*pA')
purkinje_nuclei_syn.connect(j='i%10')

nuclei_spikemon = SpikeMonitor(deep_nuclei)
nuclei_statemon = StateMonitor(deep_nuclei, 'v', record=True, dt=sample_interval)

# =============================================
# Ejecución de la simulación
# =============================================
run(duration)

# =============================================
# Funciones para generación de datasets KAN
# =============================================
def calculate_spike_features(spike_times, duration_ms, neuron_id, num_neurons):
    """Calcula características avanzadas de los spikes"""
    features = {}
    
    # Características básicas
    spike_count = len(spike_times)
    features['spike_count'] = spike_count
    features['mean_firing_rate'] = spike_count / duration_ms * 1000  # Hz
    
    # Características temporales
    if spike_count > 1:
        isi = np.diff(spike_times)
        features['mean_isi'] = np.mean(isi)
        features['isi_variability'] = np.std(isi) / (features['mean_isi'] + 1e-6)
        features['burstiness'] = (np.mean(isi) - np.min(isi)) / (np.mean(isi) + 1e-6)
    else:
        features['mean_isi'] = duration_ms
        features['isi_variability'] = 0
        features['burstiness'] = 0
    
    # Características espectrales
    if spike_count > 0:
        hist, _ = np.histogram(spike_times, bins=int(duration_ms/10), range=(0, duration_ms))
        features['spectral_power'] = np.sum(np.abs(np.fft.fft(hist))**2)
    else:
        features['spectral_power'] = 0
    
    return features

def generate_neuron_dataset(spikemon, statemon=None, group_size=None):
    """Genera un dataset completo para un tipo neuronal"""
    spike_times = spikemon.t/ms
    neuron_ids = spikemon.i
    duration_ms = duration/ms
    
    # Inicializar DataFrame
    df = pd.DataFrame(index=range(group_size))
    
    # Características básicas de spikes
    spike_counts = np.bincount(neuron_ids, minlength=group_size)
    df['spike_count'] = spike_counts
    df['mean_firing_rate'] = spike_counts / duration_ms * 1000  # Hz
    
    # Características temporales avanzadas
    spike_features = []
    for i in range(group_size):
        neuron_spikes = spike_times[neuron_ids == i]
        features = calculate_spike_features(neuron_spikes, duration_ms, i, group_size)
        spike_features.append(features)
    
    spike_features_df = pd.DataFrame(spike_features)
    df = pd.concat([df, spike_features_df], axis=1)
    
    # Datos de voltaje si están disponibles
    if statemon is not None:
        voltage_data = statemon.v/mV
        time_points = statemon.t/ms
        
        # Características básicas de voltaje
        df['voltage_mean'] = np.mean(voltage_data, axis=1)
        df['voltage_std'] = np.std(voltage_data, axis=1)
        df['voltage_min'] = np.min(voltage_data, axis=1)
        df['voltage_max'] = np.max(voltage_data, axis=1)
        
        # Características dinámicas
        for i in range(group_size):
            df.loc[i, 'voltage_autocorr_lag1'] = pd.Series(voltage_data[i]).autocorr(lag=1)
            df.loc[i, 'voltage_spectral_power'] = np.sum(np.abs(np.fft.fft(voltage_data[i]))**2)
        
        # Guardar series temporales completas
        df['voltage_timeseries'] = [voltage_data[i] for i in range(group_size)]
        df['time_points'] = [time_points for _ in range(group_size)]
    
    # Guardar spikes crudos para análisis posterior
    df['spike_times'] = [spike_times[neuron_ids == i] for i in range(group_size)]
    
    return df


# =============================================
# Generar y guardar los datasets (versión corregida)
# =============================================

# Primero generamos todos los DataFrames
mossy_df = generate_neuron_dataset(mossy_spikemon, group_size=num_mossy)
granule_df = generate_neuron_dataset(granule_spikemon, group_size=num_granule)
golgi_df = generate_neuron_dataset(golgi_spikemon, group_size=num_golgi)
basket_df = generate_neuron_dataset(basket_spikemon, group_size=num_basket)
stellate_df = generate_neuron_dataset(stellate_spikemon, group_size=num_stellate)
climbing_df = generate_neuron_dataset(climbing_spikemon, group_size=num_climbing)
purkinje_df = generate_neuron_dataset(purkinje_spikemon, purkinje_statemon, group_size=num_purkinje)
nuclei_df = generate_neuron_dataset(nuclei_spikemon, nuclei_statemon, group_size=num_nuclei)

# Ahora guardamos los datasets
def save_kan_datasets():
    """Guarda todos los datasets en formato adecuado para KAN"""
    # Guardar como archivos CSV
    mossy_df.to_csv('cerebellar_datasets/kan_mossy_dataset.csv', index=False)
    granule_df.to_csv('cerebellar_datasets/kan_granule_dataset.csv', index=False)
    golgi_df.to_csv('cerebellar_datasets/kan_golgi_dataset.csv', index=False)
    basket_df.to_csv('cerebellar_datasets/kan_basket_dataset.csv', index=False)
    stellate_df.to_csv('cerebellar_datasets/kan_stellate_dataset.csv', index=False)
    climbing_df.to_csv('cerebellar_datasets/kan_climbing_dataset.csv', index=False)
    purkinje_df.to_csv('cerebellar_datasets/kan_purkinje_dataset.csv', index=False)
    nuclei_df.to_csv('cerebellar_datasets/kan_nuclei_dataset.csv', index=False)
    
    # Guardar también en formato pickle
    with open('cerebellar_datasets/kan_full_dataset.pkl', 'wb') as f:
        pickle.dump({
            'mossy': mossy_df,
            'granule': granule_df,
            'golgi': golgi_df,
            'basket': basket_df,
            'stellate': stellate_df,
            'climbing': climbing_df,
            'purkinje': purkinje_df,
            'nuclei': nuclei_df,
            'simulation_parameters': {
                'duration': duration,
                'sample_interval': sample_interval,
                'bin_size': bin_size
            }
        }, f)

# Ejecutar la función de guardado
save_kan_datasets()

# =============================================
# Visualización corregida
# =============================================
plt.figure(figsize=(15, 10))
plt.subplot(2, 1, 1)
plt.plot(purkinje_statemon.t/ms, purkinje_statemon.v[0]/mV)
plt.title('Ejemplo de Potencial de Membrana (Purkinje)')
plt.xlabel('Tiempo (ms)')
plt.ylabel('Voltaje (mV)')

plt.subplot(2, 1, 2)
plt.hist(purkinje_df['mean_firing_rate'], bins=20)  # Ahora purkinje_df está definida
plt.title('Distribución de Tasas de Disparo (Purkinje)')
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Número de Neuronas')

plt.tight_layout()
plt.savefig('cerebellar_datasets/summary_plots.png')
plt.show()

print("Simulación completada y datasets generados en cerebellar_datasets/")