# =============================================
# Configuración extendida para generación de datasets
# =============================================
import pandas as pd
from scipy import stats
import numpy as np
from brian2 import *

# Duración de la simulación (2 segundos)
duration = 2000*ms

# Parámetros de muestreo
sample_interval = 1*ms  # Intervalo de muestreo para los StateMonitors
bin_size = 50*ms        # Tamaño del bin para características de tasa de disparo

# =============================================
# Funciones auxiliares para generación de datasets
# =============================================
def generate_neuron_dataset(spikemon, statemon=None, group_size=None):
    """Genera un dataset completo para un tipo neuronal"""
    # Datos de spikes crudos
    spike_times = spikemon.t/ms
    neuron_ids = spikemon.i
    
    # Características básicas
    spike_counts = np.bincount(neuron_ids, minlength=group_size)
    mean_firing_rate = spike_counts / (duration/ms) * 1000  # Hz
    
    # Características temporales (usando bines)
    num_bins = int(duration/bin_size)
    binned_spikes = np.zeros((group_size, num_bins))
    
    for i in range(group_size):
        neuron_spikes = spike_times[neuron_ids == i]
        binned_spikes[i], _ = np.histogram(neuron_spikes, bins=num_bins, range=(0, duration/ms))
    
    # Crear DataFrame
    data = {
        'neuron_id': np.arange(group_size),
        'mean_firing_rate': mean_firing_rate,
        'total_spikes': spike_counts,
        'spike_times': [spike_times[neuron_ids == i] for i in range(group_size)],
        'binned_firing_rates': [binned_spikes[i] for i in range(group_size)]
    }
    
    # Añadir datos de voltaje si están disponibles
    if statemon is not None:
        voltage_data = statemon.v/mV
        time_points = statemon.t/ms
        
        data['voltage_timeseries'] = [voltage_data[i] for i in range(group_size)]
        data['voltage_mean'] = np.mean(voltage_data, axis=1)
        data['voltage_std'] = np.std(voltage_data, axis=1)
        
        # Características de voltaje
        for i in range(group_size):
            data[f'voltage_autocorr_lag1_{i}'] = pd.Series(voltage_data[i]).autocorr(lag=1)
    
    df = pd.DataFrame(data)
    
    # Añadir características estadísticas
    df['firing_rate_variability'] = np.std(binned_spikes, axis=1) / (mean_firing_rate + 1e-6)
    df['burstiness'] = (df['total_spikes'] - np.mean(df['total_spikes'])) / np.std(df['total_spikes'])
    
    return df

def save_kan_datasets():
    """Guarda todos los datasets en formato adecuado para KAN"""
    # Generar datasets para cada población
    mossy_df = generate_neuron_dataset(mossy_spikemon, group_size=num_mossy)
    granule_df = generate_neuron_dataset(granule_spikemon, group_size=num_granule)
    golgi_df = generate_neuron_dataset(golgi_spikemon, group_size=num_golgi)
    basket_df = generate_neuron_dataset(basket_spikemon, group_size=num_basket)
    stellate_df = generate_neuron_dataset(stellate_spikemon, group_size=num_stellate)
    climbing_df = generate_neuron_dataset(climbing_spikemon, group_size=num_climbing)
    purkinje_df = generate_neuron_dataset(purkinje_spikemon, purkinje_statemon, group_size=num_purkinje)
    nuclei_df = generate_neuron_dataset(nuclei_spikemon, nuclei_statemon, group_size=num_nuclei)
    
    # Guardar como archivos CSV
    mossy_df.to_csv('cerebellar_datasets/kan_mossy_dataset.csv', index=False)
    granule_df.to_csv('cerebellar_datasets/kan_granule_dataset.csv', index=False)
    golgi_df.to_csv('cerebellar_datasets/kan_golgi_dataset.csv', index=False)
    basket_df.to_csv('cerebellar_datasets/kan_basket_dataset.csv', index=False)
    stellate_df.to_csv('cerebellar_datasets/kan_stellate_dataset.csv', index=False)
    climbing_df.to_csv('cerebellar_datasets/kan_climbing_dataset.csv', index=False)
    purkinje_df.to_csv('cerebellar_datasets/kan_purkinje_dataset.csv', index=False)
    nuclei_df.to_csv('cerebellar_datasets/kan_nuclei_dataset.csv', index=False)
    
    # Guardar también en formato pickle para preservar estructuras complejas
    import pickle
    with open('cerebellar_datasets/kan_full_dataset.pkl', 'wb') as f:
        pickle.dump({
            'mossy': mossy_df,
            'granule': granule_df,
            'golgi': golgi_df,
            'basket': basket_df,
            'stellate': stellate_df,
            'climbing': climbing_df,
            'purkinje': purkinje_df,
            'nuclei': nuclei_df
        }, f)

# =============================================
# Modificación de los monitores para mayor resolución
# =============================================
# Asegurarse de que todos los StateMonitors tengan el mismo intervalo de muestreo
purkinje_statemon = StateMonitor(purkinje_cells, ['v', 'm', 'h', 'n'], record=True, dt=sample_interval)
nuclei_statemon = StateMonitor(deep_nuclei, 'v', record=True, dt=sample_interval)

# =============================================
# Ejecución de la simulación
# =============================================
run(duration)

# =============================================
# Guardado de datasets para KAN
# =============================================
save_kan_datasets()

print("Datasets para KAN generados exitosamente en cerebellar_datasets/")