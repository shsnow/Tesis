from brian2 import *
import numpy as np
import os
import matplotlib.pyplot as plt

# Configuración inicial
prefs.codegen.target = 'numpy'  # Para mejor compatibilidad
start_scope()

# Directorio para datasets
os.makedirs("cerebellar_datasets", exist_ok=True)

# Duración de la simulación (2 segundos)
duration = 2000*ms

# =============================================
# 1. Mossy Fibers (Entrada sensorial/motora)
# =============================================
num_mossy = 1000
mossy_rates = 20*Hz  # Tasa base
mossy_poisson = PoissonGroup(num_mossy, rates=mossy_rates)

# Monitor de spikes
mossy_spikemon = SpikeMonitor(mossy_poisson)

# =============================================
# 2. Granule Cells (Transformación de señal)
# =============================================
num_granule = 1000  # Gran cantidad de células granulares

# Modelo AdEx para fast-spiking
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
                              'VT': -40*mV,  # Umbral alto
                              'DeltaT': 2*mV,
                              'tau_w': 30*ms,
                              'a': 2*nS,
                              'b': 0.02*nA
                          })
granule_cells.v = -70*mV
granule_cells.w = 0*pA

# Conexión Mossy -> Granule (divergencia)
mossy_granule_syn = Synapses(mossy_poisson, granule_cells, 'w_syn : 1', on_pre='I_syn += 100*pA')
mossy_granule_syn.connect(j='i*20')  # Cada Mossy conecta a 20 Granule

granule_spikemon = SpikeMonitor(granule_cells)

# =============================================
# 3. Golgi Cells (Inhibición feedback)
# =============================================
num_golgi = 1000
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
                            'b': 0.1*nA  # Parámetro de bursting
                        })
golgi_cells.v = -60*mV
golgi_cells.w = 0*pA

# Conexión Granule -> Golgi
granule_golgi_syn = Synapses(granule_cells, golgi_cells, 'w_syn : 1', on_pre='I_syn += 80*pA')
granule_golgi_syn.connect(j='i%40')  # Cada Golgi recibe de ~40 Granule

# Conexión inhibitoria Golgi -> Granule
golgi_granule_syn = Synapses(golgi_cells, granule_cells, 'w_syn : 1', on_pre='I_syn -= 150*pA')
golgi_granule_syn.connect(j='i*40')  # Cada Golgi inhibe ~40 Granule

golgi_spikemon = SpikeMonitor(golgi_cells)

# =============================================
# 4. Basket & Stellate Cells (Inhibición lateral)
# =============================================
num_basket = 1000
num_stellate = 1000

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
                             'DeltaT': 0.5*mV,  # DeltaT bajo para fast-spiking
                             'tau_w': 10*ms,
                             'a': 0*nS,  # Sin adaptación
                             'b': 0.05*nA
                         })
basket_cells.v = -65*mV
basket_cells.w = 0*pA

# Stellate Cells (Regular spiking) - VERSIÓN DEFINITIVAMENTE CORREGIDA
stellate_eqs = '''
dv/dt = (0.04*(v**2/mV) + 5*v + 140*mV - u*mV + I_syn/(1*nS)) / ms : volt
du/dt = a*(b*(v/mV) - u) : 1  # Ahora u es adimensional
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
stellate_cells.u = -14  # Ahora adimensional

# Conexión Granule -> Basket/Stellate
granule_basket_syn = Synapses(granule_cells, basket_cells, 'w_syn : 1', on_pre='I_syn += 120*pA')
granule_basket_syn.connect(j='i%20')

granule_stellate_syn = Synapses(granule_cells, stellate_cells, 'w_syn : 1', on_pre='I_syn += 120*pA')
granule_stellate_syn.connect(j='i%20')

basket_spikemon = SpikeMonitor(basket_cells)
stellate_spikemon = SpikeMonitor(stellate_cells)

# =============================================
# 5. Climbing Fibers (Señal de error)
# =============================================
num_climbing = 1000
climbing_rates = 1*Hz  # Baja frecuencia
climbing_poisson = PoissonGroup(num_climbing, rates=climbing_rates)

# Monitor de spikes
climbing_spikemon = SpikeMonitor(climbing_poisson)


# =============================================
# 6. Purkinje Cells (Salida principal) - VERSIÓN CORREGIDA
# =============================================
num_purkinje = 1000

# Modelo Hodgkin-Huxley extendido - CORRECCIÓN DE UNIDADES
# Modelo Hodgkin-Huxley extendido - VERSIÓN CORREGIDA
purkinje_eqs = '''
dv/dt = (I - (gNa*(m**3)*h*(v - ENa) + gK*(n**4)*(v - EK) + gL*(v - EL)) + I_syn/cm**2)/Cm : volt
dm/dt = alpha_m*(1 - m) - beta_m*m : 1
dn/dt = alpha_n*(1 - n) - beta_n*n : 1
dh/dt = alpha_h*(1 - h) - beta_h*h : 1
I_syn : amp
I : amp/meter**2  # Densidad de corriente

alpha_m = 0.1/mV*(25*mV - v)/(exp((25*mV - v)/(10*mV)) - 1)/ms : Hz
beta_m = 4*exp(-(v)/(18*mV))/ms : Hz
alpha_h = 0.07*exp(-(v)/(20*mV))/ms : Hz
beta_h = 1/(exp((30*mV - v)/(10*mV)) + 1)/ms : Hz
alpha_n = 0.01/mV*(10*mV - v)/(exp((10*mV - v)/(10*mV)) - 1)/ms : Hz
beta_n = 0.125*exp(-(v)/(80*mV))/ms : Hz
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
purkinje_cells.I = '(10 + 3*sin(2*pi*3*Hz*t)) * uA/cm**2'  # Corriente oscilante para bursting


purkinje_spikemon = SpikeMonitor(purkinje_cells)
purkinje_statemon = StateMonitor(purkinje_cells, 'v', record=True)


# =============================================
# 7. Cerebellar Deep Nuclei (Salida final)
# =============================================
num_nuclei = 1000

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
                            'b': 0.15*nA  # Parámetro de bursting
                        })
deep_nuclei.v = -68*mV
deep_nuclei.w = 0*pA
deep_nuclei.I_syn = '0*nA'  # Solo inhibición

# Inhibición desde Purkinje
purkinje_nuclei_syn = Synapses(purkinje_cells, deep_nuclei, 'w_syn : 1', on_pre='I_syn -= 300*pA')
purkinje_nuclei_syn.connect(j='i%10')

nuclei_spikemon = SpikeMonitor(deep_nuclei)
nuclei_statemon = StateMonitor(deep_nuclei, 'v', record=True)

# =============================================
# Ejecución de la simulación
# =============================================
run(duration)

# =============================================
# Guardado de datos
# =============================================
def save_spike_data(name, spikemon):
    """Guarda datos de spikes en formato NPZ"""
    np.savez(f"cerebellar_datasets/{name}_spikes.npz",
             i=spikemon.i,
             t=spikemon.t/ms)

save_spike_data("mossy", mossy_spikemon)
save_spike_data("granule", granule_spikemon)
save_spike_data("golgi", golgi_spikemon)
save_spike_data("basket", basket_spikemon)
save_spike_data("stellate", stellate_spikemon)
save_spike_data("climbing", climbing_spikemon)
save_spike_data("purkinje", purkinje_spikemon)
save_spike_data("nuclei", nuclei_spikemon)

np.savez("cerebellar_datasets/purkinje_voltage.npz",
         t=purkinje_statemon.t/ms,
         v0=purkinje_statemon.v[0]/mV,
         v50=purkinje_statemon.v[50]/mV)

np.savez("cerebellar_datasets/nuclei_voltage.npz",
         t=nuclei_statemon.t/ms,
         v=nuclei_statemon.v.T/mV)

print("Simulación completada y datos guardados en cerebellar_datasets/")

# =============================================
# Visualización
# =============================================
plt.figure(figsize=(15, 10))

# Mossy Fibers activity
plt.subplot(3, 2, 1)
plt.plot(mossy_spikemon.t/ms, mossy_spikemon.i, '.k', markersize=1)
plt.title('Mossy Fibers')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')

# Granule Cells activity
plt.subplot(3, 2, 2)
plt.plot(granule_spikemon.t/ms, granule_spikemon.i, '.b', markersize=1)
plt.title('Granule Cells')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')

# Purkinje Cells voltage
plt.subplot(3, 2, 3)
plt.plot(purkinje_statemon.t/ms, purkinje_statemon.v[0]/mV, label='PC 0')
plt.plot(purkinje_statemon.t/ms, purkinje_statemon.v[50]/mV, label='PC 50')
plt.title('Purkinje Cells Voltage')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')
plt.legend()

# Deep Nuclei voltage
plt.subplot(3, 2, 4)
plt.plot(nuclei_statemon.t/ms, nuclei_statemon.v.T/mV)
plt.title('Deep Nuclei Voltage')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (mV)')

# Climbing Fibers activity
plt.subplot(3, 2, 5)
plt.plot(climbing_spikemon.t/ms, climbing_spikemon.i, '.r')
plt.title('Climbing Fibers')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')

# Basket/Stellate activity
plt.subplot(3, 2, 6)
plt.plot(basket_spikemon.t/ms, basket_spikemon.i, '.g', markersize=1, label='Basket')
plt.plot(stellate_spikemon.t/ms, stellate_spikemon.i, '.m', markersize=1, label='Stellate')
plt.title('Basket & Stellate Cells')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron index')
plt.legend()

plt.tight_layout()
plt.savefig('cerebellar_datasets/simulation_results.png')
plt.show()