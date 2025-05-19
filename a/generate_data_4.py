from brian2 import *

# Parámetros de las neuronas (modelo LIF)
C = 1 * uF / cm**2
gL = 0.1 * msiemens / cm**2
EL = -65 * mV
VT = -50 * mV
Vreset = -70 * mV

# Ecuaciones generales
eqs = '''
dv/dt = (-(v - EL)*gL + I) / C : volt (unless refractory)
I : amp/meter**2
'''

# Grupos neuronales
granular = NeuronGroup(100, eqs, threshold='v > VT', reset='v = Vreset', refractory=5*ms, method='euler')
purkinje = NeuronGroup(10, eqs, threshold='v > VT', reset='v = Vreset', refractory=5*ms, method='euler')
dcn = NeuronGroup(3, eqs, threshold='v > VT', reset='v = Vreset', refractory=5*ms, method='euler')

# Inicialización
granular.v = EL
purkinje.v = EL
dcn.v = EL

# Input externo a células granulares
granular.I = 10 * uA / cm**2

# Conexiones
S_gran_purk = Synapses(granular, purkinje, on_pre='I_post += 1.5 * uA / cm**2')  # Excitatorio
S_gran_purk.connect(p=0.1)

S_purk_dcn = Synapses(purkinje, dcn, on_pre='I_post -= 2.0 * uA / cm**2')  # Inhibitorio
S_purk_dcn.connect(p=0.5)

# Estímulo directo a DCN (opcional)
dcn.I = 0.5 * uA / cm**2

# Monitores
mon_g = SpikeMonitor(granular)
mon_p = SpikeMonitor(purkinje)
mon_d = SpikeMonitor(dcn)

vmon_d = StateMonitor(dcn, 'v', record=True)

# Simulación
run(200*ms)

# Graficar spikes
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))

plt.subplot(3,1,1)
plt.plot(mon_g.t/ms, mon_g.i, '.k')
plt.title('Spikes células granulares')
plt.ylabel('Índice')

plt.subplot(3,1,2)
plt.plot(mon_p.t/ms, mon_p.i, '.r')
plt.title('Spikes células Purkinje')
plt.ylabel('Índice')

plt.subplot(3,1,3)
plt.plot(mon_d.t/ms, mon_d.i, '.b')
plt.title('Spikes células DCN')
plt.ylabel('Índice')
plt.xlabel('Tiempo (ms)')

plt.tight_layout()
plt.show()
