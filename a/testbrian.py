from brian2 import *

# Configuramos la duración de la simulación
duration = 100*ms

# Definimos el modelo de neurona LIF
eqs = '''
dv/dt = (1.0 - v) / (10*ms) : 1
'''

# Creamos el grupo de una sola neurona
G = NeuronGroup(1, eqs, threshold='v>0.8', reset='v = 0', method='exact')
G.v = 0  # voltaje inicial

# Registramos el voltaje
M = StateMonitor(G, 'v', record=True)
spikemon = SpikeMonitor(G)

# Ejecutamos la simulación
run(duration)

# Graficamos el resultado
import matplotlib.pyplot as plt

plt.plot(M.t/ms, M.v[0])
plt.xlabel('Tiempo (ms)')
plt.ylabel('Voltaje de membrana (v)')
plt.title('Neurona LIF simulada con Brian2')
plt.show()
