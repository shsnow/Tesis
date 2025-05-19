from brian2 import *

# Parámetros del modelo LIF
C = 1 * uF / cm**2       # Capacitancia por unidad de área
gL = 0.1 * msiemens / cm**2  # Conductancia de fuga
EL = -65 * mV            # Potencial de fuga

# Ecuaciones con I como densidad de corriente
eqs = '''
dv/dt = (-(v - EL)*gL + I) / C : volt
I : amp/meter**2
'''

# Crear la neurona
purkinje = NeuronGroup(1, eqs, method='euler')
purkinje.v = EL

# Aplicar una densidad de corriente directamente
purkinje.I = 10 * uA / cm**2  # No convertir a amperes totales

# Monitor de voltaje
mon = StateMonitor(purkinje, 'v', record=True)

# Ejecutar simulación
run(100*ms)

# Graficar resultados
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.plot(mon.t/ms, mon.v[0]/mV)
plt.xlabel('Tiempo (ms)')
plt.ylabel('Voltaje (mV)')
plt.title('Modelo LIF con densidad de corriente de entrada')
plt.grid()
plt.show()
