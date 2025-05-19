from brian2 import *
import pandas as pd
import os

# Directorio de salida para los CSVs
output_dir = "output_csv"
os.makedirs(output_dir, exist_ok=True)

# Parámetros para el modelo HH de la célula Purkinje
start_scope()

N = 1000  # Número de neuronas
duration = 100*ms

# Ecuaciones del modelo de Hodgkin-Huxley
eqs = '''
dv/dt = (I - gNa * m**3 * h * (v - ENa)
         - gK * n**4 * (v - EK)
         - gL * (v - EL)) / C : volt

dm/dt = alpham*(1 - m) - betam*m : 1
dn/dt = alphan*(1 - n) - betan*n : 1
dh/dt = alphah*(1 - h) - betah*h : 1

alpham = 0.1/mV * (25*mV - v) / (exp((25*mV - v)/(10*mV)) - 1)/ms : Hz
betam = 4*exp(-v/(18*mV))/ms : Hz

alphah = 0.07*exp(-v/(20*mV))/ms : Hz
betah = 1/(exp((30*mV - v)/(10*mV)) + 1)/ms : Hz

alphan = 0.01/mV * (10*mV - v) / (exp((10*mV - v)/(10*mV)) - 1)/ms : Hz
betan = 0.125*exp(-v/(80*mV))/ms : Hz

I : amp
'''

# Parámetros del modelo
C = 1*uF/cm**2
gNa = 120*msiemens/cm**2
gK = 36*msiemens/cm**2
gL = 0.3*msiemens/cm**2
ENa = 50*mV
EK = -77*mV
EL = -54.4*mV

# Creamos el grupo de neuronas
neurons = NeuronGroup(N, eqs, method='exponential_euler')
neurons.v = -65*mV
neurons.h = 0.6
neurons.m = 0.05
neurons.n = 0.32

# Corriente de entrada aleatoria entre 5-10 uA
neurons.I = np.random.uniform(5, 10, N) * uA

# Monitor para grabar voltaje y corriente
mon = StateMonitor(neurons, ['v', 'I'], record=True)

# Simulación
run(duration)

# Guardar datos en CSV
for i in range(N):
    df = pd.DataFrame({
        'time (ms)': mon.t/ms,
        'voltage (mV)': mon.v[i]/mV,
        'input_current (uA)': mon.I[i]/uA
    })
    filename = os.path.join(output_dir, f'Purkinje_{i}.csv')
    df.to_csv(filename, index=False)

print(f"✅ ¡Listo! Se generaron {N} archivos CSV de neuronas Purkinje en '{output_dir}'")
