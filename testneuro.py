from brian2 import *

start_scope()
prefs.codegen.target = 'numpy'

duration = 1*second
num_neurons = 10

# Poisson group
poisson = PoissonGroup(num_neurons, rates=50*Hz)

# Target neurons (simple leaky integrator)
eqs = '''
dv/dt = (-gL*(v - EL) + I_syn + I_inj)/C : volt
I_syn : amp
I_inj : amp
'''
target = NeuronGroup(num_neurons, eqs, threshold='v > -50*mV', reset='v = EL', method='euler',
                     namespace={'C': 200*pF, 'gL': 10*nS, 'EL': -70*mV})
target.v = -70*mV
target.I_inj = '100*pA + 20*pA*rand()'

# Synapses
syn = Synapses(poisson, target, on_pre='I_syn += 200*pA')
syn.connect(j='i')

# Monitors
poisson_mon = SpikeMonitor(poisson)
target_mon = SpikeMonitor(target)

run(duration)

# Mostrar resultados
for i in range(num_neurons):
    print(f"Neuron {i}: Poisson spikes = {poisson_mon.count[i]}, Target spikes = {target_mon.count[i]}")
