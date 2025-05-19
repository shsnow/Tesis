from brian2 import *
import numpy as np
import os

# Crear carpeta para datasets si no existe
os.makedirs("datasets", exist_ok=True)

# Tiempo de simulación
duration = 500*ms

def save_data(neuron_type, monitor):
    np.savez(f"datasets/{neuron_type}.npz", t=monitor.t/ms, v=monitor.v/mV)

# ---------------------- Purkinje Cell (Hodgkin-Huxley) ----------------------
equations_purkinje = Equations('''
dv/dt = (I - gNa*(m**3)*h*(v - ENa)
            - gK*(n**4)*(v - EK)
            - gL*(v - EL))/Cm : volt

dm/dt = alpha_m*(1 - m) - beta_m*m : 1

dh/dt = alpha_h*(1 - h) - beta_h*h : 1

dn/dt = alpha_n*(1 - n) - beta_n*n : 1

alpha_m = 0.1/mV*(25*mV - v)/(exp((25*mV - v)/(10*mV)) - 1)/ms : Hz
beta_m = 4*exp((-(v))/(18*mV))/ms : Hz
alpha_h = 0.07*exp((-(v))/(20*mV))/ms : Hz
beta_h = 1/(exp((30*mV - v)/(10*mV)) + 1)/ms : Hz
alpha_n = 0.01/mV*(10*mV - v)/(exp((10*mV - v)/(10*mV)) - 1)/ms : Hz
beta_n = 0.125*exp((-(v))/(80*mV))/ms : Hz
I : amp
''')

purkinje = NeuronGroup(1, equations_purkinje, method='exponential_euler',
                       namespace=dict(Cm=1*uF/cm**2, gNa=120*msiemens/cm**2,
                                      gK=36*msiemens/cm**2, gL=0.3*msiemens/cm**2,
                                      ENa=50*mV, EK=-77*mV, EL=-54.4*mV))
purkinje.v = -65*mV
purkinje.m = 0.05
purkinje.h = 0.6
purkinje.n = 0.32
purkinje.I = 10*uA/cm**2
monitor_purkinje = StateMonitor(purkinje, 'v', record=0)
run(duration)
save_data("purkinje", monitor_purkinje)

# ---------------------- Granule Cell (AdEx) ----------------------
equations_adex = Equations('''
dv/dt = ( -gL*(v - EL) + gL*DeltaT*exp((v - VT)/DeltaT) - w + I )/C : volt

dw/dt = (a*(v - EL) - w)/tau_w : amp
I : amp
''')

granule = NeuronGroup(1, equations_adex, threshold='v > -30*mV', reset='v = EL; w += b',
                      method='euler', namespace=dict(C=200*pF, gL=10*nS, EL=-70*mV,
                                                     VT=-50*mV, DeltaT=2*mV,
                                                     tau_w=30*ms, a=2*nS, b=0.05*nA))
granule.v = -70*mV
granule.w = 0*pA
granule.I = 0.4*nA
monitor_granule = StateMonitor(granule, 'v', record=0)
run(duration)
save_data("granule", monitor_granule)

# ---------------------- Golgi Cell (AdEx) ----------------------
golgi = NeuronGroup(1, equations_adex, threshold='v > -30*mV', reset='v = EL; w += b',
                    method='euler', namespace=dict(C=200*pF, gL=10*nS, EL=-70*mV,
                                                   VT=-55*mV, DeltaT=2*mV,
                                                   tau_w=100*ms, a=2*nS, b=0.1*nA))
golgi.v = -70*mV
golgi.w = 0*pA
golgi.I = 0.35*nA
monitor_golgi = StateMonitor(golgi, 'v', record=0)
run(duration)
save_data("golgi", monitor_golgi)

# ---------------------- Stellate Cell (AdEx) ----------------------
stellate = NeuronGroup(1, equations_adex, threshold='v > -30*mV', reset='v = EL; w += b',
                       method='euler', namespace=dict(C=100*pF, gL=10*nS, EL=-65*mV,
                                                      VT=-50*mV, DeltaT=2*mV,
                                                      tau_w=30*ms, a=0*nS, b=0.05*nA))
stellate.v = -65*mV
stellate.w = 0*pA
stellate.I = 0.3*nA
monitor_stellate = StateMonitor(stellate, 'v', record=0)
run(duration)
save_data("stellate", monitor_stellate)

# ---------------------- Basket Cell (AdEx) ----------------------
basket = NeuronGroup(1, equations_adex, threshold='v > -30*mV', reset='v = EL; w += b',
                     method='euler', namespace=dict(C=100*pF, gL=10*nS, EL=-65*mV,
                                                    VT=-52*mV, DeltaT=2*mV,
                                                    tau_w=20*ms, a=0*nS, b=0.05*nA))
basket.v = -65*mV
basket.w = 0*pA
basket.I = 0.4*nA
monitor_basket = StateMonitor(basket, 'v', record=0)
run(duration)
save_data("basket", monitor_basket)

# ---------------------- Climbing Fiber (Spike input) ----------------------
climbing_input = TimedArray([0*nA, 0.6*nA, 0*nA, 0*nA, 0*nA]*100, dt=1*ms)
climbing = NeuronGroup(1, equations_adex, threshold='v > -30*mV', reset='v = EL; w += b',
                       method='euler', namespace=dict(C=200*pF, gL=10*nS, EL=-70*mV,
                                                      VT=-50*mV, DeltaT=2*mV,
                                                      tau_w=50*ms, a=0*nS, b=0.1*nA))
climbing.v = -70*mV
climbing.w = 0*pA
climbing.run_regularly('I = climbing_input(t)', dt=1*ms)
monitor_climbing = StateMonitor(climbing, 'v', record=0)
run(duration)
save_data("climbing_fiber", monitor_climbing)

# ---------------------- Mossy Fiber (tonic input) ----------------------
mossy = NeuronGroup(1, equations_adex, threshold='v > -30*mV', reset='v = EL; w += b',
                   method='euler', namespace=dict(C=200*pF, gL=10*nS, EL=-70*mV,
                                                  VT=-50*mV, DeltaT=2*mV,
                                                  tau_w=50*ms, a=0*nS, b=0.05*nA))
mossy.v = -70*mV
mossy.w = 0*pA
mossy.I = 0.35*nA
monitor_mossy = StateMonitor(mossy, 'v', record=0)
run(duration)
save_data("mossy_fiber", monitor_mossy)

# ---------------------- Cerebellar Nuclei (AdEx, tonic + burst) ----------------------
nuclei = NeuronGroup(1, equations_adex, threshold='v > -30*mV', reset='v = EL; w += b',
                    method='euler', namespace=dict(C=250*pF, gL=10*nS, EL=-68*mV,
                                                   VT=-50*mV, DeltaT=2*mV,
                                                   tau_w=100*ms, a=0*nS, b=0.15*nA))
nuclei.v = -68*mV
nuclei.w = 0*pA
nuclei.I = 0.4*nA
monitor_nuclei = StateMonitor(nuclei, 'v', record=0)
run(duration)
save_data("cerebellar_nuclei", monitor_nuclei)

print("Datasets generados en la carpeta datasets/")
