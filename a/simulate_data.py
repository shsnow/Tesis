from brian2 import *
import numpy as np
import os

# Directorio donde guardar los datasets
DATASET_DIR = "./datasets"
os.makedirs(DATASET_DIR, exist_ok=True)

# Tiempos de simulación
simulation_time = 500*ms

def create_monitor_group(neuron_group):
    M = StateMonitor(neuron_group, ['v'], record=True)
    spikes = SpikeMonitor(neuron_group)
    return M, spikes

def save_dataset(name, M):
    voltage = M.v[:]
    time = M.t[:]
    np.savez(os.path.join(DATASET_DIR, f"{name}_dataset.npz"), time=time/ms, voltage=voltage/mV)
    print(f"Dataset saved for {name}")

# ----------------------------
# Hodgkin-Huxley para Purkinje
# ----------------------------
def purkinje_hodgkin_huxley():
    eqs = '''
    dv/dt = (I - gNa*(m**3)*h*(v - ENa) - gK*(n**4)*(v - EK) - gL*(v - EL))/Cm : volt
    dm/dt = alpha_m*(1 - m) - beta_m*m : 1
    dn/dt = alpha_n*(1 - n) - beta_n*n : 1
    dh/dt = alpha_h*(1 - h) - beta_h*h : 1
    alpha_m = 0.1*(mV**-1)*(v + 40*mV)/(1 - exp(-(v + 40*mV)/(10*mV))) : Hz
    beta_m = 4*exp(-(v + 65*mV)/(18*mV)) : Hz
    alpha_h = 0.07*exp(-(v + 65*mV)/(20*mV)) : Hz
    beta_h = 1/(1 + exp(-(v + 35*mV)/(10*mV))) : Hz
    alpha_n = 0.01*(mV**-1)*(v + 55*mV)/(1 - exp(-(v + 55*mV)/(10*mV))) : Hz
    beta_n = 0.125*exp(-(v + 65*mV)/(80*mV)) : Hz
    I : amp
    '''

    gNa = 120*msiemens/cm**2
    gK = 36*msiemens/cm**2
    gL = 0.3*msiemens/cm**2
    ENa = 50*mV
    EK = -77*mV
    EL = -54.4*mV
    Cm = 1*uF/cm**2

    G = NeuronGroup(1, eqs, method='exponential_euler')
    G.v = -65*mV
    G.h = 0.6
    G.m = 0.05
    G.n = 0.32
    G.I = 10*uA/cm**2

    M, spikes = create_monitor_group(G)
    run(simulation_time)
    save_dataset("purkinje", M)

# ----------------------------
# AdEx model for other cells
# ----------------------------

def adex_cell(name, a, b, tau_w, v_thresh, v_reset, delta_T, Iinj):
    eqs = '''
    dv/dt = ( -gL*(v - EL) + gL*delta_T*exp((v - VT)/delta_T) - w + Iinj ) / Cm : volt (unless refractory)
    dw/dt = (a*(v - EL) - w)/tau_w : amp
    '''
    
    Cm = 200*pF
    gL = 10*nS
    EL = -70*mV
    VT = -50*mV
    tau_ref = 2*ms

    G = NeuronGroup(1, eqs, threshold='v > v_thresh', reset='v = v_reset; w += b', method='euler', refractory=tau_ref)
    G.v = EL
    G.w = 0*pA

    M, spikes = create_monitor_group(G)
    run(simulation_time)
    save_dataset(name, M)

# ----------------------------
# Simulaciones específicas
# ----------------------------

def simulate_all():
    purkinje_hodgkin_huxley()
    
    adex_cell("granule", a=2*nS, b=0.05*nA, tau_w=30*ms, v_thresh=-40*mV, v_reset=-60*mV, delta_T=2*mV, Iinj=400*pA)
    adex_cell("golgi", a=4*nS, b=0.08*nA, tau_w=40*ms, v_thresh=-45*mV, v_reset=-55*mV, delta_T=2*mV, Iinj=300*pA)
    adex_cell("stellate", a=2*nS, b=0.04*nA, tau_w=20*ms, v_thresh=-40*mV, v_reset=-55*mV, delta_T=1.5*mV, Iinj=350*pA)
    adex_cell("basket", a=3*nS, b=0.06*nA, tau_w=25*ms, v_thresh=-42*mV, v_reset=-60*mV, delta_T=1.8*mV, Iinj=370*pA)

if __name__ == "__main__":
    simulate_all()
