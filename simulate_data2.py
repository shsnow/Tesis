# cerebellum_simulator.py
from brian2 import *
import numpy as np

# ===================================
# Parámetros generales de simulación
# ===================================
defaultclock.dt = 0.1*ms
duration = 200*ms  # Duración de la simulación

# ================================
# Base de clase para una célula
# ================================
class CerebellarCell:
    def __init__(self, N=1):
        self.N = N
        self.group = None
        
    def run(self):
        monitor = StateMonitor(self.group, 'v', record=True)
        run(duration)
        return monitor

# ===============================
# Célula Purkinje - Hodgkin-Huxley
# ===============================
class PurkinjeCell(CerebellarCell):
    def __init__(self, N=1):
        super().__init__(N)
        eqs = '''
        dv/dt = (I_ext - gNa*(m**3)*h*(v-ENa) - gK*(n**4)*(v-EK) - gL*(v-EL)) / C : volt
        dm/dt = alpham*(1-m) - betam*m : 1
        dn/dt = alphan*(1-n) - betan*n : 1
        dh/dt = alphah*(1-h) - betah*h : 1
        alpham = 0.1*(mV**-1)*(v+40*mV)/(1-exp(-(v+40*mV)/(10*mV)))/ms : Hz
        betam = 4*exp(-(v+65*mV)/(18*mV))/ms : Hz
        alphah = 0.07*exp(-(v+65*mV)/(20*mV))/ms : Hz
        betah = 1/(exp(-(v+35*mV)/(10*mV))+1)/ms : Hz
        alphan = 0.01*(mV**-1)*(v+55*mV)/(1-exp(-(v+55*mV)/(10*mV)))/ms : Hz
        betan = 0.125*exp(-(v+65*mV)/(80*mV))/ms : Hz
        I_ext : amp
        '''
        C = 1*uF/cm**2
        gNa = 120*msiemens/cm**2
        gK = 36*msiemens/cm**2
        gL = 0.3*msiemens/cm**2
        ENa = 50*mV
        EK = -77*mV
        EL = -54.4*mV
        
        self.group = NeuronGroup(N, eqs, threshold='v > -20*mV', reset='v = -65*mV', method='exponential_euler')
        self.group.v = -65*mV
        self.group.h = 0.6
        self.group.m = 0.05
        self.group.n = 0.32
        self.group.I_ext = 10*uA/cm**2

# =========================
# Célula de la Granulosa - AdEx
# =========================
class GranuleCell(CerebellarCell):
    def __init__(self, N=1):
        super().__init__(N)
        eqs = '''
        dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) - w + I_ext)/C : volt
        dw/dt = (a*(v - EL) - w)/tau_w : amp
        I_ext : amp
        '''
        C = 200*pF
        gL = 10*nS
        EL = -70*mV
        VT = -50*mV
        DeltaT = 2*mV
        a = 2*nS
        tau_w = 100*ms
        
        self.group = NeuronGroup(N, eqs, threshold='v > -40*mV', reset='v = EL; w += b', method='euler')
        self.group.v = EL
        self.group.w = 0*pA
        self.group.I_ext = 150*pA  # fuerte para fast-spiking
        self.group.b = 0*pA  # poco bursting

# =========================
# Mossy Fiber - Regular spiking simple (AdEx también)
# =========================
class MossyFiber(CerebellarCell):
    def __init__(self, N=1):
        super().__init__(N)
        eqs = '''
        dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) - w + I_ext)/C : volt
        dw/dt = (a*(v - EL) - w)/tau_w : amp
        I_ext : amp
        '''
        C = 200*pF
        gL = 10*nS
        EL = -70*mV
        VT = -50*mV
        DeltaT = 2*mV
        a = 2*nS
        tau_w = 100*ms
        
        self.group = NeuronGroup(N, eqs, threshold='v > -40*mV', reset='v = EL; w += b', method='euler')
        self.group.v = EL
        self.group.w = 0*pA
        self.group.I_ext = 120*pA
        self.group.b = 0*pA

# ==========================
# Golgi Cell - Bursting
# ==========================
class GolgiCell(CerebellarCell):
    def __init__(self, N=1):
        super().__init__(N)
        eqs = '''
        dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) - w + I_ext)/C : volt
        dw/dt = (a*(v - EL) - w)/tau_w : amp
        I_ext : amp
        '''
        C = 200*pF
        gL = 10*nS
        EL = -60*mV
        VT = -50*mV
        DeltaT = 2*mV
        a = 4*nS
        tau_w = 150*ms
        
        self.group = NeuronGroup(N, eqs, threshold='v > -30*mV', reset='v = EL; w += b', method='euler')
        self.group.v = EL
        self.group.w = 0*pA
        self.group.I_ext = 180*pA
        self.group.b = 60*pA

# ==========================
# Basket Cell - Fast Spiking
# ==========================
class BasketCell(CerebellarCell):
    def __init__(self, N=1):
        super().__init__(N)
        eqs = '''
        dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) - w + I_ext)/C : volt
        dw/dt = (a*(v - EL) - w)/tau_w : amp
        I_ext : amp
        '''
        C = 150*pF
        gL = 10*nS
        EL = -65*mV
        VT = -45*mV
        DeltaT = 0.5*mV
        a = 0*nS
        tau_w = 30*ms
        
        self.group = NeuronGroup(N, eqs, threshold='v > -30*mV', reset='v = EL; w += b', method='euler')
        self.group.v = EL
        self.group.w = 0*pA
        self.group.I_ext = 250*pA
        self.group.b = 0*pA

# ===========================
# Stellar Cell - Regular Spiking
# ===========================
class StellarCell(CerebellarCell):
    def __init__(self, N=1):
        super().__init__(N)
        eqs = '''
        dv/dt = (gL*(EL - v) + gL*DeltaT*exp((v - VT)/DeltaT) - w + I_ext)/C : volt
        dw/dt = (a*(v - EL) - w)/tau_w : amp
        I_ext : amp
        '''
        C = 200*pF
        gL = 10*nS
        EL = -65*mV
        VT = -50*mV
        DeltaT = 2*mV
        a = 2*nS
        tau_w = 80*ms
        
        self.group = NeuronGroup(N, eqs, threshold='v > -30*mV', reset='v = EL; w += b', method='euler')
        self.group.v = EL
        self.group.w = 0*pA
        self.group.I_ext = 150*pA
        self.group.b = 40*pA

# ===========================
# Climbing Fiber & Parallel Fiber
# ===========================
class ClimbingFiber(CerebellarCell):
    def __init__(self, N=1):
        super().__init__(N)
        eqs = '''
        dv/dt = (-v + I_ext)/tau : volt
        I_ext : volt
        tau : second
        '''
        self.group = NeuronGroup(N, eqs, threshold='v > 1*mV', reset='v = 0*mV', method='exact')
        self.group.v = 0*mV
        self.group.I_ext = 1.5*mV
        self.group.tau = 5*ms

class ParallelFiber(ClimbingFiber):
    pass

# =========================================
# Ejemplo de uso
# =========================================
if __name__ == "__main__":
    cell = PurkinjeCell(N=1)  # Cambia aquí el tipo de célula
    monitor = cell.run()
    
    # Graficar
    plot(monitor.t/ms, monitor.v[0]/mV)
    xlabel('Tiempo (ms)')
    ylabel('Voltaje (mV)')
    title('Simulación de Purkinje Cell')
    show()
