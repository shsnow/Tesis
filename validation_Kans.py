import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from brian2 import *

# Importar la clase y configuraciones de nuestra librería
from Cerebellar_Class import NeuronaCerebelarKAN, BASE_MODEL_DIR, BASE_DATA_DIR

def validar_modelo_funcional(
    nombre_celula,
    parametros_simulacion, # Parámetros del modelo Brian2 original
    duracion_ms,
    corriente_de_prueba_nA, # Un array de numpy con la corriente de entrada
    dt_ms
    ):
    """
    Valida un modelo KAN entrenado comparando su salida en bucle cerrado
    con una simulación "ground truth" de Brian2 para el mismo estímulo.
    
    Args:
        nombre_celula (str): El nombre del modelo a validar (ej. 'granule_lif').
        parametros_simulacion (dict): Parámetros para el modelo Brian2.
        duracion_ms (float): Duración de la simulación de validación.
        corriente_de_prueba_nA (np.ndarray): Array con la corriente de entrada.
        dt_ms (float): Paso de tiempo en ms.
    """
    print(f"\n===== INICIANDO VALIDACIÓN FUNCIONAL PARA: {nombre_celula.upper()} =====")
    
    # --- 1. Cargar el Modelo KAN Entrenado ---
    # Asumimos que las columnas de features son las estándar para LIF
    features = ["time_ms", "voltage_mV", "input_current_nA"]
    
    neurona_kan = NeuronaCerebelarKAN(
        nombre_celula,
        columnas_features=features
    )
    
    if not neurona_kan.cargar_modelo():
        print(f"❌ No se pudo cargar el modelo KAN para {nombre_celula}. Abortando validación.")
        return

    # --- 2. Ejecutar Simulación "Ground Truth" con Brian2 ---
    print("INFO: Ejecutando simulación 'Ground Truth' con Brian2...")
    start_scope()
    defaultclock.dt = dt_ms * ms
    
    input_current_brian = TimedArray(corriente_de_prueba_nA * nA, dt=defaultclock.dt)
    
    # Usaremos un modelo LIF simple para esta validación de ejemplo
    eqs_lif_validacion = '''
    dv/dt = (-g_L*(v - EL_lif) + I)/C : volt 
    I = I_ext(t) : amp # Corriente de entrada
    g_L : siemens
    C : farad
    EL_lif : volt 
    '''
    
    G = NeuronGroup(1, eqs_lif_validacion, 
                    threshold='v > V_th_lif',
                    reset='v = V_res_lif',
                    method='euler',
                    namespace=parametros_simulacion)
    
    G.g_L = parametros_simulacion['g_L']
    G.C = parametros_simulacion['C']
    G.v = parametros_simulacion['EL_lif']

    monitor_estado_gt = StateMonitor(G, 'v', record=0)
    monitor_spikes_gt = SpikeMonitor(G)
    
    run(duracion_ms * ms)
    
    # Extraer resultados ground truth
    tiempo_gt = monitor_estado_gt.t / ms
    voltaje_gt = monitor_estado_gt.v[0] / mV
    spikes_gt = monitor_spikes_gt.t / ms
    print("   ...simulación 'Ground Truth' completada.")

    # --- 3. Ejecutar Simulación "KAN en Bucle Cerrado" ---
    print("INFO: Ejecutando simulación en bucle cerrado con el modelo KAN...")
    
    # Inicializar arrays
    num_pasos = len(tiempo_gt)
    voltaje_kan = np.zeros(num_pasos)
    voltaje_kan[0] = parametros_simulacion['EL_lif'] / mV # Condición inicial
    spikes_kan = []
    
    for i in range(num_pasos - 1):
        # Preparar la entrada para la KAN en el paso actual
        input_df = pd.DataFrame({
            'time_ms': [tiempo_gt[i]],
            'voltage_mV': [voltaje_kan[i]],
            'input_current_nA': [corriente_de_prueba_nA[i]]
        })
        
        # Predecir si hay un spike
        _, pred_binaria = neurona_kan.predecir(input_df)
        
        # Actualizar el voltaje para el siguiente paso
        if pred_binaria[0][0] == 1:
            # Si la KAN predice un spike, se resetea el voltaje
            voltaje_kan[i+1] = parametros_simulacion['V_res_lif'] / mV
            spikes_kan.append(tiempo_gt[i])
        else:
            # Si no hay spike, se integra un paso usando la ecuación LIF
            v_actual = voltaje_kan[i] * mV
            I_actual = corriente_de_prueba_nA[i] * nA
            
            # Euler integration step
            dv = ((-parametros_simulacion['g_L'] * (v_actual - parametros_simulacion['EL_lif'])) + I_actual) / parametros_simulacion['C']
            voltaje_kan[i+1] = (v_actual + dv * (dt_ms*ms)) / mV

    print("   ...simulación KAN completada.")

    # --- 4. Comparar y Graficar ---
    print("INFO: Generando gráfico comparativo...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Graficar trazas de voltaje
    ax.plot(tiempo_gt, voltaje_gt, label='Brian2 (Ground Truth)', color='royalblue', linewidth=2)
    ax.plot(tiempo_gt, voltaje_kan, label='KAN (Predicción en Bucle Cerrado)', color='darkorange', linestyle='--', linewidth=2)
    
    # Graficar spikes
    ax.plot(spikes_gt, [np.max(voltaje_gt)] * len(spikes_gt), 'o', color='royalblue', markersize=8, label='Spikes Brian2')
    ax.plot(spikes_kan, [np.max(voltaje_kan)] * len(spikes_kan), 'x', color='darkorange', markersize=8, markeredgewidth=2, label='Spikes KAN')
    
    ax.set_title(f"Validación Funcional del Modelo KAN para: {nombre_celula.replace('_', ' ').title()}", fontsize=16)
    ax.set_xlabel("Tiempo (ms)", fontsize=12)
    ax.set_ylabel("Potencial de Membrana (mV)", fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    
    # Calcular y mostrar métricas de error (ej. RMSE)
    rmse = np.sqrt(np.mean((voltaje_gt - voltaje_kan)**2))
    ax.text(0.02, 0.95, f'RMSE = {rmse:.2f} mV', transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
            
    ruta_salida_grafico = os.path.join(BASE_MODEL_DIR, f"validacion_funcional_{nombre_celula}.png")
    plt.savefig(ruta_salida_grafico)
    plt.close()
    print(f"✅ Gráfico de validación guardado en: {ruta_salida_grafico}")


if __name__ == '__main__':
    # --- Configuración para la Validación de Ejemplo ---
    
    # 1. Definir la célula a validar
    NOMBRE_CELULA_A_VALIDAR = "granule_lif"
    DT_MS_VALIDACION = 0.1 # Debe coincidir con el dt de la simulación LIF original

    # 2. Definir los parámetros del modelo Brian2 para esa célula (deben coincidir con los de la generación)
    PARAMS_GRANULE_LIF = {
        "g_L": 5*nS, "C": 100*pF, "EL_lif": -70*mV, 
        "V_th_lif": -50*mV, "V_res_lif": -65*mV,
        "I_ext": None # La corriente se pasará como TimedArray
    }
    
    # 3. Generar un estímulo de prueba NUNCA ANTES VISTO
    duracion_validacion_ms = 500
    num_pasos_validacion = int(duracion_validacion_ms / DT_MS_VALIDACION)
    
    # Un estímulo con un pulso de corriente
    corriente_prueba = np.zeros(num_pasos_validacion)
    corriente_prueba[int(100/DT_MS_VALIDACION):int(400/DT_MS_VALIDACION)] = 0.5 # Pulso de 0.5 nA
    
    # Añadir un poco de ruido
    corriente_prueba += np.random.randn(num_pasos_validacion) * 0.05
    
    # 4. Ejecutar la validación
    validar_modelo_funcional(
        nombre_celula=NOMBRE_CELULA_A_VALIDAR,
        parametros_simulacion=PARAMS_GRANULE_LIF,
        duracion_ms=duracion_validacion_ms,
        corriente_de_prueba_nA=corriente_prueba,
        dt_ms=DT_MS_VALIDACION
    )

    print("\n\n===== DEMO DE VALIDACIÓN COMPLETADA =====")