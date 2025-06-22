import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from brian2 import *
import torch # 🌟 CORRECCIÓN: Añadido import de torch faltante

from Cerebellar_Class import NeuronaCerebelarKAN, BASE_MODEL_DIR, BASE_DATA_DIR


def generar_estimulos_de_prueba(duracion_ms, dt_ms):
    """
    Genera un diccionario con diferentes formas de onda de corriente para la validación.
    """
    print("INFO: Generando estímulos de prueba novedosos...")
    num_pasos = int(duracion_ms / dt_ms)
    tiempo = np.arange(num_pasos) * dt_ms

    # 1. Pulso Cuadrado
    corriente_pulso = np.zeros(num_pasos)
    corriente_pulso[int(100/dt_ms):int(400/dt_ms)] = 0.6 # nA

    # 2. Rampa Lineal
    corriente_rampa = np.linspace(0, 0.8, num_pasos) # De 0 a 0.8 nA

    # 3. Onda Sinusoidal
    corriente_sinusoidal = 0.4 + 0.3 * np.sin(2 * np.pi * 5 * tiempo / 1000) # 5 Hz

    # 4. Pulsos Cortos y Repetidos
    corriente_pulsos_cortos = np.zeros(num_pasos)
    for t_start in [100, 200, 300, 400]:
        corriente_pulsos_cortos[int(t_start/dt_ms):int((t_start+20)/dt_ms)] = 0.9 # Pulsos de 20ms

    estimulos = {
        "Pulso Cuadrado": corriente_pulso,
        "Rampa Lineal": corriente_rampa,
        "Onda Sinusoidal": corriente_sinusoidal,
        "Pulsos Cortos": corriente_pulsos_cortos
    }
    print("   ...estímulos generados.")
    return estimulos, tiempo


def ejecutar_simulacion_brian2(parametros_simulacion, corriente_de_prueba_nA, duracion_ms, dt_ms):
    """Ejecuta una simulación 'ground truth' en Brian2 y devuelve los resultados."""
    start_scope()
    defaultclock.dt = dt_ms * ms
    
    input_current_brian = TimedArray(corriente_de_prueba_nA * nA, dt=defaultclock.dt)
    
    eqs_lif_validacion = '''
    dv/dt = (-g_L*(v - EL_lif) + I)/C : volt 
    I = I_ext(t) : amp
    g_L : siemens
    C : farad
    EL_lif : volt 
    '''
    
    namespace_validacion = {
        'I_ext': input_current_brian,
        'V_th_lif': parametros_simulacion['V_th_lif'],
        'V_res_lif': parametros_simulacion['V_res_lif'],
        'EL_lif': parametros_simulacion['EL_lif']
    }
    
    G = NeuronGroup(1, eqs_lif_validacion, 
                    threshold='v > V_th_lif', reset='v = V_res_lif',
                    method='euler', namespace=namespace_validacion)
    
    G.g_L = parametros_simulacion['g_L']
    G.C = parametros_simulacion['C']
    G.v = parametros_simulacion['EL_lif']

    monitor_estado = StateMonitor(G, 'v', record=0)
    monitor_spikes = SpikeMonitor(G)
    
    run(duracion_ms * ms)
    
    return monitor_estado.t/ms, monitor_estado.v[0]/mV, monitor_spikes.t/ms


def ejecutar_simulacion_kan_en_bucle(neurona_kan, parametros_simulacion, corriente_de_prueba_nA, tiempo_ms, dt_ms):
    """Ejecuta una simulación en bucle cerrado con el modelo KAN."""
    num_pasos = len(tiempo_ms)
    voltaje_kan = np.zeros(num_pasos)
    voltaje_kan[0] = parametros_simulacion['EL_lif'] / mV
    spikes_kan = []
    
    for i in range(num_pasos - 1):
        input_df = pd.DataFrame({
            'time_ms': [tiempo_ms[i]],
            'voltage_mV': [voltaje_kan[i]],
            'input_current_nA': [corriente_de_prueba_nA[i]]
        })
        
        _, pred_binaria = neurona_kan.predecir(input_df)
        
        if pred_binaria is not None and pred_binaria[0][0] == 1:
            voltaje_kan[i+1] = parametros_simulacion['V_res_lif'] / mV
            spikes_kan.append(tiempo_ms[i])
        else:
            v_actual = voltaje_kan[i] * mV
            I_actual = corriente_de_prueba_nA[i] * nA
            dv = ((-parametros_simulacion['g_L'] * (v_actual - parametros_simulacion['EL_lif'])) + I_actual) / parametros_simulacion['C']
            voltaje_kan[i+1] = (v_actual + dv * (dt_ms*ms)) / mV
            
    return voltaje_kan, np.array(spikes_kan)


# --- Secciones de Validación ---

def validar_generalizacion_estimulos(nombre_celula, parametros_simulacion, dt_ms):
    """Genera un gráfico de múltiples paneles para validar la generalización del modelo KAN."""
    print(f"\n--- INICIANDO PRUEBA DE GENERALIZACIÓN PARA: {nombre_celula.upper()} ---")
    
    neurona_kan = NeuronaCerebelarKAN(nombre_celula)
    if not neurona_kan.cargar_modelo(): return

    estimulos, tiempo_base = generar_estimulos_de_prueba(duracion_ms=500, dt_ms=dt_ms)
    
    fig, axes = plt.subplots(len(estimulos), 1, figsize=(15, 5 * len(estimulos)), sharex=True)
    fig.suptitle(f"Validación de Generalización del Modelo KAN para: {nombre_celula.replace('_', ' ').title()}", fontsize=20, y=0.97)

    for i, (nombre_estimulo, corriente_prueba) in enumerate(estimulos.items()):
        ax = axes[i]
        
        # Simulación Ground Truth
        _, voltaje_gt, spikes_gt = ejecutar_simulacion_brian2(parametros_simulacion, corriente_prueba, 500, dt_ms)
        
        # Simulación KAN
        voltaje_kan, spikes_kan = ejecutar_simulacion_kan_en_bucle(neurona_kan, parametros_simulacion, corriente_prueba, tiempo_base, dt_ms)

        # Graficar
        ax.plot(tiempo_base, voltaje_gt, label='Brian2 (Ground Truth)', color='royalblue', linewidth=2, alpha=0.8)
        ax.plot(tiempo_base, voltaje_kan, label='KAN (Predicción)', color='darkorange', linestyle='--', linewidth=1.5, alpha=0.8)
        ax.plot(spikes_gt, [np.min(voltaje_gt)-5] * len(spikes_gt), '|', color='royalblue', markersize=15, markeredgewidth=2, label='Spikes Brian2')
        ax.plot(spikes_kan, [np.min(voltaje_gt)-8] * len(spikes_kan), 'x', color='darkorange', markersize=8, markeredgewidth=2, label='Spikes KAN')
        ax.set_title(f"Estímulo: {nombre_estimulo}", fontsize=14)
        ax.set_ylabel("Potencial (mV)")
        ax.legend()
        ax.grid(True)
    
    axes[-1].set_xlabel("Tiempo (ms)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    ruta_salida_grafico = os.path.join(BASE_MODEL_DIR, f"validacion_generalizacion_{nombre_celula}.png")
    plt.savefig(ruta_salida_grafico)
    plt.close()
    print(f"✅ Gráfico de generalización para {nombre_celula} guardado en: {ruta_salida_grafico}")


def validar_rendimiento_computacional(nombre_celula, parametros_simulacion, dt_ms):
    """Mide y compara el tiempo de ejecución de Brian2 vs KAN usando predicción en lote."""
    print(f"\n--- INICIANDO PRUEBA DE RENDIMIENTO PARA: {nombre_celula.upper()} ---")
    
    neurona_kan = NeuronaCerebelarKAN(nombre_celula)
    if not neurona_kan.cargar_modelo(): return

    duracion_ms = 2000
    print(f"Generando un dataset de prueba de {duracion_ms} ms para la comparación...")
    estimulos, _ = generar_estimulos_de_prueba(duracion_ms, dt_ms)
    corriente_prueba = estimulos["Onda Sinusoidal"] 
    
    tiempo_gt, voltaje_gt, _ = ejecutar_simulacion_brian2(parametros_simulacion, corriente_prueba, duracion_ms, dt_ms)
    
    # Crear el DataFrame de entrada para la predicción en lote de KAN
    df_prueba_kan = pd.DataFrame({
        'time_ms': tiempo_gt,
        'voltage_mV': voltaje_gt, 
        'input_current_nA': corriente_prueba
    })

    # Medir tiempo de Brian2
    print("Midiendo tiempo de Brian2...")
    start_time_brian2 = time.time()
    ejecutar_simulacion_brian2(parametros_simulacion, corriente_prueba, duracion_ms, dt_ms)
    end_time_brian2 = time.time()
    tiempo_brian2 = end_time_brian2 - start_time_brian2
    print(f"   ...Tiempo Brian2: {tiempo_brian2:.4f} segundos")

    # Medir tiempo de KAN (predicción en un solo lote)
    print("Midiendo tiempo de KAN en predicción por lote...")
    start_time_kan = time.time()
    neurona_kan.predecir(df_prueba_kan) 
    end_time_kan = time.time()
    tiempo_kan = end_time_kan - start_time_kan
    print(f"   ...Tiempo KAN: {tiempo_kan:.4f} segundos")
    
    if tiempo_kan > 0:
        mejora_velocidad = tiempo_brian2 / tiempo_kan
        print(f"\n✅ RESULTADO: El modelo KAN es ~{mejora_velocidad:.1f} veces más rápido que la simulación de Brian2 en esta tarea.")
    else:
        print("\nINFO: No se puede calcular la mejora de velocidad (tiempo KAN fue demasiado rápido o cero).")

def validar_interpretabilidad(nombre_celula):
    """Carga un modelo KAN y visualiza sus funciones de activación (splines)."""
    print(f"\n--- INICIANDO PRUEBA DE INTERPRETABILIDAD PARA: {nombre_celula.upper()} ---")
    
    neurona_kan = NeuronaCerebelarKAN(nombre_celula)
    if not neurona_kan.cargar_modelo(): return

    try:
        print("Generando gráfico de splines (model.plot())...")
        # CORRECCIÓN: Pasar un pequeño lote de datos para 'calentar' el modelo antes de graficar
        # Usaremos las primeras 100 filas del dataset ligero con el que se entrenó
        df_calentamiento = pd.read_csv(neurona_kan.ruta_datos_csv, nrows=100)
        X_calentamiento = df_calentamiento[neurona_kan.columnas_features].values.astype(np.float32)
        X_calentamiento_scaled = neurona_kan.scaler_cargado.transform(X_calentamiento)
        # 🌟 Acceder al dispositivo del modelo cargado de forma segura
        device = neurona_kan.modelo_kan_cargado.device
        X_calentamiento_tensor = torch.tensor(X_calentamiento_scaled).to(device)
        
        neurona_kan.modelo_kan_cargado(X_calentamiento_tensor) # Este es el paso clave de "calentamiento"
        
        fig = neurona_kan.modelo_kan_cargado.plot(beta=10) 
        fig.suptitle(f"Funciones Aprendidas por KAN para: {nombre_celula.replace('_', ' ').title()}", fontsize=16)
        ruta_salida_grafico = os.path.join(BASE_MODEL_DIR, f"interpretabilidad_{nombre_celula}.png")
        fig.savefig(ruta_salida_grafico)
        plt.close(fig)
        print(f"✅ Gráfico de interpretabilidad para {nombre_celula} guardado en: {ruta_salida_grafico}")
    except Exception as e:
        print(f"❌ ERROR al generar gráfico de interpretabilidad para {nombre_celula}: {e}")
        print("   Asegúrate de que la versión de 'pykan' que estás usando soporta el método 'plot'.")


if __name__ == '__main__':
    # --- Configuración Centralizada para Todas las Células ---
    CONFIGURACION_VALIDACION = {
        "granule_lif": {
            "params": {"g_L": 5*nS, "C": 100*pF, "EL_lif": -70*mV, "V_th_lif": -50*mV, "V_res_lif": -65*mV},
            "dt_ms": 0.1
        },
        "golgi_lif": {
            "params": {"g_L": 10*nS, "C": 200*pF, "EL_lif": -68*mV, "V_th_lif": -52*mV, "V_res_lif": -65*mV},
            "dt_ms": 0.1
        },
        "basket_lif": {
            "params": {"g_L": 12*nS, "C": 150*pF, "EL_lif": -67*mV, "V_th_lif": -50*mV, "V_res_lif": -65*mV},
            "dt_ms": 0.1
        },
        "stellate_lif": {
            "params": {"g_L": 10*nS, "C": 180*pF, "EL_lif": -68*mV, "V_th_lif": -52*mV, "V_res_lif": -66*mV},
            "dt_ms": 0.1
        },
        "deep_nuclei_lif": {
            "params": {"g_L": 15*nS, "C": 300*pF, "EL_lif": -65*mV, "V_th_lif": -50*mV, "V_res_lif": -66*mV},
            "dt_ms": 0.1
        },
        "mossy_fiber": {
            "params": {"g_L": 10*nS, "C": 150*pF, "EL_lif": -70*mV, "V_th_lif": -50*mV, "V_res_lif": -70*mV},
            "dt_ms": 0.1
        },
        "climbing_fiber": {
            "params": {"g_L": 8*nS, "C": 100*pF, "EL_lif": -65*mV, "V_th_lif": -45*mV, "V_res_lif": -68*mV},
            "dt_ms": 0.1
        }
        # Nota: La validación funcional para el modelo HH de Purkinje es más compleja de implementar
        # porque la ecuación de actualización en el bucle cerrado no es tan simple como la de LIF.
        # Por ahora, nos centramos en validar todas las células LIF.
    }

    # --- Bucle Principal de Validación ---
    print("===== INICIANDO VALIDACIÓN COMPLETA DE TODOS LOS MODELOS DE CÉLULAS =====")
    for nombre_celula, config in CONFIGURACION_VALIDACION.items():
        print(f"\n\n{'='*20} VALIDANDO: {nombre_celula.upper()} {'='*20}")
        # 1. Validación de generalización
        validar_generalizacion_estimulos(nombre_celula, config["params"], config["dt_ms"])
        
        # 2. Validación de rendimiento
        validar_rendimiento_computacional(nombre_celula, config["params"], config["dt_ms"])
        
        # 3. Validación de interpretabilidad
        validar_interpretabilidad(nombre_celula)

    print("\n\n🏁 TODAS LAS VALIDACIONES COMPLETADAS 🏁")
