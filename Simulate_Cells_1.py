from brian2 import *
import numpy as np
import pandas as pd
import os
from sklearn.utils import resample

# Configuración global de Brian2
prefs.codegen.target = "numpy" 
os.makedirs("dataset_cerebelo", exist_ok=True) # Carpeta de salida

# --- Función de Diagnóstico (Sin cambios) ---
def diagnosticar_dataset(df, nombre_dataset, df_crudo_antes_balanceo_global=None):
    """
    Realiza un diagnóstico básico del dataset generado para evaluar su calidad.
    """
    print(f"\n--- Diagnóstico del Dataset: '{nombre_dataset}' ---")
    
    if df is None or df.empty:
        print("❌ ERROR: El DataFrame para diagnóstico está vacío o es None.")
        if df_crudo_antes_balanceo_global is not None and not df_crudo_antes_balanceo_global.empty:
             print("   Información del DataFrame crudo (agregado de neuronas pre-filtradas):")
             print(f"   Filas: {len(df_crudo_antes_balanceo_global)}, Spikes: {df_crudo_antes_balanceo_global['spike'].sum() if 'spike' in df_crudo_antes_balanceo_global else 'N/A'}")
        return

    print("\n1. Verificación de Valores Inválidos:")
    nan_check = df.isnull().sum()
    inf_check_series = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        inf_check_series[col] = np.isinf(df[col]).sum()
    inf_check = pd.Series(inf_check_series)
    
    if nan_check.sum() == 0: print("   ✅ No se encontraron valores NaN.")
    else: print(f"   ⚠️ ADVERTENCIA: Se encontraron NaNs:\n{nan_check[nan_check > 0]}")
    if inf_check.sum() == 0: print("   ✅ No se encontraron valores Infinitos.")
    else: print(f"   ⚠️ ADVERTENCIA: Se encontraron Infs:\n{inf_check[inf_check > 0]}")

    print("\n2. Balance de Clases (Dataset Final Balanceado):")
    if 'spike' in df.columns:
        conteo_clases = df['spike'].value_counts()
        porcentaje_clases = df['spike'].value_counts(normalize=True) * 100
        print(f"   Conteo de clases:\n{conteo_clases}")
        print(f"   Porcentaje de clases:\n{porcentaje_clases}")
        if len(conteo_clases) == 2 and abs(porcentaje_clases.get(0,0) - porcentaje_clases.get(1,0)) < 10:
            print("   ✅ El dataset parece razonablemente balanceado.")
        elif len(conteo_clases) == 1:
             print(f"   ⚠️ ADVERTENCIA: El dataset final solo contiene una clase: {conteo_clases.index[0]}.")
        else:
            print("   ⚠️ ADVERTENCIA: El balance de clases no es ideal. Verifica la lógica.")
    else: print("   ⚠️ ADVERTENCIA: No se encontró la columna 'spike'.")

    print("\n3. Estadísticas de Características Principales:")
    columnas_para_stats = ['voltage_mV', 'input_current_nA']
    for col in columnas_para_stats:
        if col in df.columns:
            print(f"   Estadísticas para '{col}':")
            stats = df[col].describe()
            print(f"      Media: {stats['mean']:.2f}, Std: {stats['std']:.2f}, Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
            if stats['std'] < 1e-3 : 
                print(f"      ⚠️ ADVERTENCIA: Std de '{col}' es muy baja. ¿Datos poco variados?")
        else: print(f"   ⚠️ ADVERTENCIA: No se encontró la columna '{col}'.")
            
    if df_crudo_antes_balanceo_global is not None and not df_crudo_antes_balanceo_global.empty and 'spike' in df_crudo_antes_balanceo_global.columns:
        print("\n4. Información de Spikes (Agregado, antes del balanceo global):")
        spikes_agg = df_crudo_antes_balanceo_global['spike'].sum()
        total_agg = len(df_crudo_antes_balanceo_global)
        if total_agg > 0:
            porc_spikes_agg = (spikes_agg / total_agg) * 100
            print(f"   Puntos totales (pre-balanceo global): {total_agg}")
            print(f"   'Spikes' (pre-balanceo global): {spikes_agg}")
            print(f"   % 'Spikes' (pre-balanceo global): {porc_spikes_agg:.2f}%")
            if spikes_agg == 0: print("   ❌ ADVERTENCIA CRÍTICA: No se generaron spikes en datos agregados.")
        else: print("   ⚠️ ADVERTENCIA: DataFrame crudo (agregado) vacío.")

    print("\n5. Resumen de 'Salud' del Dataset:")
    num_filas = len(df)
    print(f"   Dataset final con {num_filas} filas.")
    if num_filas < 1000: print("   ⚠️ ADVERTENCIA: Dataset final pequeño.")
    
    problema_critico = any([
        nan_check.sum() > 0, 
        inf_check.sum() > 0,
        (df_crudo_antes_balanceo_global is not None and 'spike' in df_crudo_antes_balanceo_global and df_crudo_antes_balanceo_global['spike'].sum() == 0),
        ('spike' in df and len(df['spike'].value_counts()) < 2 and not (df_crudo_antes_balanceo_global is not None and 'spike' in df_crudo_antes_balanceo_global and df_crudo_antes_balanceo_global['spike'].sum() == 0))
    ])
    
    if not problema_critico: print("   ✅ El dataset parece estructuralmente válido y razonablemente balanceado.")
    else: print("   ❌ PROBLEMAS DETECTADOS. Revisa las advertencias.")
    print("--- Fin del Diagnóstico ---")

# --- Función Central de Procesamiento y Guardado de Datos (CON MUESTREO INTELIGENTE) ---
def procesar_y_guardar_datos_multi_neurona(
    nombre_base_archivo,
    monitor_estado,
    monitor_spikes,
    array_corriente_entrada_global_nA, 
    defaultclock_dt_ms,
    ratio_no_spike_to_spike_per_neuron=10,
    max_no_spikes_absoluto_per_neuron=15000,
    # 🌟 NUEVOS PARÁMETROS PARA MUESTREO INTELIGENTE
    peri_spike_window_ms=5.0, # Ventana en ms antes de un spike para considerar "rampa"
    peri_spike_sample_ratio=0.5 # Qué proporción de no-spikes tomar de esta ventana
):
    print(f"INFO: Iniciando procesamiento de datos para '{nombre_base_archivo}' con muestreo inteligente (ventana de {peri_spike_window_ms} ms)...")
    array_tiempo_ms_global = monitor_estado.t / ms

    indices_neuronas_activas = sorted(list(monitor_spikes.spike_trains().keys())) 
    if not indices_neuronas_activas: 
        active_in_monitor = [i for i, trace in enumerate(monitor_estado.v) if np.std(trace/mV) > 1e-3] 
        if not active_in_monitor:
            print(f"❌ ADVERTENCIA: Ninguna neurona generó spikes ni mostró actividad de voltaje significativa para '{nombre_base_archivo}'.")
            diagnosticar_dataset(None, nombre_base_archivo, pd.DataFrame())
            return
        else: 
            print(f"INFO: Ninguna neurona generó spikes, pero se detectó actividad de voltaje. Procesando todas las {len(monitor_estado.v)} neuronas.")
            indices_neuronas_activas = list(range(len(monitor_estado.v)))

    print(f"INFO: Se procesarán datos de {len(indices_neuronas_activas)} neuronas para '{nombre_base_archivo}'.")
    
    lista_dataframes_neurona_individual = []
    for idx_neurona in indices_neuronas_activas:
        array_voltaje_mV_neurona = monitor_estado.v[idx_neurona] / mV
        array_marcador_spike_neurona = np.zeros_like(array_tiempo_ms_global, dtype=int)
        
        if idx_neurona in monitor_spikes.spike_trains(): 
            tiempos_spike_neurona_ms = monitor_spikes.spike_trains()[idx_neurona] / ms
            indices_spike_neurona = np.round(tiempos_spike_neurona_ms / defaultclock_dt_ms).astype(int)
            indices_spike_neurona_validos = indices_spike_neurona[indices_spike_neurona < len(array_marcador_spike_neurona)]
            array_marcador_spike_neurona[indices_spike_neurona_validos] = 1

        indices_donde_hay_spike = np.where(array_marcador_spike_neurona == 1)[0]
        indices_donde_no_hay_spike = np.where(array_marcador_spike_neurona == 0)[0]
        num_spikes_esta_neurona = len(indices_donde_hay_spike)

        num_no_spikes_a_mantener = 0
        if num_spikes_esta_neurona > 0: 
            num_no_spikes_a_mantener = int(num_spikes_esta_neurona * ratio_no_spike_to_spike_per_neuron)
            num_no_spikes_a_mantener = min(num_no_spikes_a_mantener, max_no_spikes_absoluto_per_neuron)
        num_no_spikes_a_mantener = min(num_no_spikes_a_mantener, len(indices_donde_no_hay_spike)) 

        indices_no_spike_muestreados = np.array([], dtype=int)
        if num_no_spikes_a_mantener > 0:
            # --- LÓGICA DE MUESTREO INTELIGENTE ---
            # 1. Identificar los puntos de "rampa" justo antes de un spike
            window_size_in_indices = int(peri_spike_window_ms / defaultclock_dt_ms)
            indices_peri_spike = []
            for idx_spike in indices_donde_hay_spike:
                start_window = max(0, idx_spike - window_size_in_indices)
                indices_peri_spike.extend(range(start_window, idx_spike))
            
            # Intersecar con los no-spikes para obtener solo los de la rampa
            indices_rampa = np.intersect1d(np.unique(indices_peri_spike), indices_donde_no_hay_spike)
            
            # Identificar los no-spikes que están lejos de un spike (en reposo)
            indices_reposo = np.setdiff1d(indices_donde_no_hay_spike, indices_rampa)
            
            # 2. Decidir cuántos tomar de cada grupo
            num_a_tomar_de_rampa = int(num_no_spikes_a_mantener * peri_spike_sample_ratio)
            num_a_tomar_de_rampa = min(num_a_tomar_de_rampa, len(indices_rampa)) # No tomar más de los que hay

            num_a_tomar_de_reposo = num_no_spikes_a_mantener - num_a_tomar_de_rampa
            num_a_tomar_de_reposo = min(num_a_tomar_de_reposo, len(indices_reposo)) # No tomar más de los que hay

            # 3. Muestrear de cada grupo
            indices_rampa_muestreados = np.random.choice(indices_rampa, size=num_a_tomar_de_rampa, replace=False) if num_a_tomar_de_rampa > 0 else np.array([])
            indices_reposo_muestreados = np.random.choice(indices_reposo, size=num_a_tomar_de_reposo, replace=False) if num_a_tomar_de_reposo > 0 else np.array([])
            
            indices_no_spike_muestreados = np.concatenate([indices_rampa_muestreados, indices_reposo_muestreados])
            # -----------------------------------
        
        indices_finales_para_esta_neurona = np.sort(np.unique(np.concatenate([indices_donde_hay_spike, indices_no_spike_muestreados])))
        
        if len(indices_finales_para_esta_neurona) == 0:
            continue 

        df_neurona = pd.DataFrame({
            'time_ms': array_tiempo_ms_global[indices_finales_para_esta_neurona],
            'voltage_mV': array_voltaje_mV_neurona[indices_finales_para_esta_neurona],
            'input_current_nA': array_corriente_entrada_global_nA[indices_finales_para_esta_neurona], 
            'spike': array_marcador_spike_neurona[indices_finales_para_esta_neurona]
        })
        lista_dataframes_neurona_individual.append(df_neurona)

    if not lista_dataframes_neurona_individual:
        print(f"❌ No se generaron DataFrames individuales válidos para '{nombre_base_archivo}'.")
        diagnosticar_dataset(None, nombre_base_archivo, pd.DataFrame())
        return
        
    df_agregado_pre_balanceo = pd.concat(lista_dataframes_neurona_individual, ignore_index=True)
    
    nan_antes_balanceo = df_agregado_pre_balanceo.isnull().sum().sum()
    if nan_antes_balanceo > 0:
        print(f"INFO ({nombre_base_archivo}): Se encontraron {nan_antes_balanceo} NaNs en datos agregados antes del balanceo. Se eliminarán las filas con NaNs.")
        df_agregado_pre_balanceo.dropna(inplace=True)

    if df_agregado_pre_balanceo.empty:
        print(f"❌ ADVERTENCIA: DataFrame agregado para '{nombre_base_archivo}' quedó vacío después de eliminar NaNs. No se puede continuar.")
        diagnosticar_dataset(None, nombre_base_archivo, df_agregado_pre_balanceo) 
        return

    df_con_spike_global = df_agregado_pre_balanceo[df_agregado_pre_balanceo['spike'] == 1]
    df_sin_spike_global = df_agregado_pre_balanceo[df_agregado_pre_balanceo['spike'] == 0]

    if len(df_con_spike_global) == 0:
        print(f"INFO: No se capturaron spikes en el DataFrame agregado para '{nombre_base_archivo}'. El dataset no se balanceará globalmente.")
        df_balanceado_final = df_agregado_pre_balanceo.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        print(f"🔎 Para '{nombre_base_archivo}': Agregados {len(df_con_spike_global)} spikes y {len(df_sin_spike_global)} no-spikes (después de muestreo por neurona, antes del balanceo global).")
        n_spikes_global = len(df_con_spike_global)
        n_no_spikes_global = len(df_sin_spike_global)
        
        df_con_spike_remuestreado_global = df_con_spike_global 
        df_sin_spike_remuestreado_global = df_sin_spike_global 

        if n_no_spikes_global == 0 and n_spikes_global > 0: 
             df_balanceado_final = df_con_spike_remuestreado_global.sample(frac=1, random_state=42).reset_index(drop=True)
             print("INFO: Solo hay spikes en el dataset agregado. No se necesita balanceo.")
        elif n_spikes_global < n_no_spikes_global and n_spikes_global > 0 : 
            df_con_spike_remuestreado_global = resample(df_con_spike_global, replace=True, n_samples=n_no_spikes_global, random_state=42)
            df_balanceado_final = pd.concat([df_con_spike_remuestreado_global, df_sin_spike_remuestreado_global])
        elif n_spikes_global > n_no_spikes_global and n_no_spikes_global > 0 : 
             df_sin_spike_remuestreado_global = resample(df_sin_spike_global, replace=True, n_samples=n_spikes_global, random_state=42)
             df_balanceado_final = pd.concat([df_con_spike_remuestreado_global, df_sin_spike_remuestreado_global])
        elif n_spikes_global == n_no_spikes_global and n_spikes_global > 0: 
            df_balanceado_final = pd.concat([df_con_spike_remuestreado_global, df_sin_spike_remuestreado_global])
        else: 
            print(f"INFO: No se pudo realizar el balanceo estándar para '{nombre_base_archivo}' (spikes: {n_spikes_global}, no-spikes: {n_no_spikes_global}). Se usará el df agregado.")
            df_balanceado_final = df_agregado_pre_balanceo.sample(frac=1, random_state=42).reset_index(drop=True)


        if not df_balanceado_final.empty:
            df_balanceado_final = df_balanceado_final.sample(frac=1, random_state=42).reset_index(drop=True)
        else: 
            print(f"❌ ADVERTENCIA: df_balanceado_final está vacío para '{nombre_base_archivo}' antes del diagnóstico.")


    diagnosticar_dataset(df_balanceado_final, nombre_base_archivo, df_agregado_pre_balanceo)

    if not df_balanceado_final.empty:
        ruta_completa_csv = os.path.join("dataset_cerebelo", f"{nombre_base_archivo}_kan_ready.csv")
        ruta_ligera_csv = os.path.join("dataset_cerebelo", f"{nombre_base_archivo}_light.csv")
        df_balanceado_final.to_csv(ruta_completa_csv, index=False)
        print(f"✅ Dataset '{nombre_base_archivo}' guardado en '{ruta_completa_csv}' con {len(df_balanceado_final)} filas.")
        n_muestras_ligero = min(100_000, len(df_balanceado_final))
        if n_muestras_ligero > 0 :
            df_balanceado_final.sample(n=n_muestras_ligero, random_state=42).to_csv(ruta_ligera_csv, index=False)
            print(f"📁 Versión ligera '{nombre_base_archivo}_light.csv' guardada con {n_muestras_ligero} filas.")
    else:
        print(f"❌ ERROR: El DataFrame balanceado final para '{nombre_base_archivo}' está vacío. No se guardaron archivos.")

# --- Funciones de Simulación Específicas para cada Tipo de Célula ---

def simular_celula_lif( 
    nombre_base_archivo,
    parametros_lif, 
    duracion_simulacion_ms=1000, 
    numero_neuronas=100, 
    corriente_base_nA=0.8, 
    sigma_ruido_nA=0.3, 
    tau_sinaptico_ms=5.0, 
    porcentaje_heterogeneidad=0.05, 
    ratio_no_spike_to_spike=10, 
    max_no_spikes_neurona=15000,
    dt_simulacion_ms=0.1,
    peri_spike_window_ms=5.0, 
    peri_spike_sample_ratio=0.5
):
    start_scope()
    defaultclock.dt = dt_simulacion_ms * ms 

    eqs_lif = '''
    dv/dt = (-g_L*(v - EL_eq) + I_syn)/C : volt 
    dI_syn/dt = (-I_syn + I0_syn)/tau_syn + (sigma_syn/sqrt(tau_syn))*xi : amp
    g_L : siemens
    C : farad
    EL_eq : volt (shared) 
    I0_syn : amp (shared)
    sigma_syn : amp (shared)
    tau_syn : second (shared)
    V_th_lif : volt (shared)    
    V_res_lif : volt (shared)   
    '''
    
    namespace_completo_lif = {
        'EL_eq': parametros_lif["EL_lif"],
        'V_th_lif': parametros_lif["V_th_lif"], 
        'V_res_lif': parametros_lif["V_res_lif"]         
    }
    
    G = NeuronGroup(numero_neuronas, eqs_lif,
                    threshold='v > V_th_lif',    
                    reset='v = V_res_lif',       
                    method='euler', 
                    namespace=namespace_completo_lif 
                    )
    
    print(f"INFO ({nombre_base_archivo}): Aplicando {porcentaje_heterogeneidad*100:.1f}% de variabilidad a g_L y C...")
    for param_name in ['g_L', 'C']: 
        if param_name not in parametros_lif:
             raise ValueError(f"El parámetro '{param_name}' es esperado en 'parametros_lif' para la célula {nombre_base_archivo}")
        base_val = parametros_lif[param_name]
        setattr(G, param_name, base_val * (1 + porcentaje_heterogeneidad * (np.random.rand(numero_neuronas) - 0.5)))

    G.I0_syn = corriente_base_nA * nA
    G.sigma_syn = sigma_ruido_nA * nA 
    G.tau_syn = tau_sinaptico_ms * ms 
    
    G.v = parametros_lif["EL_lif"]
    G.I_syn = corriente_base_nA * nA

    monitor_estado = StateMonitor(G, ['v', 'I_syn'], record=True, dt=dt_simulacion_ms*ms) 
    monitor_spikes = SpikeMonitor(G)

    print(f"🚀 Iniciando simulación LIF para '{nombre_base_archivo}' por {duracion_simulacion_ms} ms (dt={dt_simulacion_ms}ms)...")
    run(duracion_simulacion_ms * ms)
    print("   ...simulación LIF completada.")

    if numero_neuronas > 0 and hasattr(monitor_estado, 'I_syn') and monitor_estado.I_syn.size > 0:
         array_corriente_entrada_global_nA = monitor_estado.I_syn[0] / nA
    else:
        print(f"WARN ({nombre_base_archivo}): No se pudo obtener I_syn del monitor para LIF. Usando corriente_base_nA.")
        num_pasos_tiempo = int(duracion_simulacion_ms / dt_simulacion_ms) 
        array_corriente_entrada_global_nA = np.full(num_pasos_tiempo, corriente_base_nA)

    procesar_y_guardar_datos_multi_neurona(
        nombre_base_archivo,
        monitor_estado,
        monitor_spikes,
        array_corriente_entrada_global_nA,
        dt_simulacion_ms, 
        ratio_no_spike_to_spike_per_neuron=ratio_no_spike_to_spike,
        max_no_spikes_absoluto_per_neuron=max_no_spikes_neurona,
        peri_spike_window_ms=peri_spike_window_ms,
        peri_spike_sample_ratio=peri_spike_sample_ratio
    )

def simular_purkinje_hh_mejorado( 
    nombre_base_archivo="purkinje_hh", 
    duracion_simulacion_ms=2000, 
    numero_neuronas=100, 
    corriente_media_uA_cm2=20.0, 
    corriente_std_uA_cm2=10.0,  
    frecuencia_ruido_Hz=0.2, 
    porcentaje_heterogeneidad=0.05,
    ratio_no_spike_to_spike=10,
    max_no_spikes_neurona=15000,
    peri_spike_window_ms=5.0,
    peri_spike_sample_ratio=0.5
):
    start_scope() 
    dt_hh_ms = 0.01 
    defaultclock.dt = dt_hh_ms * ms 

    print(f"INFO ({nombre_base_archivo}): Generando estímulo de corriente variable...")
    num_pasos_tiempo = int(duracion_simulacion_ms / (defaultclock.dt/ms))
    ruido_base_corriente = np.random.randn(num_pasos_tiempo)
    tamano_ventana_filtro = max(1, int(1.0 / (frecuencia_ruido_Hz * (defaultclock.dt/ms))) if frecuencia_ruido_Hz > 0 else 1)
    ruido_suavizado = np.convolve(ruido_base_corriente, np.ones(tamano_ventana_filtro)/tamano_ventana_filtro, mode='same') if tamano_ventana_filtro > 1 else ruido_base_corriente
    corriente_temporal_uA_cm2 = corriente_media_uA_cm2 + corriente_std_uA_cm2 * ruido_suavizado
    corriente_entrada_brian = TimedArray(corriente_temporal_uA_cm2 * uA / cm**2, dt=defaultclock.dt)

    eqs_purkinje = '''
    dv/dt = (I_ext(t) - gNa*(m**3)*h*(v - ENa) - gK*(n**4)*(v - EK) - gL*(v - EL)) / Cm : volt
    dm/dt = alpha_m*(1 - m) - beta_m*m : 1
    dh/dt = alpha_h*(1 - h) - beta_h*h : 1
    dn/dt = alpha_n*(1 - n) - beta_n*n : 1
    alpha_m = (0.1/mV) * (25*mV - v) / (exp((25*mV - v) / (10*mV)) - 1) / ms : Hz 
    beta_m = 4 * exp(-v / (18*mV)) / ms : Hz
    alpha_h = 0.07 * exp(-v / (20*mV)) / ms : Hz
    beta_h = 1 / (exp((30*mV - v) / (10*mV)) + 1) / ms : Hz 
    alpha_n = (0.01/mV) * (10*mV - v) / (exp((10*mV - v) / (10*mV)) - 1) / ms : Hz
    beta_n = 0.125 * exp(-v / (80*mV)) / ms : Hz
    gL : siemens/meter**2 
    '''
    G = NeuronGroup(numero_neuronas, eqs_purkinje, threshold='v > -40*mV', reset='v = -65*mV', 
                    method='exponential_euler',
                    namespace={
                        'Cm': 1*uF/cm**2, 'gNa': 120*msiemens/cm**2, 'gK': 36*msiemens/cm**2,
                        'ENa': 50*mV, 'EK': -77*mV, 'EL': -54.4*mV, 'I_ext': corriente_entrada_brian
                    })

    print(f"INFO ({nombre_base_archivo}): Aplicando {porcentaje_heterogeneidad*100:.1f}% de variabilidad a gL...")
    gL_base = 0.3*msiemens/cm**2
    G.gL = gL_base * (1 + porcentaje_heterogeneidad * (np.random.rand(numero_neuronas) - 0.5))
    G.v = -65*mV; G.m = 0.05; G.h = 0.6; G.n = 0.32

    monitor_estado = StateMonitor(G, ['v'], record=True, dt=dt_hh_ms*ms) 
    monitor_spikes = SpikeMonitor(G)

    print(f"🚀 Iniciando simulación HH para '{nombre_base_archivo}' por {duracion_simulacion_ms} ms (dt={dt_hh_ms}ms)...")
    run(duracion_simulacion_ms * ms)
    print("   ...simulación HH completada.")
    
    array_corriente_entrada_global_nA_equivalente = corriente_temporal_uA_cm2 

    procesar_y_guardar_datos_multi_neurona(
        nombre_base_archivo,
        monitor_estado,
        monitor_spikes,
        array_corriente_entrada_global_nA_equivalente, 
        dt_hh_ms, 
        ratio_no_spike_to_spike_per_neuron=ratio_no_spike_to_spike,
        max_no_spikes_absoluto_per_neuron=max_no_spikes_neurona,
        peri_spike_window_ms=peri_spike_window_ms,
        peri_spike_sample_ratio=peri_spike_sample_ratio
    )


# --- Función Principal para Orquestar Todas las Simulaciones ---
def simular_todas_las_celulas():
    print("===== INICIANDO SIMULACIÓN DE TODAS LAS CÉLULAS DEL CEREBELO =====")
    
    dt_lif_general_ms = 0.1 

    # Célula Granular (LIF)
    params_granule_lif = {
        "g_L": 5*nS, "C": 100*pF, "EL_lif": -70*mV, "V_th_lif": -50*mV, "V_res_lif": -65*mV 
    }
    simular_celula_lif("granule_lif", params_granule_lif, numero_neuronas=100, duracion_simulacion_ms=1500,
                        corriente_base_nA=0.25, sigma_ruido_nA=0.1, tau_sinaptico_ms=5.0, 
                        porcentaje_heterogeneidad=0.1, dt_simulacion_ms=dt_lif_general_ms,
                        peri_spike_window_ms=4.0, peri_spike_sample_ratio=0.6) # 60% de no-spikes vendrán de la rampa

    # Célula de Golgi (LIF)
    params_golgi_lif = {
        "g_L": 10*nS, "C": 200*pF, "EL_lif": -68*mV, "V_th_lif": -52*mV, "V_res_lif": -65*mV
    }
    simular_celula_lif("golgi_lif", params_golgi_lif, numero_neuronas=50, duracion_simulacion_ms=2000,
                        corriente_base_nA=0.6, sigma_ruido_nA=0.2, tau_sinaptico_ms=8.0, 
                        porcentaje_heterogeneidad=0.05, dt_simulacion_ms=dt_lif_general_ms,
                        peri_spike_window_ms=5.0, peri_spike_sample_ratio=0.5)

    # Célula en Cesta (Basket) (LIF)
    params_basket_lif = {
       "g_L": 12*nS, "C": 150*pF, "EL_lif": -67*mV, "V_th_lif": -50*mV, "V_res_lif": -65*mV
    }
    simular_celula_lif("basket_lif", params_basket_lif, numero_neuronas=50, duracion_simulacion_ms=1500,
                        corriente_base_nA=0.7, sigma_ruido_nA=0.25, tau_sinaptico_ms=4.0, 
                        porcentaje_heterogeneidad=0.05, dt_simulacion_ms=dt_lif_general_ms,
                        peri_spike_window_ms=3.0, peri_spike_sample_ratio=0.7)

    # Célula Estrellada (Stellate) (LIF)
    params_stellate_lif = {
        "g_L": 10*nS, "C": 180*pF, "EL_lif": -68*mV, "V_th_lif": -52*mV, "V_res_lif": -66*mV
    }
    simular_celula_lif("stellate_lif", params_stellate_lif, numero_neuronas=50, duracion_simulacion_ms=1800,
                        corriente_base_nA=0.5, sigma_ruido_nA=0.15, tau_sinaptico_ms=6.0, 
                        porcentaje_heterogeneidad=0.05, dt_simulacion_ms=dt_lif_general_ms)

    # Célula del Núcleo Profundo (LIF)
    params_nuclei_lif = {
       "g_L": 15*nS, "C": 300*pF, "EL_lif": -65*mV, "V_th_lif": -50*mV, "V_res_lif": -66*mV
    }
    simular_celula_lif("deep_nuclei_lif", params_nuclei_lif, numero_neuronas=30, duracion_simulacion_ms=2500,
                        corriente_base_nA=0.8, sigma_ruido_nA=0.3, tau_sinaptico_ms=10.0, 
                        porcentaje_heterogeneidad=0.05, dt_simulacion_ms=dt_lif_general_ms)
    
    # Célula de Purkinje (Hodgkin-Huxley Mejorado)
    simular_purkinje_hh_mejorado(
        nombre_base_archivo="purkinje_hh_dinamico", 
        duracion_simulacion_ms=2000, 
        numero_neuronas=50, 
        corriente_media_uA_cm2=20.0, 
        corriente_std_uA_cm2=10.0, 
        frecuencia_ruido_Hz=0.2, 
        porcentaje_heterogeneidad=0.05,
        ratio_no_spike_to_spike=8,
        max_no_spikes_neurona=12000,
        peri_spike_window_ms=10.0, # Purkinje tiene spikes más anchos
        peri_spike_sample_ratio=0.4
    )

    # Fibras Musgosas (LIF)
    params_mossy = {
        "g_L": 10*nS, "C": 150*pF, "EL_lif": -70*mV, 
        "V_th_lif": -50*mV, "V_res_lif": -70*mV 
    }
    simular_celula_lif(
        "mossy_fiber", 
        parametros_lif=params_mossy, 
        duracion_simulacion_ms=1500, numero_neuronas=80,
        corriente_base_nA=0.6, sigma_ruido_nA=0.4, tau_sinaptico_ms=3.0, 
        porcentaje_heterogeneidad=0.07,
        ratio_no_spike_to_spike=5, max_no_spikes_neurona=8000,
        dt_simulacion_ms=dt_lif_general_ms
    )

    # Fibras Trepadoras (LIF)
    params_climbing = {
        "g_L": 8*nS, "C": 100*pF, "EL_lif": -65*mV, 
        "V_th_lif": -45*mV, "V_res_lif": -68*mV 
    }
    simular_celula_lif(
        "climbing_fiber", 
        parametros_lif=params_climbing, 
        duracion_simulacion_ms=3000, numero_neuronas=30, 
        corriente_base_nA=0.1, sigma_ruido_nA=0.05, tau_sinaptico_ms=10.0, 
        porcentaje_heterogeneidad=0.05,
        ratio_no_spike_to_spike=30, max_no_spikes_neurona=20000,
        dt_simulacion_ms=dt_lif_general_ms
    )

    print("\n===== SIMULACIÓN DE TODAS LAS CÉLULAS COMPLETADA =====")

if __name__ == "__main__":
    simular_todas_las_celulas()
