# preprocesador_csv.py
import pandas as pd
import numpy as np
import os

def cargar_y_procesar_csv(path_csv, ventana_ms=20, paso_ms=1, predecir='spike'):
    """
    Procesa un CSV de neurona cerebelosa y genera X (features) e y (targets) para entrenamiento de KAN.

    Argumentos:
    - path_csv: ruta al archivo CSV (debe contener columnas como 'voltage_mV', 'input_current_nA', 'spike')
    - ventana_ms: largo de la ventana de entrada (en muestras)
    - paso_ms: paso entre ventanas
    - predecir: columna a predecir ('spike', 'voltage_mV', etc.)

    Retorna:
    - X: np.ndarray [num_ejemplos, ventana * num_features]
    - y: np.ndarray [num_ejemplos]
    """
    df = pd.read_csv(path_csv)

    # Validar columnas disponibles
    columnas_validas = ['voltage_mV', 'input_current_nA', 'spike']
    features = [c for c in columnas_validas if c in df.columns]
    if predecir not in df.columns:
        raise ValueError(f"La columna objetivo '{predecir}' no existe en {path_csv}")

    datos_X = []
    datos_y = []
    long = len(df)
    ventana = int(ventana_ms)
    paso = int(paso_ms)

    for i in range(0, long - ventana, paso):
        ventana_X = df[features].iloc[i:i+ventana].values.flatten()
        target_y = df[predecir].iloc[i+ventana]  # predecimos un paso adelante
        datos_X.append(ventana_X)
        datos_y.append(target_y)

    X = np.array(datos_X)
    y = np.array(datos_y)
    return X, y

# Ejemplo de uso interactivo
if __name__ == '__main__':
    nombre_csv = "dataset/granule_kan_ready.csv"
    if os.path.exists(nombre_csv):
        X, y = cargar_y_procesar_csv(nombre_csv, ventana_ms=20, paso_ms=1, predecir='spike')
        print("X shape:", X.shape)
        print("y shape:", y.shape)
        print("y ejemplo:", y[:10])
        np.savez("granule_kan_dataset.npz", X=X, y=y)
        print("✅ Dataset guardado como .npz listo para KAN")
    else:
        print(f"❌ No se encontró el archivo {nombre_csv}")
