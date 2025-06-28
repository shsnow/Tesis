# Cerebellar_Class.py

import os
import torch
import joblib
import pandas as pd
import numpy as np
from kan import KAN

# Detectar dispositivo (GPU si está disponible)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Rutas base
BASE_MODEL_DIR = "models_cerebelo"
BASE_DATA_DIR = "dataset_cerebelo"
os.makedirs(BASE_MODEL_DIR, exist_ok=True)
os.makedirs(BASE_DATA_DIR, exist_ok=True)

class NeuronaCerebelarKAN:
    """
    Interfaz para cargar y usar un modelo KAN entrenado
    para emular la dinámica de disparo de una célula cerebelosa.
    """

    def __init__(self,
                 nombre_celula: str,
                 ruta_base_datos: str = BASE_DATA_DIR,
                 ruta_base_modelos: str = BASE_MODEL_DIR,
                 columnas_features: list = None,
                 columna_target: str = "spike"):
        self.nombre_celula = nombre_celula
        # Columnas de entrada que espera el modelo
        self.columnas_features = columnas_features or ["time_ms", "voltage_mV", "input_current_nA"]
        self.columna_target = columna_target

        # Elige dataset completo si existe, si no el ligero
        full = os.path.join(ruta_base_datos, f"{nombre_celula}_kan_ready.csv")
        light = os.path.join(ruta_base_datos, f"{nombre_celula}_light.csv")
        self.ruta_datos_csv = full if os.path.exists(full) else light

        # Rutas de modelo y scaler
        self.ruta_modelo = os.path.join(ruta_base_modelos, f"kan_model_{nombre_celula}.pt")
        self.ruta_scaler = os.path.join(ruta_base_modelos, f"scaler_{nombre_celula}.joblib")

        # Configuración KAN por defecto (ajústala si cambiaste arquitectura)
        self.config_kan = {
            "width": [len(self.columnas_features), 32, 16, 1],
            "grid": 5,
            "k": 3,
            "seed": 42
        }

        # Será poblado por cargar_modelo()
        self.scaler_cargado = None
        self.modelo_kan_cargado = None

    def cargar_modelo(self) -> bool:
        """
        Carga scaler y modelo desde disco.
        Devuelve True si tuvo éxito.
        """
        if not os.path.exists(self.ruta_scaler):
            print(f"❌ No se encontró scaler en {self.ruta_scaler}")
            return False
        if not os.path.exists(self.ruta_modelo):
            print(f"❌ No se encontró modelo en {self.ruta_modelo}")
            return False

        # 1. Cargar scaler
        self.scaler_cargado = joblib.load(self.ruta_scaler)

        # 2. Instanciar y cargar modelo
        try:
            m = KAN(width=self.config_kan["width"],
                    grid=self.config_kan["grid"],
                    k=self.config_kan["k"],
                    seed=self.config_kan["seed"])
            state = torch.load(self.ruta_modelo, map_location=DEVICE)
            m.load_state_dict(state)
            m.to(DEVICE).eval()
            self.modelo_kan_cargado = m
        except Exception as e:
            print(f"❌ Error al cargar KAN: {e}")
            return False

        print(f"✅ Modelo y scaler cargados para {self.nombre_celula} en {DEVICE}")
        return True

    def predecir(self, df: pd.DataFrame):
        """
        Recibe un DataFrame con las columnas de entrada y devuelve
        (probs, preds) como arrays de forma (N,1).
        """
        if self.scaler_cargado is None or self.modelo_kan_cargado is None:
            print("❌ Llama primero a cargar_modelo()")
            return None, None

        # Verificar columnas
        for col in self.columnas_features:
            if col not in df.columns:
                print(f"❌ Falta columna '{col}' en el DataFrame")
                return None, None

        # Escalar y convertir a tensor
        X = df[self.columnas_features].values.astype(np.float32)
        Xs = self.scaler_cargado.transform(X)
        Xt = torch.tensor(Xs, dtype=torch.float32, device=DEVICE)

        # Inferencia
        with torch.no_grad():
            logits = self.modelo_kan_cargado(Xt)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

        return probs, preds

    def plot_splines(self, beta: float = 10.0, **kwargs):
        """
        Genera y devuelve la figura matplotlib con las
        funciones spline aprendidas (si está disponible).
        """
        if self.modelo_kan_cargado is None:
            print("❌ Modelo no cargado")
            return None
        try:
            fig = self.modelo_kan_cargado.plot(beta=beta, **kwargs)
            return fig
        except Exception as e:
            print(f"❌ No se pudo plotear splines: {e}")
            return None

    def info(self):
        """
        Imprime un resumen de rutas y configuración.
        """
        print(f"--- NeuronaCerebelarKAN: {self.nombre_celula} ---")
        print(f" CSV datos:    {self.ruta_datos_csv}")
        print(f" Modelo KAN:   {self.ruta_modelo}")
        print(f" Scaler:       {self.ruta_scaler}")
        print(f" Features:     {self.columnas_features}")
        print(f" Arquitectura: {self.config_kan}")
        print(f" Dispositivo:  {DEVICE}")
