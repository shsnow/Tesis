import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import os
import joblib # Para guardar/cargar el scaler

# Asegúrate de que la biblioteca KAN esté instalada
try:
    from kan import KAN
except ImportError:
    print("ADVERTENCIA: La biblioteca 'pykan' no está instalada. Por favor, instálala: pip install pykan")
    print("O si estás usando una versión específica: pip install git+https://github.com/KindXiaoming/pykan.git")
    # Podrías querer que el script termine aquí si KAN es esencial.
    # exit() 

# Configuración global
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_MODEL_DIR_LIB = "models_cerebelo" # Donde se guardan/cargan modelos KAN y scalers
BASE_DATA_DIR_LIB = "dataset_cerebelo"   # Donde se esperan los CSVs generados

os.makedirs(BASE_MODEL_DIR_LIB, exist_ok=True)

# --- Clase Principal de la "Librería" ---
class NeuronaCerebelarKAN:
    def __init__(self, nombre_celula, ruta_base_datos=BASE_DATA_DIR_LIB, ruta_base_modelos=BASE_MODEL_DIR_LIB, columnas_features=None, columna_target="spike"):
        self.nombre_celula = nombre_celula
        self.columnas_features = columnas_features if columnas_features is not None else ["time_ms", "voltage_mV", "input_current_nA"]
        self.columna_target = columna_target
        
        # Rutas dinámicas basadas en el nombre de la célula
        self.ruta_datos_csv = os.path.join(ruta_base_datos, f"{self.nombre_celula}_light.csv") # Por defecto usa _light para entrenamiento rápido
        self.ruta_datos_completos_csv = os.path.join(ruta_base_datos, f"{self.nombre_celula}_kan_ready.csv")
        
        self.ruta_modelo_kan = os.path.join(ruta_base_modelos, f"kan_model_{self.nombre_celula}.pt")
        self.ruta_plot_roc = os.path.join(ruta_base_modelos, f"roc_kan_{self.nombre_celula}.png")
        self.ruta_scaler = os.path.join(ruta_base_modelos, f"scaler_{self.nombre_celula}.joblib")
        
        # Configuraciones por defecto, pueden ser sobrescritas
        self.config_kan = {
            "width": [len(self.columnas_features), 32, 16, 1],
            "grid_size": 5,
            "k": 3,
            "seed": 42
        }
        self.config_entrenamiento = {
            "num_phases": 3,
            "epochs_per_phase": 70, 
            "learning_rate_schedule": {0: 1e-3, 1: 5e-4, 2: 1e-4},
            "max_grad_norm": 1.0,
            "weight_decay": 1e-5
        }
        
        self.modelo_kan_cargado = None
        self.scaler_cargado = None
        print(f"Instancia de NeuronaCerebelarKAN creada para: {self.nombre_celula}. Modelo KAN se guardará/cargará de: {self.ruta_modelo_kan}")

    def configurar_kan_personalizado(self, width_hidden_layers=None, grid_size=None, k=None, seed=None):
        """Permite personalizar la arquitectura KAN. width_hidden_layers no incluye la capa de entrada/salida."""
        if width_hidden_layers: 
            self.config_kan["width"] = [len(self.columnas_features)] + width_hidden_layers + [1]
        if grid_size: self.config_kan["grid_size"] = grid_size
        if k: self.config_kan["k"] = k
        if seed: self.config_kan["seed"] = seed
        print(f"INFO ({self.nombre_celula}): Configuración KAN actualizada: {self.config_kan}")

    def configurar_entrenamiento_personalizado(self, num_phases=None, epochs_per_phase=None, lr_schedule=None, max_grad_norm=None, weight_decay=None):
        if num_phases: self.config_entrenamiento["num_phases"] = num_phases
        if epochs_per_phase: self.config_entrenamiento["epochs_per_phase"] = epochs_per_phase
        if lr_schedule: self.config_entrenamiento["learning_rate_schedule"] = lr_schedule
        if max_grad_norm: self.config_entrenamiento["max_grad_norm"] = max_grad_norm
        if weight_decay: self.config_entrenamiento["weight_decay"] = weight_decay
        print(f"INFO ({self.nombre_celula}): Configuración de Entrenamiento actualizada: {self.config_entrenamiento}")

    def entrenar_modelo(self, usar_datos_completos=False, forzar_reentrenamiento=False):
        print(f"INFO ({self.nombre_celula}): Solicitud de entrenamiento...")
        ruta_datos_a_usar = self.ruta_datos_completos_csv if usar_datos_completos else self.ruta_datos_csv

        if not forzar_reentrenamiento and os.path.exists(self.ruta_modelo_kan) and os.path.exists(self.ruta_scaler):
            print(f"INFO ({self.nombre_celula}): Modelo KAN y scaler ya existen. No se reentrenará.")
            return self.cargar_modelo() # Cargar y confirmar

        print(f"INFO ({self.nombre_celula}): Procediendo con el entrenamiento usando {ruta_datos_a_usar}.")
        auc_final, _ = self._ejecutar_entrenamiento_evaluacion(ruta_datos_a_usar)
        
        if auc_final is not None:
            print(f"INFO ({self.nombre_celula}): Entrenamiento completado. Modelo guardado en {self.ruta_modelo_kan} con AUC final {auc_final:.4f}")
            return self.cargar_modelo() # Cargar el modelo recién entrenado
        else:
            print(f"ERROR ({self.nombre_celula}): Entrenamiento falló.")
            return False

    def cargar_modelo(self):
        print(f"INFO ({self.nombre_celula}): Intentando cargar modelo KAN desde {self.ruta_modelo_kan} y scaler desde {self.ruta_scaler}...")
        if not os.path.exists(self.ruta_modelo_kan):
            print(f"❌ ERROR: Archivo de modelo KAN no encontrado en {self.ruta_modelo_kan}")
            self.modelo_kan_cargado = None
            return False
        if not os.path.exists(self.ruta_scaler):
            print(f"❌ ERROR: Archivo de scaler no encontrado en {self.ruta_scaler}")
            self.scaler_cargado = None
            return False

        try:
            # Para cargar, la arquitectura debe coincidir. width se basa en config_kan.
            self.modelo_kan_cargado = KAN(
                width=self.config_kan["width"], 
                grid=self.config_kan["grid_size"], 
                k=self.config_kan["k"], 
                seed=self.config_kan.get("seed", 42)
            ).to(DEVICE)
            self.modelo_kan_cargado.load_state_dict(torch.load(self.ruta_modelo_kan, map_location=DEVICE))
            self.modelo_kan_cargado.eval()
            self.scaler_cargado = joblib.load(self.ruta_scaler)
            print(f"✅ Modelo KAN y scaler para {self.nombre_celula} cargados exitosamente.")
            return True
        except Exception as e:
            print(f"❌ ERROR al cargar modelo KAN o scaler para {self.nombre_celula}: {e}")
            self.modelo_kan_cargado = None
            self.scaler_cargado = None
            return False

    def predecir(self, datos_entrada_nuevos_df):
        """
        Realiza predicciones usando el modelo KAN cargado.
        Args:
            datos_entrada_nuevos_df (pd.DataFrame): Nuevos datos con las mismas
                                                   columnas de características que el entrenamiento.
        Returns:
            tuple (np.ndarray, np.ndarray) or (None, None): Probabilidades y Predicciones binarias.
        """
        if self.modelo_kan_cargado is None or self.scaler_cargado is None:
            print(f"INFO ({self.nombre_celula}): Modelo o scaler no cargado. Intentando cargar...")
            if not self.cargar_modelo():
                print(f"❌ ERROR ({self.nombre_celula}): No se pudo cargar el modelo o scaler para predicción.")
                return None, None
        
        try:
            X_nuevos_raw = datos_entrada_nuevos_df[self.columnas_features].values.astype(np.float32)
        except KeyError as e:
             print(f"❌ ERROR ({self.nombre_celula}): Faltan columnas en datos_entrada_nuevos_df: {e}")
             return None, None
        except Exception as e_general:
            print(f"❌ ERROR ({self.nombre_celula}): Problema con datos de entrada: {e_general}")
            return None, None


        X_nuevos_scaled = self.scaler_cargado.transform(X_nuevos_raw)
        X_nuevos_tensor = torch.tensor(X_nuevos_scaled, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            logits = self.modelo_kan_cargado(X_nuevos_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
        
        return probs, preds

    def _ejecutar_entrenamiento_evaluacion(self, ruta_archivo_datos):
        # Esta función es una adaptación de la lógica de entrenamiento original
        # Se mantiene "privada" (prefijo _) ya que es llamada por self.entrenar_modelo()
        
        print(f"INFO ({self.nombre_celula}): Cargando datos desde {ruta_archivo_datos} para entrenamiento...")
        try:
            df = pd.read_csv(ruta_archivo_datos)
        except FileNotFoundError:
            print(f"❌ ERROR: Archivo no encontrado: {ruta_archivo_datos}.")
            return None, None

        try:
            X_raw = df[self.columnas_features].values.astype(np.float32)
            y_raw = df[self.columna_target].values.astype(np.float32)
        except KeyError as e:
            print(f"❌ ERROR: Columnas no encontradas en {ruta_archivo_datos}: {e}.")
            return None, None

        # Re-instanciar scaler para este entrenamiento específico
        scaler_entrenamiento = StandardScaler()
        X_scaled = scaler_entrenamiento.fit_transform(X_raw)
        joblib.dump(scaler_entrenamiento, self.ruta_scaler) # Guardar el scaler ajustado
        print(f"INFO ({self.nombre_celula}): Scaler ajustado y guardado en {self.ruta_scaler}")
        self.scaler_cargado = scaler_entrenamiento # Actualizar el scaler de la instancia

        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
            X_scaled, y_raw, test_size=0.2, stratify=y_raw, random_state=self.config_kan.get("seed", 42)
        )

        X_train = torch.tensor(X_train_np, dtype=torch.float32).to(DEVICE)
        y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        X_test = torch.tensor(X_test_np, dtype=torch.float32).to(DEVICE)
        y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1).to(DEVICE)

        if X_train.shape[0] == 0 or X_test.shape[0] == 0 or len(torch.unique(y_train)) < 2 or len(torch.unique(y_test)) < 2:
            print(f"❌ ERROR ({self.nombre_celula}): Datos insuficientes o solo una clase después del split. Abortando entrenamiento.")
            return None, None
            
        print(f"INFO ({self.nombre_celula}): Creando nueva instancia de modelo KAN para entrenamiento...")
        # Usar self.config_kan["width"] que ya está ajustado para el número de features
        modelo = KAN(
            width=self.config_kan["width"], 
            grid=self.config_kan["grid_size"], 
            k=self.config_kan["k"], 
            seed=self.config_kan.get("seed", 42)
        ).to(DEVICE)
        self.modelo_kan_cargado = modelo # Asignar a la instancia para posible uso si el entrenamiento es largo

        criterion = nn.BCEWithLogitsLoss()
        best_model_cell_auc = 0.0

        for phase in range(self.config_entrenamiento["num_phases"]):
            print(f"\n🚀 Fase de Entrenamiento {phase + 1}/{self.config_entrenamiento['num_phases']} para {self.nombre_celula}")
            current_lr = self.config_entrenamiento["learning_rate_schedule"].get(phase, 1e-4)
            optimizer = optim.AdamW(modelo.parameters(), lr=current_lr, weight_decay=self.config_entrenamiento.get("weight_decay", 1e-5))
            print(f"   Tasa de aprendizaje: {current_lr}")

            for epoch in range(self.config_entrenamiento["epochs_per_phase"]):
                modelo.train()
                optimizer.zero_grad()
                logits = modelo(X_train)
                loss = criterion(logits, y_train)
                if torch.isnan(loss):
                    print(f"❌ ERROR ({self.nombre_celula}, Fase {phase+1}, Epoch {epoch+1}): Loss es NaN.")
                    return None, None
                loss.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), self.config_entrenamiento.get("max_grad_norm", 1.0))
                optimizer.step()
                if (epoch + 1) % 20 == 0: 
                    print(f"   Epoch {epoch+1}/{self.config_entrenamiento['epochs_per_phase']}: Loss = {loss.item():.6f}")

            modelo.eval()
            with torch.no_grad():
                logits_test_phase = modelo(X_test)
                probs_test_phase = torch.sigmoid(logits_test_phase)
                try:
                    auc_test_phase = roc_auc_score(y_test.cpu().numpy(), probs_test_phase.cpu().numpy())
                    print(f"   AUC en Test Set (Fase {phase+1}): {auc_test_phase:.4f}")
                    if auc_test_phase > best_model_cell_auc:
                        best_model_cell_auc = auc_test_phase
                        torch.save(modelo.state_dict(), self.ruta_modelo_kan)
                        print(f"   🎉 Nuevo mejor modelo para {self.nombre_celula} guardado con AUC: {best_model_cell_auc:.4f} en {self.ruta_modelo_kan}")
                except ValueError as e:
                     print(f"   ⚠️ No se pudo calcular AUC en Test Set (Fase {phase+1}) para {self.nombre_celula}: {e}")
            
            if hasattr(modelo, 'prune'):
                print(f"   Podando modelo para {self.nombre_celula}...")
                try:
                    modelo.prune()
                    print("   Modelo podado.")
                except Exception as e_prune:
                    print(f"   ⚠️ Error durante la poda para {self.nombre_celula}: {e_prune}")
        
        # Evaluación final del modelo entrenado en esta sesión
        print(f"\n🔬 Evaluación Final Post-Entrenamiento para {self.nombre_celula}")
        modelo.eval()
        with torch.no_grad():
            logits_test = modelo(X_test)
            probs = torch.sigmoid(logits_test)
            preds = (probs > 0.5).float()
            y_test_cpu = y_test.cpu().numpy()
            probs_cpu = probs.cpu().numpy()
            preds_cpu = preds.cpu().numpy()

            if len(np.unique(y_test_cpu)) < 2:
                print(f"❌ ERROR ({self.nombre_celula}): Test set final solo tiene una clase.")
                return best_model_cell_auc, self.ruta_modelo_kan

            final_auc = roc_auc_score(y_test_cpu, probs_cpu)
            final_acc = accuracy_score(y_test_cpu, preds_cpu)
            print(f"\nRESULTADOS DE EVALUACIÓN PARA {self.nombre_celula.upper()}: AUC={final_auc:.4f}, Acc={final_acc:.4f}")
            print(classification_report(y_test_cpu, preds_cpu, digits=4, zero_division=0))
            
            fpr, tpr, _ = roc_curve(y_test_cpu, probs_cpu)
            plt.figure(figsize=(8, 6)) 
            plt.plot(fpr, tpr, label=f"KAN {self.nombre_celula.replace('_', ' ').title()} (AUC = {final_auc:.4f})")
            plt.plot([0, 1], [0, 1], '--', color='dimgray')
            plt.title(f"Curva ROC - KAN {self.nombre_celula.replace('_', ' ').title()}")
            plt.legend(loc='lower right'); plt.grid(True); plt.tight_layout()
            plt.savefig(self.ruta_plot_roc); plt.close()
            print(f"📈 Curva ROC guardada: {self.ruta_plot_roc}")
        
        return final_auc, self.ruta_modelo_kan


# --- Bloque Principal para demostrar uso (si se ejecuta este archivo directamente) ---
if __name__ == '__main__':
    print(f"Ejecutando {__file__} como script principal.")
    print("Este script define la clase NeuronaCerebelarKAN y una función de entrenamiento.")
    print("Para usarla como librería, importa NeuronaCerebelarKAN en otro script.")
    print("\nEjemplo de cómo se podría usar (ver 'ejemplo_uso_libreria.py' para un ejemplo completo):")

    # Ejemplo básico: Instanciar y entrenar una célula (si los datos existen)
    FEATURES_LIF = ["time_ms", "voltage_mV", "input_current_nA"]
    config_granule = {
        "nombre": "granule_lif", 
        "archivo": "granule_lif_light.csv", 
        "features": FEATURES_LIF
    }
    # Corregir el NameError usando la variable global correcta
    ruta_datos_granule = os.path.join(BASE_DATA_DIR_LIB, config_granule["archivo"])

    if os.path.exists(ruta_datos_granule):
        print(f"\n--- DEMO: Entrenando Célula Granular (LIF) ---")
        granule_model = NeuronaCerebelarKAN(
            nombre_celula=config_granule["nombre"],
            # Pasar la ruta completa al constructor de la clase
            ruta_base_datos=os.path.dirname(ruta_datos_granule), # Pasar el directorio base
            columnas_features=config_granule["features"]
        )
        # El constructor ya forma self.ruta_datos_csv, pero para ser explícitos en la demo:
        granule_model.ruta_datos_csv = ruta_datos_granule 


        # Para la demo, reducimos las épocas
        granule_model.configurar_entrenamiento_personalizado(epochs_per_phase=10) 
        granule_model.entrenar_modelo(forzar_reentrenamiento=True) # Usar el método renombrado

        if granule_model.modelo_kan_cargado:
            print("\n--- DEMO: Realizando una predicción de prueba con el modelo de Granule ---")
            datos_dummy_raw = np.random.rand(5, len(FEATURES_LIF)).astype(np.float32)
            datos_dummy_raw[:,0] *= 1000 
            datos_dummy_raw[:,1] = (datos_dummy_raw[:,1] * -30) -40 
            datos_dummy_raw[:,2] *= 0.5 

            df_dummy_pred = pd.DataFrame(datos_dummy_raw, columns=FEATURES_LIF)
            
            print("Datos de entrada dummy para predicción:")
            print(df_dummy_pred)
            
            probs, preds = granule_model.predecir(df_dummy_pred)
            if probs is not None:
                print("\nResultados de la predicción dummy:")
                for i in range(len(probs)):
                    print(f"  Muestra {i+1}: Probabilidad(spike) = {probs[i][0]:.4f}, Predicción = {preds[i][0]}")
    else:
        print(f"\nADVERTENCIA PARA DEMO: No se encontró el archivo de datos {ruta_datos_granule}. No se ejecutará la demo de entrenamiento.")

