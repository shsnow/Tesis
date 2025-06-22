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
import joblib # 🌟 AÑADIDO: Para guardar el scaler

# Asegúrate de que la biblioteca KAN (por ejemplo, pykan de KindXiaoming) esté instalada.
# pip install pykan
from kan import KAN

# ✅ 0. Configuración Global
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")
os.makedirs("models_cerebelo", exist_ok=True) # Carpeta para guardar modelos y gráficos

# --- Configuración de KAN y Entrenamiento (Común para todos los modelos por ahora) ---
KAN_WIDTH = [3, 32, 16, 1]  # [num_features, hidden1, hidden2, ..., num_outputs]
KAN_GRID_SIZE = 5
KAN_K = 3
KAN_SEED = 42

NUM_PHASES = 3
EPOCHS_PER_PHASE = 70 
LEARNING_RATE_SCHEDULE = {0: 1e-3, 1: 5e-4, 2: 1e-4}
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 1e-5

FEATURE_COLUMNS_DEFAULT = ["time_ms", "voltage_mV", "input_current_nA"]
TARGET_COLUMN = "spike"

def train_and_evaluate_kan_for_cell(
    cell_name, 
    data_file_path, 
    feature_columns,
    target_column,
    kan_width,
    kan_grid_size,
    kan_k,
    kan_seed,
    num_phases,
    epochs_per_phase,
    learning_rate_schedule,
    max_grad_norm,
    weight_decay
    ):
    """
    Carga datos, entrena un modelo KAN y lo evalúa para un tipo de célula específico.
    """
    print(f"\n\n===== INICIANDO PROCESO PARA CÉLULA: {cell_name.upper()} =====")
    
    # ✅ 1. Cargar y preparar datos
    print(f"INFO: Cargando datos desde {data_file_path}...")
    try:
        df = pd.read_csv(data_file_path)
    except FileNotFoundError:
        print(f"❌ ERROR: Archivo no encontrado: {data_file_path}. Omitiendo célula {cell_name}.")
        return

    print(f"INFO: Usando características: {feature_columns} y objetivo: {target_column}")
    try:
        # Asegurarse de que el nombre de la columna de corriente sea correcto
        if "input_current_uA_cm2" in df.columns and "input_current_uA_cm2" in feature_columns:
            print("   (Usando input_current_uA_cm2 para Purkinje)")
        elif "input_current_nA" in df.columns and "input_current_nA" in feature_columns:
            print("   (Usando input_current_nA para LIF)")
        
        X_raw = df[feature_columns].values.astype(np.float32)
        y_raw = df[target_column].values.astype(np.float32)
    except KeyError as e:
        print(f"❌ ERROR: Una o más columnas no encontradas en {data_file_path}: {e}. Omitiendo célula {cell_name}.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 🌟 AÑADIDO: Guardar el scaler ajustado
    scaler_path = os.path.join("models_cerebelo", f"scaler_{cell_name}.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"   Scaler guardado en: {scaler_path}")


    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_scaled, y_raw, test_size=0.2, stratify=y_raw, random_state=kan_seed
    )

    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1).to(device)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1).to(device)

    print(f"Shapes -> X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print(f"❌ ERROR: No hay suficientes datos para {cell_name} después del split. Omitiendo.")
        return
    if len(torch.unique(y_train)) < 2 or len(torch.unique(y_test)) < 2:
        print(f"❌ ERROR: {cell_name} no tiene ambas clases en train o test set después del split. Omitiendo.")
        return


    # ✅ 2. Crear y configurar el modelo KAN (Nueva instancia para cada célula)
    print(f"INFO ({cell_name}): Creando modelo KAN...")
    current_kan_width = [X_train.shape[1]] + kan_width[1:]
    
    model = KAN(
        width=current_kan_width, 
        grid=kan_grid_size,                  
        k=kan_k,                             
        seed=kan_seed 
    ).to(device)
    print(f"   Número de parámetros KAN: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    criterion = nn.BCEWithLogitsLoss()
    
    # ✅ 3. Bucle de Entrenamiento
    best_model_cell_path = os.path.join("models_cerebelo", f"kan_model_{cell_name}.pt")
    best_model_cell_auc = 0.0

    for phase in range(num_phases):
        print(f"\n🚀 Fase de Entrenamiento {phase + 1}/{num_phases} para {cell_name}")
        
        current_lr = learning_rate_schedule.get(phase, 1e-4)
        optimizer = optim.AdamW(model.parameters(), lr=current_lr, weight_decay=weight_decay)
        print(f"   Tasa de aprendizaje: {current_lr}")

        for epoch in range(epochs_per_phase):
            model.train()
            optimizer.zero_grad()
            
            logits = model(X_train)
            loss = criterion(logits, y_train)

            if torch.isnan(loss):
                print(f"❌ ERROR ({cell_name}, Fase {phase+1}, Epoch {epoch+1}): Loss es NaN. Deteniendo entrenamiento para esta célula.")
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            if (epoch + 1) % 20 == 0: 
                print(f"   Epoch {epoch+1}/{epochs_per_phase}: Loss = {loss.item():.6f}")

        model.eval()
        with torch.no_grad():
            logits_test_phase = model(X_test)
            probs_test_phase = torch.sigmoid(logits_test_phase)
            try:
                auc_test_phase = roc_auc_score(y_test.cpu().numpy(), probs_test_phase.cpu().numpy())
                print(f"   AUC en Test Set (Fase {phase+1}): {auc_test_phase:.4f}")
                if auc_test_phase > best_model_cell_auc:
                    best_model_cell_auc = auc_test_phase
                    torch.save(model.state_dict(), best_model_cell_path)
                    print(f"   🎉 Nuevo mejor modelo para {cell_name} guardado con AUC: {best_model_cell_auc:.4f}")
            except ValueError as e:
                 print(f"   ⚠️ No se pudo calcular AUC en Test Set (Fase {phase+1}) para {cell_name}: {e}")
        
        if hasattr(model, 'prune'):
            print(f"   Podando modelo para {cell_name}...")
            try:
                model.prune()
                print("   Modelo podado.")
            except Exception as e_prune:
                print(f"   ⚠️ Error durante la poda para {cell_name}: {e_prune}")
        else:
            print(f"   INFO: El modelo KAN actual no tiene un método 'prune'. Omitiendo poda.")


    # ✅ 4. Evaluación Final
    print(f"\n\n🔬 Iniciando Evaluación Final para {cell_name}")

    if not os.path.exists(best_model_cell_path):
        print(f"❌ No se encontró un modelo guardado para {cell_name} en {best_model_cell_path}. Omitiendo evaluación.")
        return

    final_model = KAN(width=current_kan_width, grid=kan_grid_size, k=kan_k, seed=kan_seed).to(device)
    try:
        final_model.load_state_dict(torch.load(best_model_cell_path, map_location=device))
    except Exception as e_load:
        print(f"❌ ERROR al cargar el modelo {best_model_cell_path} para {cell_name}: {e_load}. Omitiendo evaluación.")
        return
        
    final_model.eval()

    print(f"Cargando el mejor modelo para {cell_name} desde: {best_model_cell_path} (Mejor AUC durante entrenamiento: {best_model_cell_auc:.4f})")

    with torch.no_grad():
        logits_test = final_model(X_test)
        probs = torch.sigmoid(logits_test)
        preds = (probs > 0.5).float()

        y_test_cpu = y_test.cpu().numpy()
        probs_cpu = probs.cpu().numpy()
        preds_cpu = preds.cpu().numpy()

        if len(np.unique(y_test_cpu)) < 2:
            print(f"❌ ERROR ({cell_name}): El conjunto de prueba solo tiene una clase. No se puede calcular AUC ni el reporte completo.")
            acc = accuracy_score(y_test_cpu, preds_cpu)
            print(f"Accuracy Final para {cell_name}: {acc:.4f} (solo una clase en y_test)")
            return

        final_auc = roc_auc_score(y_test_cpu, probs_cpu)
        final_acc = accuracy_score(y_test_cpu, preds_cpu)
        
        print(f"\n🎉🎉🎉 RESULTADOS FINALES PARA {cell_name.upper()} 🎉🎉🎉")
        print(f"AUC Final: {final_auc:.4f}")
        print(f"Accuracy Final: {final_acc:.4f}")

        print("\n📊 Reporte de Clasificación Final:")
        print(classification_report(y_test_cpu, preds_cpu, digits=4, zero_division=0))
        
        fpr, tpr, _ = roc_curve(y_test_cpu, probs_cpu)
        plt.figure(figsize=(8, 6)) 
        plt.plot(fpr, tpr, label=f"KAN {cell_name.replace('_', ' ').title()} (AUC = {final_auc:.4f})", color='mediumblue', linewidth=2.5)
        plt.plot([0, 1], [0, 1], '--', color='dimgray', label='Aleatorio')
        plt.xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12)
        plt.ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12)
        plt.title(f"Curva ROC - KAN {cell_name.replace('_', ' ').title()}", fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        roc_save_path = os.path.join("models_cerebelo", f"roc_kan_{cell_name}.png")
        plt.savefig(roc_save_path)
        plt.close() # Cerrar la figura para liberar memoria
        print(f"\n📈 Curva ROC para {cell_name} guardada como {roc_save_path}")

    print(f"🏁 Proceso completado para {cell_name}.")


# --- Lista de Células a Entrenar y sus Archivos de Datos ---
# Asegúrate de que estas rutas y nombres de archivo coincidan con lo que generaste.
# Se usan los archivos '_light.csv' para un entrenamiento más rápido como ejemplo.
# Cambia a '_kan_ready.csv' para usar los datasets completos.
CELL_DATA_CONFIG = [
    {
        "name": "purkinje_hh_dinamico", 
        "file": "dataset_cerebelo/purkinje_hh_dinamico_light.csv",
        "features": ["time_ms", "voltage_mV", "input_current_uA_cm2"] # Purkinje usa uA/cm2
    },
    {"name": "granule_lif", "file": "dataset_cerebelo/granule_lif_light.csv", "features": FEATURE_COLUMNS_DEFAULT},
    {"name": "golgi_lif", "file": "dataset_cerebelo/golgi_lif_light.csv", "features": FEATURE_COLUMNS_DEFAULT},
    {"name": "basket_lif", "file": "dataset_cerebelo/basket_lif_light.csv", "features": FEATURE_COLUMNS_DEFAULT},
    {"name": "stellate_lif", "file": "dataset_cerebelo/stellate_lif_light.csv", "features": FEATURE_COLUMNS_DEFAULT},
    {"name": "deep_nuclei_lif", "file": "dataset_cerebelo/deep_nuclei_lif_light.csv", "features": FEATURE_COLUMNS_DEFAULT},
    {"name": "mossy_fiber", "file": "dataset_cerebelo/mossy_fiber_light.csv", "features": FEATURE_COLUMNS_DEFAULT},
    {"name": "climbing_fiber", "file": "dataset_cerebelo/climbing_fiber_light.csv", "features": FEATURE_COLUMNS_DEFAULT},
]

def main():
    for config in CELL_DATA_CONFIG:
        train_and_evaluate_kan_for_cell(
            cell_name=config["name"],
            data_file_path=config["file"],
            feature_columns=config["features"],
            target_column=TARGET_COLUMN,
            kan_width=KAN_WIDTH,
            kan_grid_size=KAN_GRID_SIZE,
            kan_k=KAN_K,
            kan_seed=KAN_SEED,
            num_phases=NUM_PHASES,
            epochs_per_phase=EPOCHS_PER_PHASE,
            learning_rate_schedule=LEARNING_RATE_SCHEDULE,
            max_grad_norm=MAX_GRAD_NORM,
            weight_decay=WEIGHT_DECAY
        )
    print("\n\n🎉🎉🎉 TODOS LOS ENTRENAMIENTOS COMPLETADOS 🎉🎉🎉")

if __name__ == "__main__":
    main()
