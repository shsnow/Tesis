#!/usr/bin/env python3
# train_all_kan_models.py

import os
import glob
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve,classification_report
import matplotlib.pyplot as plt
import json
# Asegúrate de tener instalada la implementación de KAN (por ejemplo: pip install pykan)
from kan import KAN

# ──────────────────────────────────────────────────────────
# Logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
all_metrics = []
# Detectar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Usando dispositivo: {device}")

# Parámetros globales de entrenamiento
KAN_WIDTH = [3, 32, 16, 1]
KAN_GRID  = 5
KAN_K     = 3
KAN_SEED  = 42

NUM_PHASES        = 3
EPOCHS_PER_PHASE  = 70
LR_SCHEDULE       = {0: 1e-3, 1: 5e-4, 2: 1e-4}
WEIGHT_DECAY      = 1e-5
MAX_GRAD_NORM     = 1.0
TEST_SIZE         = 0.2
RANDOM_STATE      = 42

DATA_DIR  = "dataset_cerebelo"
OUT_DIR   = "models_cerebelo"
os.makedirs(OUT_DIR, exist_ok=True)

def train_one_cell(csv_path):
    cell_name = os.path.basename(csv_path).replace("_kan_ready.csv", "")
    logging.info(f"\n===== Procesando célula: {cell_name} =====")
    
    # 1) Carga y preprocesamiento
    df = pd.read_csv(csv_path)
    X = df[["time_ms", "voltage_mV", "input_current_nA"]].values.astype(np.float32)
    y = df["spike"].values.astype(np.float32)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    # Guardar scaler
    scaler_path = os.path.join(OUT_DIR, f"scaler_{cell_name}.joblib")
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler guardado en: {scaler_path}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    # Tensores
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=device)
    y_test_t  = torch.tensor(y_test,  dtype=torch.float32, device=device).unsqueeze(1)
    
    # 2) Construir modelo
    width = [X_train_t.shape[1]] + KAN_WIDTH[1:]
    model = KAN(width=width, grid=KAN_GRID, k=KAN_K, seed=KAN_SEED).to(device)
    logging.info(f"{cell_name}: modelo KAN con {sum(p.numel() for p in model.parameters() if p.requires_grad)} parámetros")
    
    criterion = nn.BCEWithLogitsLoss()
    best_auc = 0.0
    best_model_path = os.path.join(OUT_DIR, f"kan_{cell_name}.pt")
    
    # 3) Entrenamiento por fases
    for phase in range(NUM_PHASES):
        lr = LR_SCHEDULE.get(phase, LR_SCHEDULE[NUM_PHASES-1])
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        logging.info(f"Fase {phase+1}/{NUM_PHASES} - LR = {lr:.1e}")
        
        for epoch in range(1, EPOCHS_PER_PHASE+1):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train_t)
            loss = criterion(logits, y_train_t)
            if torch.isnan(loss):
                logging.error("Loss es NaN, deteniendo entrenamiento")
                return
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            
            if epoch % 20 == 0 or epoch==EPOCHS_PER_PHASE:
                logging.info(f"  Epoch {epoch}/{EPOCHS_PER_PHASE} - Loss = {loss.item():.6f}")
        
        # Evaluar fase
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X_test_t)).cpu().numpy()
            auc = roc_auc_score(y_test, probs)
            logging.info(f"  AUC en test (fase {phase+1}): {auc:.4f}")
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"  🎉 Nuevo mejor modelo guardado: AUC = {best_auc:.4f}")
        
        # Poda si existe
        if hasattr(model, "prune"):
            try:
                model.prune()
                logging.info("  Modelo podado exitosamente")
            except Exception as e:
                logging.warning(f"  Poda fallida: {e}")
    
    # 4) Evaluación final y curva ROC
    logging.info(f"\n--- Evaluación final para {cell_name} ---")
    final_model = KAN(width=width, grid=KAN_GRID, k=KAN_K, seed=KAN_SEED).to(device)
    final_model.load_state_dict(torch.load(best_model_path, map_location=device))
    final_model.eval()
    
    with torch.no_grad():
        logits = final_model(X_test_t)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs > 0.5).astype(int)
        logging.info(f"Predicciones para {cell_name} generadas")
        auc   = roc_auc_score(y_test, probs)
        acc   = accuracy_score(y_test, preds)
        prec  = precision_score(y_test, preds, zero_division=0)
        rec   = recall_score(y_test, preds, zero_division=0)
        f1    = f1_score(y_test, preds, zero_division=0)
        logging.info(f"▶️ AUC final: {auc:.4f}, Accuracy: {acc:.4f}")
        print(classification_report(y_test, preds, digits=4, zero_division=0))
        
        # ROC plot
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0,1],[0,1],"--", color="gray")
        plt.title(f"ROC KAN - {cell_name}")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        roc_path = os.path.join(OUT_DIR, f"roc_{cell_name}.png")
        plt.savefig(roc_path)
        all_metrics.append({
            "cell": cell_name,
            "auc": float(auc),
            "accuracy": float(acc),
            "roc_curve": roc_path,
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        })
        plt.close()
        logging.info(f"Curva ROC guardada en: {roc_path}")

if __name__ == "__main__":
    # Buscar todos los archivos *_kan_ready.csv
    csv_files = glob.glob(os.path.join(DATA_DIR, "*_light.csv"))
    if not csv_files:
        logging.error(f"No se encontraron datasets en {DATA_DIR}")
        exit(1)
    
    for csv in sorted(csv_files):
        train_one_cell(csv)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(OUT_DIR, "summary_metrics.csv"), index=False)
    print(f"▶️ Métricas resumen guardadas en {OUT_DIR}/summary_metrics.csv")
    logging.info("🎉🎉 Todos los entrenamientos completados 🎉🎉")
