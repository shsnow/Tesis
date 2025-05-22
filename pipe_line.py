# kan_pipeline_master.py
# =======================================
# Script principal para generar datasets, entrenar modelos KAN por célula cerebelosa
# =======================================

import os
import subprocess
from pathlib import Path

# === Rutas ===
BASE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "model"

# === Paso 1: Generar CSVs de simulaciones ===
print("\n📦 Ejecutando simulador neuronal para generar CSVs...")
sim_script = BASE_DIR / "03_generate_data_allcells_1.py"
subprocess.run(["python", sim_script], check=True)

# === Paso 2: Convertir CSVs a .npz ===
print("\n📂 Convirtiendo CSVs a NPZ...")
convert_script = BASE_DIR / "procesador_kan_4.py"
subprocess.run(["python", convert_script, "--convert_only"], check=True)

# === Paso 3: Entrenar modelos KAN por tipo celular ===
print("\n🧠 Entrenando modelos KAN por célula...")
subprocess.run(["python", convert_script, "--train_only"], check=True)

# === Verificación final ===
print("\n✅ Pipeline completo ejecutado con éxito")
print(f"📁 Datasets en: {DATASET_DIR}")
print(f"📁 Modelos en: {MODEL_DIR}")
