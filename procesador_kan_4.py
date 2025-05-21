import os
import glob
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from kan import KAN

# === CONFIGURACION ===
DATASET_DIR = "dataset"
NPZ_DIR = "dataset_npz"
MODELS_DIR = "models"
GRAPHS_DIR = "plots"
os.makedirs(NPZ_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(GRAPHS_DIR, exist_ok=True)

# === FUNCION: Preprocesamiento ===
def convertir_csv_a_npz(nombre_archivo_csv):
    df = pd.read_csv(nombre_archivo_csv)

    if 'input_current_nA' not in df.columns:
        print(f"⚠️ CSV sin corriente: {nombre_archivo_csv}")
        df['input_current_nA'] = 0.0  # dummy input para Poisson

    X = df[['voltage_mV', 'input_current_nA']].values.astype(np.float32)
    y = df['spike'].values.astype(np.float32).reshape(-1, 1)

    base = os.path.basename(nombre_archivo_csv).replace(".csv", ".npz")
    np.savez_compressed(os.path.join(NPZ_DIR, base), X=X, y=y)
    print(f"✅ Convertido {nombre_archivo_csv} → {base}")

# === FUNCION: Entrenamiento ===
def entrenar_kan(npz_file):
    data = np.load(npz_file)
    X, y = data['X'], data['y']
    y = y.reshape(-1, 1)

    if np.sum(y) == 0:
        print(f"⛔ Dataset sin spikes: {npz_file}, omitiendo...")
        return

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train).to(device)
    y_train_tensor = torch.tensor(y_train).to(device)
    X_val_tensor = torch.tensor(X_val).to(device)
    y_val_tensor = torch.tensor(y_val).to(device)

    model = KAN(width=[2, 8, 8, 1], grid=8, k=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    losses = []

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = loss_fn(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % 20 == 0:
            print(f"{os.path.basename(npz_file)} | Epoch {epoch} Loss: {loss.item():.4f}")

    # === EVALUACION ===
    model.eval()
    with torch.no_grad():
        y_pred_val = torch.sigmoid(model(X_val_tensor)).cpu().numpy()
        y_pred_class = (y_pred_val > 0.5).astype(int)
        y_true = y_val_tensor.cpu().numpy()
        acc = accuracy_score(y_true, y_pred_class)
        auc = roc_auc_score(y_true, y_pred_val)

    print(f"📊 Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    # === GRAFICO ===
    plt.plot(losses)
    plt.title(f"Loss - {os.path.basename(npz_file)}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(os.path.join(GRAPHS_DIR, os.path.basename(npz_file).replace(".npz", "_loss.png")))
    plt.close()

    torch.save(model.state_dict(), os.path.join(MODELS_DIR, os.path.basename(npz_file).replace(".npz", ".pt")))
    print(f"💾 Modelo guardado como {os.path.basename(npz_file).replace('.npz', '.pt')}\n")

# === MAIN ===
if __name__ == "__main__":
    csv_files = glob.glob(os.path.join(DATASET_DIR, "*_kan_ready.csv"))
    for csv in csv_files:
        convertir_csv_a_npz(csv)

    print("\n=== ENTRENANDO MODELOS KAN ===\n")
    npz_files = glob.glob(os.path.join(NPZ_DIR, "*.npz"))
    for npz in npz_files:
        entrenar_kan(npz)

    print("\n✅ Todos los modelos entrenados y guardados correctamente.")
