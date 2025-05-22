import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from kan import KAN
import pandas as pd

# Directorios
DATA_DIR = "dataset"
MODEL_DIR = "model"
DATASET_NPZ_DIR = "dataset_npz"
os.makedirs(MODEL_DIR, exist_ok=True)

# Balanceador simple
def balance_dataset(X, y):
    y = y.reshape(-1)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0:
        print("❌ No hay clase positiva para balancear.")
        return X, y.reshape(-1, 1)
    neg_idx = np.random.choice(neg_idx, size=len(pos_idx), replace=False)
    idx_total = np.concatenate([pos_idx, neg_idx])
    np.random.shuffle(idx_total)
    return X[idx_total], y[idx_total].reshape(-1, 1)

# Entrenamiento
def train_model(X, y, name, epochs=200, lr=0.005):
    X, y = balance_dataset(X, y)

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

    model = KAN(width=[X.shape[1], 8, 8, 1], grid=8, k=3)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_tensor)
        loss = loss_fn(pred, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"{name} | Epoch {epoch} Loss: {loss.item():.4f}")

    # Validación
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
        logits = model(X_val_tensor).cpu().numpy()
        prob = 1 / (1 + np.exp(-logits))
        pred = (prob > 0.5).astype(int)

    # Métricas
    acc = accuracy_score(y_val, pred)
    auc = roc_auc_score(y_val, prob)
    f1 = f1_score(y_val, pred)
    print(f"\U0001F4CA {name}: Accuracy = {acc:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_val, prob)
    plt.plot(fpr, tpr)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve - {name}")
    plt.grid(True)
    plt.savefig(f"{MODEL_DIR}/{name}_roc.png")
    plt.close()

    # Guardar modelo
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"{name}.pt"))
    print(f"📅 Modelo guardado como {name}.pt\n")



def preprocess_csv_to_npz():
    print("\n=== PREPROCESANDO .CSV a .NPZ ===")
    for file in os.listdir(DATA_DIR):
        if file.endswith(".csv"):
            name = file.replace(".csv", "")
            path = os.path.join(DATA_DIR, file)
            df = pd.read_csv(path)

            if not {'voltage_mV', 'input_current_nA', 'spike'}.issubset(df.columns):
                print(f"⚠️ {file} no tiene el formato correcto. Skipping.")
                continue

            X = df[['voltage_mV', 'input_current_nA']].values.astype(np.float32)
            y = df['spike'].values.astype(np.float32)
            np.savez_compressed(os.path.join(DATASET_NPZ_DIR, name + ".npz"), X=X, y=y)
            print(f"✅ Convertido {file} → {name}.npz")



# Procesamiento global
for file in os.listdir(DATASET_NPZ_DIR):
    if file.endswith(".npz"):
        path = os.path.join(DATASET_NPZ_DIR, file)
        data = np.load(path)
        X = data['X']
        y = data['y']
        name = file.replace(".npz", "")
        train_model(X, y, name)

print("\u2705 Todos los modelos entrenados y evaluados.")
