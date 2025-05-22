# procesador_kan_4.py
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from kan import KAN
import matplotlib.pyplot as plt

DATASET_DIR = "dataset"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

def preprocess_csv_to_npz():
    print("\n=== PREPROCESANDO .CSV a .NPZ ===")
    for file in os.listdir(DATASET_DIR):
        if file.endswith(".csv"):
            name = file.replace(".csv", "")
            path = os.path.join(DATASET_DIR, file)
            df = pd.read_csv(path)

            if not {'voltage_mV', 'input_current_nA', 'spike'}.issubset(df.columns):
                print(f"⚠️ {file} no tiene el formato correcto. Skipping.")
                continue

            X = df[['voltage_mV', 'input_current_nA']].values.astype(np.float32)
            y = df['spike'].values.astype(np.float32)
            np.savez_compressed(os.path.join(DATASET_DIR, name + ".npz"), X=X, y=y)
            print(f"✅ Convertido {file} → {name}.npz")

def entrenar_kan(X, y, nombre, epochs=200, lr=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

    model = KAN(width=[2, 8, 8, 1], grid=8, k=3)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_hist = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_tensor)
        loss = loss_fn(pred, y_train_tensor)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"{nombre} | Epoch {epoch} Loss: {loss.item():.4f}")

    # Evaluar
    model.eval()
    with torch.no_grad():
        logits = model(X_test_tensor).cpu().numpy()
        probs = 1 / (1 + np.exp(-logits))
        y_pred = (probs > 0.5).astype(np.float32)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, probs)

    print(f"📊 Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, nombre + ".pt"))
    print(f"💾 Modelo guardado como {nombre}.pt")

    # Graficar
    plt.figure()
    plt.plot(loss_hist)
    plt.title(f"KAN Loss - {nombre}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(os.path.join(MODEL_DIR, nombre + "_loss.png"))
    plt.close()

def entrenar_todos():
    print("\n=== ENTRENANDO MODELOS KAN ===")
    for file in os.listdir(DATASET_DIR):
        if file.endswith(".npz"):
            path = os.path.join(DATASET_DIR, file)
            nombre = file.replace(".npz", "")
            data = np.load(path)
            X, y = data['X'], data['y']
            if len(np.unique(y)) < 2:
                print(f"⚠️ {file} tiene clases insuficientes. Skipping.")
                continue
            entrenar_kan(X, y, nombre)

def pipeline_kan():
    preprocess_csv_to_npz()
    entrenar_todos()

if __name__ == "__main__":
    pipeline_kan()
