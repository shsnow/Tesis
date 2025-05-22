import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from kan import KAN

# === CONFIGURACIÓN ===
EPOCHS = 200
LR = 0.005
DATA_DIR = "dataset"
DATASET_NPZ_DIR = "dataset_npz"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_npz(path):
    data = np.load(path)
    X, y = data['X'], data['y']
    return X, y

def balance_dataset(X, y):
    y = y.flatten()  # Asegura que y sea 1D
    X_pos = X[y == 1]
    X_neg = X[y == 0]

    n_samples = min(len(X_pos), len(X_neg))
    if n_samples == 0:
        print("❌ Error: No hay suficientes muestras de ambas clases para balancear.")
        return X, y.reshape(-1, 1)

    # Undersample
    X_bal = np.vstack([X_pos[:n_samples], X_neg[:n_samples]])
    y_bal = np.hstack([np.ones(n_samples), np.zeros(n_samples)])

    return X_bal, y_bal.reshape(-1, 1)

def evaluate(y_true, y_pred):
    y_pred_bin = (y_pred >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred_bin),
        "auc": roc_auc_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred_bin),
        "recall": recall_score(y_true, y_pred_bin),
        "f1": f1_score(y_true, y_pred_bin)
    }

def plot_roc(y_true, y_pred, name):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_true, y_pred):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {name}')
    plt.legend()
    plt.grid()
    plt.savefig(f"{OUTPUT_DIR}/{name}_roc_curve.png")
    plt.close()

def train_model(X, y, name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = balance_dataset(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    model = KAN(width=[2, 8, 8, 1], grid=8, k=3)
    model.to(device)

    pos_weight = torch.tensor([len(y_train) / y_train.sum()]).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_tensor)
        loss = loss_fn(pred, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            print(f"{name} | Epoch {epoch} Loss: {loss.item():.4f}")

    model.eval()
    y_pred = torch.sigmoid(model(X_test_tensor)).detach().cpu().numpy()
    metrics = evaluate(y_test, y_pred)
    print(f"\U0001F4CA {name}: Accuracy = {metrics['accuracy']:.4f}, AUC = {metrics['auc']:.4f}, F1 = {metrics['f1']:.4f}")
    plot_roc(y_test, y_pred, name)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{name}.pt"))
    print(f"\U0001F4BE Modelo guardado como {name}.pt\n")

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


def main():
    print("=== ENTRENANDO MODELOS KAN CON PÉRDIDA PONDERADA Y AUMENTO DE DATOS ===")
    preprocess_csv_to_npz()
    for file in os.listdir(DATASET_NPZ_DIR):
        if file.endswith(".npz"):
            name = file.replace(".npz", "")
            path = os.path.join(DATASET_NPZ_DIR, file)
            X, y = load_npz(path)
            train_model(X, y, name)

if __name__ == "__main__":
    main()
