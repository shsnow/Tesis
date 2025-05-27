import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from sklearn.utils import resample
from kan import KAN

# === CONFIGURACIÓN ===
EPOCHS = 200
LR = 0.005
DATASET_PATH = "dataset/purkinje_kan_ready.csv"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_purkinje(path):
    df = pd.read_csv(path)
    
    # Features dinámicas
    df['dvdt'] = df['voltage_mV'].diff().fillna(0) / df['time_ms'].diff().fillna(1)
    df['dvdt2'] = df['dvdt'].diff().fillna(0) / df['time_ms'].diff().fillna(1)

    # Normalización
    for col in ['voltage_mV', 'input_current_nA', 'dvdt', 'dvdt2']:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    X = df[['voltage_mV', 'input_current_nA', 'dvdt', 'dvdt2']].values.astype(np.float32)
    y = df['spike'].values.astype(np.float32)
    return X, y

def balance_dataset(X, y):
    y = y.flatten()
    X_pos = X[y == 1]
    y_pos = y[y == 1]
    X_neg = X[y == 0]
    y_neg = y[y == 0]

    if len(X_pos) == 0 or len(X_neg) == 0:
        print("❌ Error: No hay suficientes muestras de ambas clases para balancear.")
        return X, y.reshape(-1, 1)

    # Oversample spikes hasta que tengamos 1:2
    n_samples = int(len(X_neg) * 0.5)
    X_pos_upsampled, y_pos_upsampled = resample(X_pos, y_pos, replace=True, n_samples=n_samples, random_state=42)

    X_bal = np.vstack((X_neg, X_pos_upsampled))
    y_bal = np.hstack((y_neg, y_pos_upsampled))
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

def train_purkinje():
    print("\n=== ENTRENANDO MODELO KAN PARA PURKINJE ===")
    X, y = preprocess_purkinje(DATASET_PATH)
    X, y = balance_dataset(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    model = KAN(width=[4, 12, 12, 1], grid=8, k=3)
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
            print(f"purkinje_kan_experiment | Epoch {epoch} Loss: {loss.item():.4f}")

    model.eval()
    y_pred = torch.sigmoid(model(X_test_tensor)).detach().cpu().numpy()
    metrics = evaluate(y_test, y_pred)
    print(f"\U0001F4CA purkinje_kan_experiment: Accuracy = {metrics['accuracy']:.4f}, AUC = {metrics['auc']:.4f}, F1 = {metrics['f1']:.4f}")
    plot_roc(y_test, y_pred, "purkinje_kan_experiment")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "purkinje_kan_experiment.pt"))
    print("\U0001F4BE Modelo guardado como purkinje_kan_experiment.pt\n")

if __name__ == "__main__":
    train_purkinje()
