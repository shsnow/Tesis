import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from kan import KAN
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    X = df[['voltage_mV', 'input_current_nA']].values.astype(np.float32)
    y = df['spike'].values.astype(np.float32).reshape(-1, 1)
    return X, y

def balance_data(X, y):
    y = y.reshape(-1)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        print("⚠️ Datos desbalanceados sin ejemplos positivos o negativos")
        return X, y.reshape(-1, 1)

    n_samples = min(len(pos_idx), len(neg_idx))
    indices = np.concatenate([np.random.choice(pos_idx, n_samples), np.random.choice(neg_idx, n_samples)])
    np.random.shuffle(indices)

    return X[indices], y[indices].reshape(-1, 1)

def train_model(X, y, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    model = KAN(width=config['width'], grid=config['grid'], k=config['k']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    loss_history = []
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = loss_fn(output, y_tensor)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

    return model, loss_history

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    f1 = f1_score(y_test, preds)
    return acc, auc, f1

def plot_loss(losses, labels):
    plt.figure(figsize=(10, 6))
    for loss, label in zip(losses, labels):
        plt.plot(loss, label=label)
    plt.title("KAN Training Loss for Purkinje Experiments")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def run_experiments():
    csv_file = "dataset/purkinje_kan_ready.csv"
    X, y = load_dataset(csv_file)

    configs = [
        {"width": [2, 8, 1], "grid": 8, "k": 3, "lr": 0.01, "epochs": 200},
        {"width": [2, 16, 1], "grid": 16, "k": 3, "lr": 0.005, "epochs": 200},
        {"width": [2, 32, 1], "grid": 32, "k": 3, "lr": 0.003, "epochs": 200},
        {"width": [2, 16, 8, 1], "grid": 16, "k": 3, "lr": 0.001, "epochs": 200}
    ]

    os.makedirs("model", exist_ok=True)

    all_losses = []
    labels = []

    for i, config in enumerate(configs):
        print(f"\n🔬 Experiment {i+1}: {config}")

        X_bal, y_bal = balance_data(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_bal, y_bal, test_size=0.2, random_state=42)

        model, losses = train_model(X_train, y_train, config)
        acc, auc, f1 = evaluate_model(model, X_test, y_test)

        print(f"📊 Accuracy = {acc:.4f}, AUC = {auc:.4f}, F1 = {f1:.4f}")
        torch.save(model.state_dict(), f"model/purkinje_exp{i+1}.pt")
        print(f"💾 Modelo guardado como purkinje_exp{i+1}.pt")

        all_losses.append(losses)
        labels.append(f"exp{i+1} (AUC={auc:.3f})")

    plot_loss(all_losses, labels)

if __name__ == "__main__":
    run_experiments()
