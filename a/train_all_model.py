import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from kan import KAN

def cargar_dataset(path):
    df = pd.read_csv(path)
    X = df[['voltage_mV', 'input_current_nA']].values.astype(np.float32)
    y = df['spike'].values.astype(np.float32).reshape(-1, 1)
    return X, y

def entrenar_modelo_kan(X, y, epochs=200, lr=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KAN(width=[2, 8, 8, 1], grid=8, k=3).to(device)
    
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_hist = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = loss_fn(pred, y_tensor)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())

    return model, loss_hist

def evaluar_modelo(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        pred_logits = model(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy()
        pred_probs = 1 / (1 + np.exp(-pred_logits))
        pred_bin = (pred_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, pred_bin)
    auc = roc_auc_score(y_test, pred_probs)
    return acc, auc, pred_probs

def graficar_y_guardar(loss_hist, y_test, pred_probs, nombre, output_dir):
    # Pérdida
    plt.figure()
    plt.plot(loss_hist)
    plt.title(f"{nombre} - Pérdida")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(f"{output_dir}/{nombre}_loss.png")
    plt.close()

    # ROC
    fpr, tpr, _ = roc_curve(y_test, pred_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, pred_probs):.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title(f"{nombre} - Curva ROC")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/{nombre}_roc.png")
    plt.close()

def procesar_dataset(path, output_dir):
    nombre = os.path.basename(path).replace("_kan_ready.csv", "")
    print(f"📦 Procesando: {nombre}")
    
    X, y = cargar_dataset(path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, loss_hist = entrenar_modelo_kan(X_train, y_train)
    acc, auc, pred_probs = evaluar_modelo(model, X_test, y_test)

    torch.save(model.state_dict(), f"{output_dir}/{nombre}_kan.pt")
    graficar_y_guardar(loss_hist, y_test, pred_probs, nombre, output_dir)

    print(f"✅ Modelo {nombre} | Acc: {acc:.2f} | AUC: {auc:.2f}\n")

def main():
    input_dir = "dataset"  # Directorio donde están los CSV
    output_dir = "modelos_kan"
    os.makedirs(output_dir, exist_ok=True)

    archivos = [f for f in os.listdir(input_dir) if f.endswith("_kan_ready.csv")]
    for archivo in archivos:
        procesar_dataset(os.path.join(input_dir, archivo), output_dir)

if __name__ == "__main__":
    main()
