import pandas as pd
import numpy as np
import torch
from kan import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_and_prepare(csv_file):
    df = pd.read_csv(csv_file)

    # Inputs: voltaje e intensidad → [V, I]
    X = df[['voltage_mV', 'input_current_nA']].values.astype(np.float32)
    # Output: spike (0 o 1)
    y = df['spike'].values.astype(np.float32).reshape(-1, 1)

    return X, y

def train_kan(X, y, epochs=100, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    model = KAN(width=[2, 8, 8, 1], grid=8, k=3)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()  # clasificación binaria

    loss_hist = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_tensor)
        loss = loss_fn(pred, y_tensor)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    return model, loss_hist

def plot_loss(loss_hist):
    plt.plot(loss_hist)
    plt.title("KAN training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()

# -------------------------- USO --------------------------

# Cambia este nombre por otras células
csv_file = "dataset/granule_kan_ready.csv"

X, y = load_and_prepare(csv_file)
model, loss_hist = train_kan(X, y, epochs=200, lr=0.005)
plot_loss(loss_hist)

# Guardar modelo
torch.save(model.state_dict(), "granule_kan_model.pt")
print("✅ Modelo guardado en granule_kan_model.pt")
