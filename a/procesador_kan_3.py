import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kan import KAN

def convertir_csv_a_npz(csv_path, output_path, window_size=30):
    df = pd.read_csv(csv_path)
    nombre = os.path.basename(csv_path).replace("_kan_ready.csv", "")

    if 'input_current_nA' not in df.columns:
        df['input_current_nA'] = 0.0  # Para Poisson

    X = []
    y = []
    for i in range(len(df) - window_size):
        ventana = df.iloc[i:i+window_size]
        X.append(ventana[['voltage_mV', 'input_current_nA']].values.flatten())
        y.append(df['spike'].iloc[i + window_size])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    np.savez(output_path, X=X, y=y)
    print(f"✅ {nombre} convertido a .npz con {len(X)} muestras.")

def cargar_dataset(npz_path):
    data = np.load(npz_path)
    X = torch.tensor(data['X'], dtype=torch.float32)
    y = torch.tensor(data['y'], dtype=torch.float32).unsqueeze(1)
    return X, y

def entrenar_kan(X, y, epochs=200, lr=0.005):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = X.to(device), y.to(device)

    model = KAN(width=[X.shape[1], 16, 16, 1], grid=8, k=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    loss_hist = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    return model, loss_hist

def guardar_grafico(loss_hist, nombre):
    plt.figure()
    plt.plot(loss_hist)
    plt.title(f"KAN Loss - {nombre}")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.grid()
    plt.savefig(f"models/{nombre}_kan_loss.png")
    plt.close()

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("dataset_npz", exist_ok=True)

    # Convertir CSVs
    csv_dir = "dataset"
    for file in os.listdir(csv_dir):
        if file.endswith("_kan_ready.csv"):
            nombre = file.replace("_kan_ready.csv", "")
            csv_path = os.path.join(csv_dir, file)
            npz_path = os.path.join("dataset_npz", f"{nombre}.npz")
            convertir_csv_a_npz(csv_path, npz_path)

    # Entrenar modelos
    for file in os.listdir("dataset_npz"):
        if file.endswith(".npz"):
            nombre = file.replace(".npz", "")
            print(f"\n📦 Entrenando modelo KAN para: {nombre}")
            X, y = cargar_dataset(os.path.join("dataset_npz", file))
            model, loss_hist = entrenar_kan(X, y)
            torch.save(model.state_dict(), f"models/{nombre}_kan.pt")
            guardar_grafico(loss_hist, nombre)
            print(f"✅ Modelo guardado: models/{nombre}_kan.pt")

if __name__ == "__main__":
    main()
