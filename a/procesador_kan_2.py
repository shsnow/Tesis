import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from kan import KAN

# Carpetas
os.makedirs("models", exist_ok=True)
os.makedirs("metrics", exist_ok=True)
os.makedirs("plots", exist_ok=True)

CELLS = [
    "granule", "golgi", "basket", "stellate", "nuclei", "purkinje", "mossy", "climbing"
]


def load_data(cell_name):
    data = np.load(f"dataset/{cell_name}_kan_ready.npz")
    return data['X'], data['y']


def train_and_save_model(cell_name, epochs=200, lr=0.005, test_size=0.2):
    print(f"\n=== Entrenando modelo para {cell_name} ===")
    X, y = load_data(cell_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KAN(width=[X.shape[1], 16, 16, 1], grid=8, k=3).to(device)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

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

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    # Guardar modelo
    model_path = f"models/{cell_name}_kan.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Modelo guardado en {model_path}")

    # Evaluación
    model.eval()
    with torch.no_grad():
        y_pred = torch.sigmoid(model(X_test_tensor)).cpu().numpy()
        y_pred_binary = (y_pred > 0.5).astype(np.float32)

    acc = accuracy_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    print(f"🎯 Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

    # Guardar métricas
    with open(f"metrics/{cell_name}_metrics.txt", "w") as f:
        f.write(f"Accuracy: {acc}\nF1 Score: {f1}\n")

    # Guardar gráfica de pérdida
    plt.figure()
    plt.plot(loss_hist)
    plt.title(f"Loss - {cell_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(f"plots/{cell_name}_loss.png")
    plt.close()

    print(f"📈 Gráfica de pérdida guardada en plots/{cell_name}_loss.png")
    print(f"📊 Métricas guardadas en metrics/{cell_name}_metrics.txt")


if __name__ == "__main__":
    for cell in CELLS:
        try:
            
            train_and_save_model(cell)
        except Exception as e:
            print(f"❌ Error al entrenar {cell}: {e}")
