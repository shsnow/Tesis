import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from sklearn.utils import resample
from kan import KAN
from brian2 import *
import pandas as pd
# === CONFIGURACIÓN ===
EPOCHS = 200
LR = 0.005
OUTPUT_DIR = "models_grid"
ERROR_LOG = os.path.join(OUTPUT_DIR, "error_log.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === HIPERPARÁMETROS A PROBAR ===
I_EXT_VALUES = [20, 30, 40, 50]  # uA/cm^2
DURATIONS = [1000, 2000, 3000]  # ms
WIDTHS = [[4, 8, 8, 1], [4, 12, 12, 1], [4, 16, 16, 1]]
GRIDS = [8, 16]
K_VALUES = [2, 3]

# === SIMULACIÓN PURKINJE ===
def simular_purkinje(i_ext_uA_cm2=20, duracion_ms=1000):


    start_scope()
    defaultclock.dt = 0.01*ms
    duracion = duracion_ms * ms

    eqs = '''
    dv/dt = (I_ext - gNa*(m**3)*h*(v - ENa) - gK*(n**4)*(v - EK) - gL*(v - EL)) / Cm : volt
    dm/dt = alpha_m*(1 - m) - beta_m*m : 1
    dh/dt = alpha_h*(1 - h) - beta_h*h : 1
    dn/dt = alpha_n*(1 - n) - beta_n*n : 1
    I_ext : amp/meter**2
    alpha_m = 0.1 * (25 - v/mV) / (exp((25 - v/mV) / 10) - 1) / ms : Hz
    beta_m = 4 * exp(-v / (18*mV)) / ms : Hz
    alpha_h = 0.07 * exp(-v / (20*mV)) / ms : Hz
    beta_h = 1 / (exp((30 - v/mV) / 10) + 1) / ms : Hz
    alpha_n = 0.01 * (10 - v/mV) / (exp((10 - v/mV) / 10) - 1) / ms : Hz
    beta_n = 0.125 * exp(-v / (80*mV)) / ms : Hz
    '''

    G = NeuronGroup(100, eqs, threshold='v > -40*mV', reset='v = -65*mV', method='exponential_euler',
                    namespace={'Cm': 1*uF/cm**2, 'gNa': 120*msiemens/cm**2, 'gK': 36*msiemens/cm**2, 'gL': 0.3*msiemens/cm**2,
                              'ENa': 50*mV, 'EK': -77*mV, 'EL': -54.4*mV})
    G.v = -65*mV; G.m = 0.05; G.h = 0.6; G.n = 0.32
    G.I_ext = i_ext_uA_cm2 * uA / cm**2

    mon = StateMonitor(G, ['v', 'I_ext'], record=True)
    spikes = SpikeMonitor(G)
    run(duracion)

    t_array = mon.t / ms
    if len(spikes.i) == 0:
        print("❌ Ninguna neurona disparó. CSV no generado.")
        return False
    idx = int(spikes.i[0])
    v_array = mon.v[idx] / mV
    I_array = np.full_like(v_array, i_ext_uA_cm2, dtype=np.float32)
    spike_array = np.zeros_like(t_array)
    for st in spikes.spike_trains()[idx]/ms:
        spike_array[np.isclose(t_array, st, atol=0.05)] = 1

    df = pd.DataFrame({
        'time_ms': t_array,
        'voltage_mV': v_array,
        'input_current_nA': I_array,
        'spike': spike_array
    })
    df.to_csv("purkinje_kan_ready.csv", index=False)
    return True

# === PREPROCESAMIENTO ===
def preprocess():
    df = pd.read_csv("purkinje_kan_ready.csv")
    if df.empty or df['spike'].sum() == 0:
        print("❌ CSV vacío o sin spikes. Skipping...")
        return np.array([]), np.array([]), df

    dt = df['time_ms'].diff().fillna(1)
    df['dvdt'] = df['voltage_mV'].diff().fillna(0) / dt.replace(0, 1)
    df['dvdt2'] = df['dvdt'].diff().fillna(0) / dt.replace(0, 1)
    for col in ['voltage_mV', 'input_current_nA', 'dvdt', 'dvdt2']:
        std = df[col].std()
        df[col] = (df[col] - df[col].mean()) / std if std > 0 else df[col] - df[col].mean()

    X = df[['voltage_mV', 'input_current_nA', 'dvdt', 'dvdt2']].values.astype(np.float32)
    y = df['spike'].values.astype(np.float32)
    return balance_dataset(X, y) + (df,)

# (el resto del código permanece igual...)


def balance_dataset(X, y):
    y = y.flatten()
    X_pos = X[y == 1]
    y_pos = y[y == 1]
    X_neg = X[y == 0]
    y_neg = y[y == 0]
    if len(X_pos) < 5 or len(X_neg) < 5:
        print("⚠️ Dataset insuficiente: menos de 5 muestras por clase.")
        return np.array([]), np.array([])
    n_samples = int(len(X_neg) * 0.5)
    X_pos_upsampled, y_pos_upsampled = resample(X_pos, y_pos, replace=True, n_samples=n_samples, random_state=42)
    X_bal = np.vstack((X_neg, X_pos_upsampled))
    y_bal = np.hstack((y_neg, y_pos_upsampled))
    return X_bal, y_bal.reshape(-1, 1)

def log_error(tag, df, reason):
    with open(ERROR_LOG, 'a') as f:
        f.write(f"\n=== Error en {tag} ===\n")
        f.write(f"Razón: {reason}\n")
        f.write(df.head(10).to_string())
        f.write("\n--------------------------\n")

def evaluate(y_true, y_pred):
    y_pred_bin = (y_pred >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred_bin),
        "auc": roc_auc_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred_bin),
        "recall": recall_score(y_true, y_pred_bin),
        "f1": f1_score(y_true, y_pred_bin)
    }

def train_model(X, y, width, grid, k, tag):
    if len(X) == 0 or len(y) == 0 or np.isnan(X).any() or np.isnan(y).any():
        print(f"❌ Skipping {tag}: datos vacíos o con NaNs.")
        return {"tag": tag, "loss": np.nan, "accuracy": 0.0, "auc": 0.0, "f1": 0.0}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    model = KAN(width=width, grid=grid, k=k).to(device)
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

    model.eval()
    y_pred = torch.sigmoid(model(X_test_tensor)).detach().cpu().numpy()
    metrics = evaluate(y_test, y_pred)
    result = {"tag": tag, "loss": loss.item(), **metrics}
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"{tag}.pt"))
    return result

def main():
    results = []
    for i_ext, dur, width, grid, k in product(I_EXT_VALUES, DURATIONS, WIDTHS, GRIDS, K_VALUES):
        tag = f"purkinje_I{i_ext}_T{dur}_W{width[1]}_G{grid}_K{k}"
        print(f"\n🚀 Ejecutando experimento: {tag}")
        try:
            valid = simular_purkinje(i_ext, dur)
            if not valid:
                print(f"⏭️ Skipping {tag}: sin actividad detectable.")
                continue
            X, y, df = preprocess()
            if len(X) == 0 or len(y) == 0 or np.isnan(X).any() or np.isnan(y).any():
                print(f"❌ Error en {tag}: NaNs o datos vacíos detectados")
                log_error(tag, df, "NaNs o datos insuficientes")
                result = {"tag": tag, "loss": np.nan, "accuracy": 0.0, "auc": 0.0, "f1": 0.0}
            else:
                result = train_model(X, y, width, grid, k, tag)
        except Exception as e:
            print(f"❌ Error crítico en {tag}: {e}")
            df = pd.read_csv("purkinje_kan_ready.csv") if os.path.exists("purkinje_kan_ready.csv") else pd.DataFrame()
            log_error(tag, df, str(e))
            result = {"tag": tag, "loss": np.nan, "accuracy": 0.0, "auc": 0.0, "f1": 0.0}
        results.append(result)

    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "purkinje_grid_results.csv"), index=False)
    print("\n📊 Resultados guardados en purkinje_grid_results.csv")
    print("📝 Log de errores guardado en error_log.txt")

if __name__ == "__main__":
    main()
