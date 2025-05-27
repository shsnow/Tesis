import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("dataset/purkinje_kan_ready.csv")

plt.figure(figsize=(12, 4))
plt.plot(df['time_ms'], df['voltage_mV'], label="Voltaje")
plt.plot(df['time_ms'], df['spike']*50, label="Spike", color="red", alpha=0.6)
plt.legend(); plt.title("Actividad de Purkinje"); plt.grid(); plt.show()

print(df['spike'].value_counts(normalize=True))