import pandas as pd
df = pd.read_csv('dataset/purkinje_kan_ready.csv')
print(df['spike'].value_counts(normalize=True))

import matplotlib.pyplot as plt
plt.plot(df['time_ms'], df['voltage_mV'], label='Voltaje')
plt.plot(df['time_ms'], df['spike'] * 50, label='Spikes (amplificado)')
plt.legend(); plt.show()