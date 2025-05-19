import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from kan import KAN
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import pickle


## ----------------------------
## 1. Carga y Preparación de Datos
## ----------------------------

def load_and_preprocess_data():
    """Carga todos los datasets y prepara los datos para KAN"""
    # Cargar todos los datasets
    mossy_df = pd.read_csv('cerebellar_datasets/kan_mossy_dataset.csv')
    granule_df = pd.read_csv('cerebellar_datasets/kan_granule_dataset.csv')
    golgi_df = pd.read_csv('cerebellar_datasets/kan_golgi_dataset.csv')
    basket_df = pd.read_csv('cerebellar_datasets/kan_basket_dataset.csv')
    stellate_df = pd.read_csv('cerebellar_datasets/kan_stellate_dataset.csv')
    climbing_df = pd.read_csv('cerebellar_datasets/kan_climbing_dataset.csv')
    purkinje_df = pd.read_csv('cerebellar_datasets/kan_purkinje_dataset.csv')
    nuclei_df = pd.read_csv('cerebellar_datasets/kan_nuclei_dataset.csv')

    # Crear datasets combinados para modelar conexiones
    # Ejemplo 1: Mossy -> Granule -> Purkinje
    X_mossy = mossy_df[['mean_firing_rate', 'burstiness']].values[:100]
    #X_granule = granule_df[['mean_firing_rate', 'voltage_mean']].values[:100]
    X_granule = granule_df[['mean_firing_rate', 'burstiness']].values[:100]
    
    X_combined = np.hstack([X_mossy, X_granule])
    y_purkinje = purkinje_df['mean_firing_rate'].values[:100]


    # Ejemplo 2: Señal de error (climbing fibers) + actividad Purkinje
    X_purkinje = purkinje_df[['mean_firing_rate', 'voltage_mean']].values[:100]
    #X_purkinje = purkinje_df[['mean_firing_rate', 'burstiness']].values[:100]

    X_error = climbing_df[['mean_firing_rate']].values[:100]
    
    X_learn = np.hstack([X_purkinje, X_error])
    y_learn = purkinje_df['voltage_max'].values[:100] - purkinje_df['voltage_min'].values[:100]

    # Normalización
    scaler1 = StandardScaler()
    X_combined_scaled = scaler1.fit_transform(X_combined)
    
    scaler2 = StandardScaler()
    X_learn_scaled = scaler2.fit_transform(X_learn)

    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined_scaled, y_purkinje, test_size=0.2, random_state=42
    )
    
    Xl_train, Xl_test, yl_train, yl_test = train_test_split(
        X_learn_scaled, y_learn, test_size=0.2, random_state=42
    )

    return {
        'train_data': (X_train, y_train),
        'test_data': (X_test, y_test),
        'learn_train': (Xl_train, yl_train),
        'learn_test': (Xl_test, yl_test),
        'scalers': (scaler1, scaler2),
        'all_dfs': {
            'mossy': mossy_df,
            'granule': granule_df,
            'golgi': golgi_df,
            'basket': basket_df,
            'stellate': stellate_df,
            'climbing': climbing_df,
            'purkinje': purkinje_df,
            'nuclei': nuclei_df
        }
    }

## ----------------------------
## 2. Modelado con KAN
## ----------------------------

def build_kan_model(input_dim, output_dim=1):
    """Construye un modelo KAN para el cerebelo"""
    model = KAN(
        layers=[input_dim, 128, 64, output_dim],
        activation_fun='silu',
        grid_size=5,
        num_epochs=50,
        lr=1e-3,
        batch_size=32
    )
    return model

def build_learning_model(input_dim, output_dim=1):
    """Modelo para el aprendizaje tipo cerebelo"""
    model = KAN(
        layers=[input_dim, 64, 32, output_dim],
        activation_fun='tanh',
        grid_size=5,
        num_epochs=100,
        lr=1e-4,
        batch_size=16
    )
    return model

## ----------------------------
## 3. Entrenamiento y Evaluación
## ----------------------------

def train_and_evaluate(models, data):
    """Entrena y evalúa los modelos KAN"""
    # Modelo principal
    main_model = models['main']
    X_train, y_train = data['train_data']
    X_test, y_test = data['test_data']
    
    print("\nEntrenando modelo principal...")
    history = main_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        steps=1000,
        metrics=['mse']
    )
    
    # Modelo de aprendizaje
    learn_model = models['learning']
    Xl_train, yl_train = data['learn_train']
    Xl_test, yl_test = data['learn_test']
    
    print("\nEntrenando modelo de aprendizaje...")
    learn_history = learn_model.fit(
        Xl_train, yl_train,
        validation_data=(Xl_test, yl_test),
        steps=2000,
        metrics=['mse']
    )
    
    return {
        'main_history': history,
        'learn_history': learn_history,
        'main_test_loss': main_model.evaluate(X_test, y_test),
        'learn_test_loss': learn_model.evaluate(Xl_test, yl_test)
    }

## ----------------------------
## 4. Visualización de Resultados
## ----------------------------

def plot_results(histories, test_losses):
    """Visualiza los resultados del entrenamiento"""
    plt.figure(figsize=(15, 6))
    
    # Pérdidas del modelo principal
    plt.subplot(1, 2, 1)
    plt.plot(histories['main_history']['train_loss'], label='Train Loss')
    plt.plot(histories['main_history']['val_loss'], label='Validation Loss')
    plt.title(f'Modelo Principal\nTest MSE: {test_losses["main_test_loss"]:.4f}')
    plt.xlabel('Steps')
    plt.ylabel('MSE')
    plt.legend()
    
    # Pérdidas del modelo de aprendizaje
    plt.subplot(1, 2, 2)
    plt.plot(histories['learn_history']['train_loss'], label='Train Loss')
    plt.plot(histories['learn_history']['val_loss'], label='Validation Loss')
    plt.title(f'Modelo de Aprendizaje\nTest MSE: {test_losses["learn_test_loss"]:.4f}')
    plt.xlabel('Steps')
    plt.ylabel('MSE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('cerebellar_kan_results.png')
    plt.show()

## ----------------------------
## 5. Pipeline Completo
## ----------------------------

def main():
    # 1. Cargar y preparar datos
    print("Cargando y preparando datos...")
    data = load_and_preprocess_data()
    
    # 2. Construir modelos
    print("Construyendo modelos KAN...")
    input_dim = data['train_data'][0].shape[1]
    learn_input_dim = data['learn_train'][0].shape[1]
    
    models = {
        'main': build_kan_model(input_dim),
        'learning': build_learning_model(learn_input_dim)
    }
    
    # 3. Entrenar y evaluar
    print("Iniciando entrenamiento...")
    results = train_and_evaluate(models, data)
    
    # 4. Visualizar resultados
    print("Visualizando resultados...")
    plot_results(results, {
        'main_test_loss': results['main_test_loss'],
        'learn_test_loss': results['learn_test_loss']
    })
    
    # 5. Guardar modelos
    print("Guardando modelos...")
    models['main'].save('cerebellar_main_kan.h5')
    models['learning'].save('cerebellar_learning_kan.h5')
    
    # Guardar scalers
    with open('cerebellar_scalers.pkl', 'wb') as f:
        pickle.dump(data['scalers'], f)
    
    print("¡Proceso completado con éxito!")

if __name__ == "__main__":
    main()