"""
Script de Barrido de Hiperparámetros (Grid Search) para el Agente TD.
Ejecuta secuencialmente varios entrenamientos con distintas configuraciones
y registra cuál obtiene el mejor rendimiento promedio al final del entrenamiento.
"""

import os
import itertools
import multiprocessing as mp
import pandas as pd
from datetime import datetime

# Importamos la función train de nuestro script de entrenamiento
from train_td import train

def run_grid_search():
    # 1. Definimos la cuadrícula de búsqueda (Grid)
    # Cuidado: El número total de experimentos es la multiplicación de todas las opciones.
    param_grid = {
        'gamma': [0.9, 0.99, 0.999],                  # Corto plazo vs. Largo plazo extremo
        'initial_alpha': [0.001, 0.0005, 0.0001],     # Ritmos de aprendizaje iniciales
        'epsilon_decay_episodes': [1000, 1500, 1800]  # Exploración corta vs. larga
    }

    # Configuraciones constantes para todos los experimentos
    TOTAL_EPISODES = 5000
    MIN_ALPHA = 0.00001
    ALPHA_DECAY = 1500 # Qué tan rápido decae alpha

    # 2. Generamos todas las combinaciones posibles
    keys = param_grid.keys()
    combinations = list(itertools.product(*param_grid.values()))
    
    print(f"Iniciando Grid Search con {len(combinations)} combinaciones.")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_sweep_dir = f"runs/sweep_{timestamp}"
    os.makedirs(base_sweep_dir, exist_ok=True)

    results = []

    # 3. Iteramos sobre cada combinación
    for idx, values in enumerate(combinations):
        config = dict(zip(keys, values))
        
        # Generar un nombre único para la carpeta de esta configuración
        run_name = f"run_{idx+1:02d}_g{config['gamma']}_a{config['initial_alpha']}_epsDecay{config['epsilon_decay_episodes']}"
        run_dir = os.path.join(base_sweep_dir, run_name)
        
        print(f"\n{'='*60}")
        print(f"Experimento {idx+1}/{len(combinations)}: {run_name}")
        print(f"{'='*60}")
        
        # Llamamos al orquestador de entrenamiento
        score = train(
            total_episodes=TOTAL_EPISODES,
            initial_alpha=config['initial_alpha'],
            min_alpha=MIN_ALPHA,
            alpha_decay_episodes=ALPHA_DECAY,
            gamma=config['gamma'],
            epsilon_decay_episodes=config['epsilon_decay_episodes'],
            run_dir=run_dir
        )
        
        # Registramos los resultados
        config['run_name'] = run_name
        config['final_score'] = score
        results.append(config)
        
        # Guardamos progresivamente en caso de que se cancele a la mitad
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(base_sweep_dir, "sweep_results.csv"), index=False)
        
    # 4. Resumen Final
    print(f"\n{'*'*60}")
    print("BARRIDO DE HIPERPARÁMETROS FINALIZADO")
    print(f"{'*'*60}")
    
    # Ordenar resultados de mejor a peor
    best_results = df.sort_values(by='final_score', ascending=False) # type: ignore
    print("\nTop 3 Mejores Configuraciones:")
    print(best_results.head(3).to_string(index=False))
    print(f"\nTodos los datos y gráficas guardados en: {base_sweep_dir}/")

if __name__ == "__main__":
    # Importante en Windows/WSL para evitar errores de multiprocessing
    mp.freeze_support()
    run_grid_search()