"""Script principal para entrenar el agente TD(0) en Tetris.

Ejecutar desde la raíz del proyecto:
$ python train_td.py
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Importamos nuestro entorno y el agente
from src.tetris_rl.env.tetris_env import TetrisEnv
from src.tetris_rl.agents.td_agent import TDAgent
from src.tetris_rl.features import FEATURE_NAMES

def train(
    episodes: int = 2000,
    alpha: float = 0.001,
    gamma: float = 0.99,
    initial_epsilon: float = 0.5,
    min_epsilon: float = 0.01,
    epsilon_decay_episodes: int = 1500
):
    print(f"--- Iniciando Entrenamiento TD(0) por {episodes} episodios ---")
    
    env = TetrisEnv()
    agent = TDAgent(alpha=alpha, gamma=gamma, epsilon=initial_epsilon)
    
    # Historial para estadísticas
    history_lines = []
    history_weights = []
    start_time = time.time()
    
    for episode in range(1, episodes + 1):
        env.reset()
        
        # 1. Obtenemos las opciones iniciales
        placements = env.legal_placements()
        
        # 2. Elegimos la primera acción (A_t)
        action = agent.select(placements)
        
        while not env.done and action is not None:
            # Extraemos las características del afterstate actual (S_t)
            phi_current = agent._get_phi(action)
            
            # 3. Ejecutamos la acción en el entorno
            _, reward, done, _ = env.step(action)
            
            if done:
                # Si el juego terminó, no hay estado siguiente
                agent.update(phi_current, reward, phi_next=None)
                break
                
            # 4. Obtenemos el siguiente estado y elegimos la siguiente acción (S_{t+1}, A_{t+1})
            next_placements = env.legal_placements()
            next_action = agent.select(next_placements)
            
            if next_action is None:
                # Top-out (no hay jugadas legales)
                agent.update(phi_current, reward, phi_next=None)
                break
                
            phi_next = agent._get_phi(next_action)
            
            # 5. Aplicamos la regla TD(0) / SARSA
            agent.update(phi_current, reward, phi_next)
            
            # Avanzamos al siguiente paso
            action = next_action
            
        # Registramos las estadísticas del episodio
        history_lines.append(env.lines_cleared)
        history_weights.append(agent.weights.copy())
        
        # Decaimiento lineal de Epsilon (explorar menos, explotar más)
        if episode <= epsilon_decay_episodes:
            agent.epsilon = initial_epsilon - (initial_epsilon - min_epsilon) * (episode / epsilon_decay_episodes)
        
        # Logs en consola cada 100 episodios
        if episode % 100 == 0:
            avg_lines = np.mean(history_lines[-100:])
            max_lines = np.max(history_lines[-100:])
            weights_str = ", ".join([f"{w:5.2f}" for w in agent.weights])
            print(f"Episodio {episode:4d} | Líneas (Promedio: {avg_lines:5.1f} | Max: {max_lines:4d}) | eps: {agent.epsilon:.3f}")
            print(f"   Pesos: [{weights_str}]")
            
    # Al finalizar, guardamos los pesos aprendidos
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "td_weights.npy")
    agent.save_weights(save_path)
    
    elapsed = time.time() - start_time
    print(f"\n--- Entrenamiento Finalizado en {elapsed:.1f} segundos ---")
    print(f"Pesos finales guardados en: {save_path}")

    # --- Generación de Gráficas ---
    print("Generando gráficas de entrenamiento...")
    os.makedirs("plots", exist_ok=True)
    
    # Gráfica 1: Curva de Aprendizaje (Líneas)
    plt.figure(figsize=(10, 5))
    plt.plot(history_lines, alpha=0.3, color='blue', label='Líneas (bruto)')
    window = 50
    if len(history_lines) >= window:
        ma = np.convolve(history_lines, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(history_lines)), ma, color='red', linewidth=2, label=f'Media móvil ({window})')
    plt.title('Curva de Aprendizaje: Líneas Limpiadas por Episodio')
    plt.xlabel('Episodios')
    plt.ylabel('Líneas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join("plots", "curva_aprendizaje.png"))
    plt.close()

    # Gráfica 2: Evolución de los Pesos
    plt.figure(figsize=(10, 5))
    weights_array = np.array(history_weights)
    for i, name in enumerate(FEATURE_NAMES):
        plt.plot(weights_array[:, i], label=name, linewidth=2)
    plt.title('Evolución de los Pesos (w) durante el Entrenamiento')
    plt.xlabel('Episodios')
    plt.ylabel('Valor del Peso')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout() # Evita que la leyenda se corte al guardar
    plt.savefig(os.path.join("plots", "evolucion_pesos.png"))
    plt.close()
    
    print("Gráficas guardadas en el directorio 'plots/'.")

if __name__ == "__main__":
    # Reducimos alpha y aumentamos episodios, ya que ahora aprenderá más lento pero seguro
    train(episodes=2000, alpha=0.0001)