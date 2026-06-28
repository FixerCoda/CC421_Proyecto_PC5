"""Script principal para entrenar el agente TD(0) en Tetris.
Versión: Paralelización Asíncrona preparda para Hyperparameter Sweeping.
"""

import os
import time
import ctypes
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from src.tetris_rl.env.tetris_env import TetrisEnv
from src.tetris_rl.agents.td_agent import TDAgent
from src.tetris_rl.features import FEATURE_NAMES

# Variables globales para apuntar a la RAM compartida
shared_weights_array = None
shared_lock = None

def init_worker(shared_arr, lock_obj):
    """Inicializa los punteros a la memoria compartida en cada núcleo."""
    global shared_weights_array
    global shared_lock
    shared_weights_array = shared_arr
    shared_lock = lock_obj

def play_episode_async(args):
    """Juega un episodio HASTA PERDER (sin límite) y sincroniza su aprendizaje."""
    global shared_weights_array, shared_lock
    assert shared_lock is not None
    assert shared_weights_array is not None
    
    episode_id, alpha, gamma, epsilon = args
    
    env = TetrisEnv()
    
    # 1. Leemos los pesos más recientes de la memoria compartida de forma segura
    with shared_lock:
        local_weights = np.array(shared_weights_array[:], dtype=np.float64)
        
    agent = TDAgent(alpha=alpha, gamma=gamma, epsilon=epsilon, weights=local_weights.copy())
    
    env.reset()
    placements = env.legal_placements()
    action = agent.select(placements)
    
    step_count = 0
    sync_frequency = 1000  # Sincronizamos aprendizaje cada 1000 piezas colocadas
    
    # Bucle de juego infinito (solo se rompe si el agente pierde)
    while not env.done and action is not None:
        step_count += 1
        phi_current = agent._get_phi(action)
        
        _, reward, done, _ = env.step(action)
        
        if done:
            agent.update(phi_current, reward, phi_next=None)
            break
            
        next_placements = env.legal_placements()
        next_action = agent.select(next_placements)
        
        if next_action is None:
            agent.update(phi_current, reward, phi_next=None)
            break
            
        phi_next = agent._get_phi(next_action)
        agent.update(phi_current, reward, phi_next)
        action = next_action
        
        # --- Sincronización en Caliente ---
        # Si el agente es muy bueno, transfiere su conocimiento sin detener la partida
        if step_count % sync_frequency == 0:
            delta_w = agent.weights - local_weights
            with shared_lock:
                for i in range(len(shared_weights_array)):
                    shared_weights_array[i] += delta_w[i]
                # Leemos la sabiduría combinada actual de todos los núcleos
                local_weights = np.array(shared_weights_array[:], dtype=np.float64)
            # Actualizamos la base del agente
            agent.weights = local_weights.copy()
            
    # Sincronización final al terminar (Game Over)
    delta_w = agent.weights - local_weights
    
    # 3. Sumamos nuestro Delta a los pesos globales directamente en RAM
    with shared_lock:
        for i in range(len(shared_weights_array)):
            shared_weights_array[i] += delta_w[i]
            
    return episode_id, env.lines_cleared, step_count

def train(
    total_episodes: int = 2000,
    initial_alpha: float = 0.0005,
    min_alpha: float = 0.00001,
    alpha_decay_episodes: int = 1500,
    gamma: float = 0.99,
    initial_epsilon: float = 0.5,
    min_epsilon: float = 0.01,
    epsilon_decay_episodes: int = 1500,
    run_dir: str = "runs/default"
) -> float:
    """
    Ejecuta un ciclo de entrenamiento completo.
    Retorna: El promedio de líneas limpiadas en los últimos 100 episodios.
    """
    num_cores = max(1, mp.cpu_count() - 1)
    print(f"\n--- Iniciando Entrenamiento Asíncrono en {num_cores} núcleos ---")
    print(f"Directorio de guardado: {run_dir}")
    print(f"Params: alpha={initial_alpha}->{min_alpha}, gamma={gamma}, eps={initial_epsilon}->{min_epsilon}")
    
    # Preparamos las subcarpetas para este experimento específico
    models_dir = os.path.join(run_dir, "models")
    plots_dir = os.path.join(run_dir, "plots")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    shared_arr = mp.Array(ctypes.c_double, len(FEATURE_NAMES))
    lock_obj = mp.Lock()
    
    # Decaimiento Lineal para Epsilon
    def get_eps(ep):
        if ep <= epsilon_decay_episodes:
            return initial_epsilon - (initial_epsilon - min_epsilon) * (ep / epsilon_decay_episodes)
        return min_epsilon

    # Decaimiento Exponencial para Alpha (mucho mejor para convergencia de pesos)
    def get_alpha(ep):
        if ep <= alpha_decay_episodes:
            decay_rate = np.log(min_alpha / initial_alpha) / alpha_decay_episodes
            return initial_alpha * np.exp(decay_rate * ep)
        return min_alpha

    worker_args = [
        (ep, get_alpha(ep), gamma, get_eps(ep)) 
        for ep in range(1, total_episodes + 1)
    ]
    
    history_lines = []
    history_weights = []
    start_time = time.time()
    completed_episodes = 0
    
    # Iniciamos el Pool inyectando la memoria compartida
    with mp.Pool(processes=num_cores, initializer=init_worker, initargs=(shared_arr, lock_obj)) as pool:
        
        # imap_unordered nos devuelve los resultados en el instante exacto en que un hilo termina
        for result in pool.imap_unordered(play_episode_async, worker_args):
            ep_id, lines, steps = result
            completed_episodes += 1
            
            history_lines.append(lines)
            
            # Guardamos el estado actual de los pesos para la gráfica
            with lock_obj:
                current_weights = np.array(shared_arr[:])
            history_weights.append(current_weights)
            
            if completed_episodes % 200 == 0:
                avg_lines = np.mean(history_lines[-100:])
                print(f"Episodios {completed_episodes-199:4d}-{completed_episodes:4d} | Líneas (Promedio: {avg_lines:5.1f}) | eps: {get_eps(completed_episodes):.3f} | alpha: {get_alpha(completed_episodes):.6f}")

    # Guardado final de pesos
    save_path = os.path.join(models_dir, "td_weights.npy")
    final_weights = np.array(shared_arr[:])
    np.save(save_path, final_weights)
    
    elapsed = time.time() - start_time
    print(f"--- Entrenamiento Finalizado en {elapsed:.1f} segundos ---")

    # --- Generación de Gráficas en el directorio específico ---
    plt.figure(figsize=(10, 5))
    plt.plot(history_lines, alpha=0.3, color='blue', label='Líneas (bruto)')
    window = 50
    if len(history_lines) >= window:
        ma = np.convolve(history_lines, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(history_lines)), ma, color='red', linewidth=2, label=f'Media móvil ({window})')
    plt.title(f'Curva de Aprendizaje - $\\gamma$={gamma}')
    plt.xlabel('Episodios Completados')
    plt.ylabel('Líneas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(plots_dir, "curva_aprendizaje.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    weights_array = np.array(history_weights)
    for i, name in enumerate(FEATURE_NAMES):
        plt.plot(weights_array[:, i], label=name, linewidth=2)
    plt.title('Evolución de los Pesos (w)')
    plt.xlabel('Episodios Completados')
    plt.ylabel('Valor del Peso')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "evolucion_pesos.png"))
    plt.close()

    # Retornamos el score de rendimiento (promedio de líneas de los últimos 100 episodios)
    final_score = np.mean(history_lines[-100:]) if len(history_lines) >= 100 else np.mean(history_lines)
    return final_score # type: ignore

if __name__ == "__main__":
    mp.freeze_support() 
    # Ejecución por defecto si se llama directamente al script
    train(total_episodes=5000, run_dir="runs/debug_run")