"""Script para evaluar visualmente y sin restricciones al agente entrenado.

Ejecutar:
$ python play_model.py
"""

import time
import numpy as np

from src.tetris_rl.env.tetris_env import TetrisEnv
from src.tetris_rl.agents.td_agent import TDAgent

def evaluate_model(weights_path: str = "models/td_weights.npy", delay: float = 0.05):
    print(f"Cargando pesos desde: {weights_path}...")
    try:
        weights = np.load(weights_path)
    except FileNotFoundError:
        print("Error: No se encontró el archivo de pesos. Entrena el modelo primero.")
        return

    # Creamos el entorno y el agente (epsilon=0.0 significa 100% explotación, nada al azar)
    env = TetrisEnv()
    agent = TDAgent(epsilon=0.0, weights=weights)
    
    env.reset()
    step = 0
    
    print("\nIniciando partida en 3 segundos...")
    time.sleep(3)
    
    while not env.done:
        step += 1
        placements = env.legal_placements()
        
        # El agente elige la mejor jugada
        action = agent.select(placements)
        
        if action is None:
            break
            
        # Ejecutamos la acción
        env.step(action)
        
        # Imprimimos el estado actual en la consola
        # (Usamos secuencias de escape ANSI para borrar la pantalla y crear efecto de animación)
        print("\033[H\033[J", end="") # Limpia la consola
        print(f"--- PASO {step} ---")
        print(env.render())
        
        # Pequeña pausa para que los humanos puedan ver el juego
        time.sleep(delay)

    print("\n===============================")
    print(f"GAME OVER")
    print(f"Líneas totales limpiadas: {env.lines_cleared}")
    print(f"Piezas colocadas: {step}")
    print("===============================")

if __name__ == "__main__":
    # Puedes bajar el delay a 0.0 para que juegue a velocidad máxima
    evaluate_model(delay=0.02)