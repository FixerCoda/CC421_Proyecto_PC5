"""Entrena la función de valor lineal por CEM y guarda models/cem_weights.npy.

Entrypoint simétrico a train_td.py: la lógica está en tetris_rl.agents.cem.
"""

import argparse
import os

import numpy as np

from tetris_rl.agents.cem import evaluate, train_cem
from tetris_rl.env.tetris_env import TetrisEnv
from tetris_rl.features import FEATURE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena pesos lineales por CEM.")
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=5, help="partidas por candidato")
    parser.add_argument("--max-steps", type=int, default=500, help="tope de colocaciones por partida")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    weights, history = train_cem(generations=args.generations, population=args.population,
                                 episodes_per_eval=args.episodes, max_steps=args.max_steps,
                                 seed=args.seed)

    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, "cem_weights.npy")
    np.save(save_path, weights)

    # Fitness final de los pesos consolidados (semillas nuevas, evaluación honesta).
    env = TetrisEnv()
    final_seeds = range(1000, 1000 + 20)
    final_fitness = evaluate(weights, env, final_seeds, max_steps=args.max_steps)

    print(f"Pesos guardados en {save_path}")
    print("Pesos por feature:")
    for name, w in zip(FEATURE_NAMES, weights):
        print(f"  {name:18s} {w:+.4f}")
    print(f"Líneas promedio (20 partidas de evaluación): {final_fitness:.1f}")


if __name__ == "__main__":
    main()
