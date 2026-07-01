"""Compara agentes (aleatorio, heurístico, TD, CEM) bajo el mismo protocolo.

Carga models/td_weights.npy y models/cem_weights.npy, juega N partidas con las
mismas semillas y un tope de colocaciones, e imprime una tabla con líneas
promedio, desviación y máximo. También reporta la norma ‖w‖ de cada modelo
aprendido (evidencia la divergencia de TD frente al gauge ‖w‖=1 de CEM).

Entrypoint simétrico a train_td.py / train_cem.py.
"""

import argparse
import os

import numpy as np

from tetris_rl.agents.baselines import HeuristicAgent, RandomAgent
from tetris_rl.agents.cem import play_episode
from tetris_rl.agents.linear_agent import LinearAgent
from tetris_rl.env.tetris_env import TetrisEnv


def evaluate_agent(env, agent, seeds, max_steps):
    """Líneas (media, desviación, máximo) jugando greedy sobre las partidas `seeds`."""
    lines = [play_episode(env, agent, s, max_steps) for s in seeds]
    return float(np.mean(lines)), float(np.std(lines)), int(np.max(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compara agentes bajo el mismo protocolo.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=1000, help="tope de colocaciones por partida")
    parser.add_argument("--seed-base", type=int, default=0)
    args = parser.parse_args()

    here = os.path.dirname(__file__)
    td_path = os.path.join(here, "models", "td_weights.npy")
    cem_path = os.path.join(here, "models", "cem_weights.npy")

    seeds = list(range(args.seed_base, args.seed_base + args.episodes))
    env = TetrisEnv()

    agents = {"Aleatorio": RandomAgent(0), "Heuristico": HeuristicAgent()}
    if os.path.exists(td_path):
        agents["TD"] = LinearAgent(np.load(td_path))
    if os.path.exists(cem_path):
        agents["CEM"] = LinearAgent(np.load(cem_path))

    print(f"Protocolo: {args.episodes} partidas, tope {args.max_steps} colocaciones, política greedy")
    print(f"{'Agente':12s} {'media':>8s} {'desv':>8s} {'max':>6s}")
    for name, agent in agents.items():
        mean, std, mx = evaluate_agent(env, agent, seeds, args.max_steps)
        print(f"{name:12s} {mean:8.1f} {std:8.1f} {mx:6d}")

    for name, path in [("TD", td_path), ("CEM", cem_path)]:
        if os.path.exists(path):
            print(f"||w|| {name} = {np.linalg.norm(np.load(path)):.1f}")


if __name__ == "__main__":
    main()
