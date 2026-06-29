"""Entrenador Cross-Entropy Method (CEM) para la función de valor lineal.

Optimiza los 6 pesos `w` sin gradientes: mantiene una gaussiana sobre `w`,
muestrea una población, evalúa cada candidato jugando con LinearAgent, conserva
la élite y reajusta la gaussiana hacia ella. Fitness = líneas completadas
promedio por episodio.

Para que la comparación entre candidatos sea justa, en cada generación todos se
evalúan con las mismas semillas de partida (números aleatorios comunes). Un
ruido extra que decae evita que la varianza colapse antes de tiempo
(Szita & Lőrincz, 2006).
"""

from __future__ import annotations

import numpy as np

from ..env.tetris_env import TetrisEnv
from ..features import FEATURE_NAMES
from .linear_agent import LinearAgent

N_FEATURES = len(FEATURE_NAMES)


def play_episode(env: TetrisEnv, agent: LinearAgent, seed: int | None = None) -> int:
    """Juega una partida greedy y devuelve las líneas completadas."""
    state = env.reset(seed=seed)
    done = bool(state["done"])
    while not done:
        placements = env.legal_placements()
        if not placements:
            break
        state, _, done, _ = env.step(agent.select(placements))
    return int(state["lines_cleared"])


def evaluate(weights, env: TetrisEnv, seeds) -> float:
    """Fitness = líneas promedio jugando con `weights` sobre las partidas `seeds`."""
    agent = LinearAgent(weights)
    return float(np.mean([play_episode(env, agent, s) for s in seeds]))


def train_cem(generations: int = 20, population: int = 30, elite_frac: float = 0.2,
              episodes_per_eval: int = 5, init_std: float = 1.0, extra_noise: float = 0.5,
              seed: int = 0, rows: int = 20, cols: int = 10):
    """Entrena los pesos por CEM. Devuelve (pesos_finales, history)."""
    rng = np.random.default_rng(seed)
    env = TetrisEnv(rows=rows, cols=cols)
    n_elite = max(1, round(population * elite_frac))

    mean = np.zeros(N_FEATURES)
    std = np.full(N_FEATURES, init_std)
    history = []

    for gen in range(generations):
        pop = rng.normal(mean, std, size=(population, N_FEATURES))
        # Semillas comunes para todos los candidatos de esta generación.
        ep_seeds = rng.integers(0, 2**31 - 1, size=episodes_per_eval)
        fitness = np.array([evaluate(w, env, ep_seeds) for w in pop])

        elite = pop[np.argsort(fitness)[-n_elite:]]
        mean = elite.mean(axis=0)
        # Ruido extra decreciente para no converger prematuramente.
        std = elite.std(axis=0) + max(0.0, extra_noise * (1 - gen / generations))

        history.append({"gen": gen, "best": float(fitness.max()),
                        "mean": float(fitness.mean())})

    return mean, history
