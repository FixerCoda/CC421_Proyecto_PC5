"""Pipeline de generación: corre N partidas con un agente y guarda un CSV.

El CSV (uno por corrida) queda en data/raw/ vía TransitionLogger. Reproducible:
el RNG del entorno se siembra en el primer reset y luego el stream continúa, así
los episodios son deterministas pero distintos entre sí.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..agents import RandomAgent
from ..env.tetris_env import TetrisEnv
from .logger import TransitionLogger


def generate(agent, episodes: int, seed: int = 0, out_dir: str | Path = "data/raw",
             run_name: str | None = None, rows: int = 20, cols: int = 10,
             max_steps: int | None = None) -> Path:
    """Juega `episodes` partidas con `agent`, registra cada transición y devuelve la ruta del CSV.

    `max_steps` corta cada partida tras esa cantidad de colocaciones; útil para
    agentes fuertes (TD/CEM), que de otro modo jugarían partidas casi infinitas.
    """
    env = TetrisEnv(rows=rows, cols=cols)
    if run_name is None:
        run_name = f"{type(agent).__name__}_seed{seed}_ep{episodes}"
    path = Path(out_dir) / f"{run_name}.csv"

    with TransitionLogger(path) as logger:
        for episode in range(episodes):
            state = env.reset(seed=seed if episode == 0 else None)
            done = bool(state["done"])
            step = 0
            while not done:
                if max_steps is not None and step >= max_steps:
                    break
                placements = env.legal_placements()
                if not placements:
                    break
                piece = env.piece.name  # la pieza que se va a colocar
                placement = agent.select(placements)
                state, reward, done, _ = env.step(placement)
                logger.log(episode, step, piece, placement, reward)
                step += 1
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un CSV de transiciones con RandomAgent.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="data/raw")
    args = parser.parse_args()

    path = generate(RandomAgent(seed=args.seed), episodes=args.episodes, seed=args.seed,
                    out_dir=args.out_dir)
    print(f"CSV guardado en {path}")


if __name__ == "__main__":
    main()
