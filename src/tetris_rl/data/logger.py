"""Logger a CSV de transiciones, una fila por colocación.

Columnas: episode, step, piece, rotation, x, reward + las 6 FEATURE_NAMES.
lines_cleared no se repite: va una sola vez como feature #3 (= placement.lines).
"""

from __future__ import annotations

import csv
from pathlib import Path

from ..env.tetris_env import Placement
from ..features import FEATURE_NAMES, extract_features

FIELDS = ["episode", "step", "piece", "rotation", "x", "reward", *FEATURE_NAMES]


class TransitionLogger:
    """Escribe transiciones a un CSV (una corrida por archivo). Usar con `with`."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._writer = None

    def __enter__(self) -> "TransitionLogger":
        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        return self

    def log(self, episode: int, step: int, piece: str,
            placement: Placement, reward: float) -> None:
        phi = extract_features(placement.afterstate, placement.lines)
        row = {
            "episode": episode,
            "step": step,
            "piece": piece,
            "rotation": placement.rotation,
            "x": placement.x,
            "reward": reward,
        }
        row.update(zip(FEATURE_NAMES, phi))
        self._writer.writerow(row)

    def __exit__(self, *exc) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
