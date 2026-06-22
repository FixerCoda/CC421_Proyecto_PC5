"""Agentes baseline para comparar contra el CEM.

Ambos exponen select(placements) -> Placement | None, recibiendo la lista que
da env.legal_placements() y devolviendo None si está vacía (sin jugadas).
"""

from __future__ import annotations

import numpy as np

from ..env.tetris_env import Placement
from ..features import FEATURE_NAMES, extract_features

# Pesos del HeuristicAgent (se MINIMIZA pesos · features). Los cuatro primeros
# son los de El-Tetris (Lee, 2013) con el signo invertido por minimizar: altura,
# huecos y bumpiness penalizan, las líneas premian. max_height y wells (que no
# están en El-Tetris) llevan una penalización manual leve.
DEFAULT_WEIGHTS: dict[str, float] = {
    "aggregate_height": 0.51,
    "holes": 0.36,
    "bumpiness": 0.18,
    "lines_cleared": -0.76,
    "max_height": 0.30,
    "wells": 0.30,
}


class RandomAgent:
    """Elige una colocación al azar. Reproducible vía seed."""

    def __init__(self, seed: int | None = None):
        self._rng = np.random.default_rng(seed)

    def select(self, placements: list[Placement]) -> Placement | None:
        if not placements:
            return None
        return placements[int(self._rng.integers(len(placements)))]


class HeuristicAgent:
    """Elige la colocación que minimiza pesos · features (determinista)."""

    def __init__(self, weights: dict[str, float] | None = None):
        w = DEFAULT_WEIGHTS if weights is None else weights
        self.weights = np.array([w[name] for name in FEATURE_NAMES], dtype=np.float64)

    def score(self, placement: Placement) -> float:
        phi = extract_features(placement.afterstate, placement.lines)
        return float(self.weights @ phi)

    def select(self, placements: list[Placement]) -> Placement | None:
        if not placements:
            return None
        costs = [self.score(p) for p in placements]
        return placements[int(np.argmin(costs))]
