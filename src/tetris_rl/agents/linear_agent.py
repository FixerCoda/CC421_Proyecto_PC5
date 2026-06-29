"""Agente lineal greedy: juega con un vector de pesos dado, sin aprender.

Elige la colocación que MAXIMIZA w·φ (convención de función de valor, igual que
el TDAgent). Lo usan el entrenador CEM para evaluar cada candidato de pesos y la
evaluación final de los agentes que aprenden (TD, CEM).
"""

from __future__ import annotations

import numpy as np

from ..env.tetris_env import Placement
from ..features import FEATURE_NAMES, extract_features


class LinearAgent:
    """Política greedy sobre la función de valor lineal V(s) = w·φ(s)."""

    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=np.float64)
        if self.weights.shape != (len(FEATURE_NAMES),):
            raise ValueError(f"weights debe tener longitud {len(FEATURE_NAMES)}")

    def value(self, placement: Placement) -> float:
        return float(self.weights @ extract_features(placement.afterstate, placement.lines))

    def select(self, placements: list[Placement]) -> Placement | None:
        if not placements:
            return None
        values = [self.value(p) for p in placements]
        return placements[int(np.argmax(values))]
