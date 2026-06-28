"""Agente de Aprendizaje por Diferencia Temporal (TD) Lineal.

Implementa un agente que aprende la función de valor aproximada
mediante descenso de gradiente semi-estocástico (TD(0)).
"""

from __future__ import annotations

import numpy as np

from ..env.tetris_env import Placement
from ..features import FEATURE_NAMES, extract_features


class TDAgent:
    """Aprende a jugar Tetris actualizando un vector de pesos lineales."""

    def __init__(
        self,
        alpha: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        weights: np.ndarray | None = None
    ):
        """
        Args:
            alpha: Tasa de aprendizaje (learning rate).
            gamma: Factor de descuento (cuánto le importa el futuro).
            epsilon: Probabilidad de tomar una acción aleatoria (exploración).
            weights: Vector de pesos inicial. Si es None, inicia en 0.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Si no nos dan pesos previos (ej. de models/), iniciamos desde cero.
        if weights is None:
            self.weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        else:
            self.weights = weights.astype(np.float64)

    def _get_phi(self, placement: Placement) -> np.ndarray:
        """Extrae el vector de características de un placement."""
        return extract_features(placement.afterstate, placement.lines)

    def predict(self, phi: np.ndarray) -> float:
        """Calcula el valor esperado del estado: V(s) = w^T * phi."""
        return float(self.weights @ phi)

    def select(self, placements: list[Placement]) -> Placement | None:
        """Elige una colocación usando la política epsilon-greedy."""
        if not placements:
            return None

        # 1. Exploración: A veces elegimos una jugada al azar
        if np.random.rand() < self.epsilon:
            return placements[int(np.random.randint(len(placements)))]

        # 2. Explotación: Elegimos la jugada con mayor V(s) predicho
        best_placement = None
        max_value = -float('inf')

        for p in placements:
            phi = self._get_phi(p)
            val = self.predict(phi)
            if val > max_value:
                max_value = val
                best_placement = p

        return best_placement

    def update(self, phi_current: np.ndarray, reward: float, phi_next: np.ndarray | None):
        """
        Aplica la regla de actualización TD(0) sobre los pesos.
        w <- w + alpha * [R + gamma * V(s') - V(s)] * phi(s)
        
        Args:
            phi_current: Características del estado del que venimos.
            reward: La recompensa numérica obtenida al hacer la transición.
            phi_next: Características del siguiente estado (None si perdimos).
        """
        v_current = self.predict(phi_current)

        if phi_next is None:
            # Estado terminal: el valor del siguiente estado es 0
            v_next = 0.0
        else:
            v_next = self.predict(phi_next)

        # Calculamos el objetivo y el error (delta)
        td_target = reward + self.gamma * v_next
        td_error = td_target - v_current
        
        # Recortamos el error TD para evitar Gradientes Explosivos
        # Limitamos la sorpresa a un rango de [-100, 100]
        td_error = np.clip(td_error, -100.0, 100.0)

        # Actualización por gradiente
        self.weights += self.alpha * td_error * phi_current

    def save_weights(self, filepath: str):
        """Guarda los pesos aprendidos (útil para guardar en models/)."""
        np.save(filepath, self.weights)

    def load_weights(self, filepath: str):
        """Carga pesos previamente entrenados."""
        self.weights = np.load(filepath)