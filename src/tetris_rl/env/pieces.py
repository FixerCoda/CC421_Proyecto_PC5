"""Tetrominós estándar y la pieza en caída.

Se usan los 7 tetrominós clásicos para que las métricas sean comparables.
Cada tetrominó se define por sus matrices de rotación (0 = vacío, 
1 = ocupado), recortadas sin filas/columnas vacías.
"""

from __future__ import annotations

import numpy as np

# Matrices de rotación por pieza (orientación spawn primero).
PIECES: dict[str, list[np.ndarray]] = {
    "I": [
        np.array([[1, 1, 1, 1]], dtype=np.int8),
        np.array([[1], [1], [1], [1]], dtype=np.int8),
    ],
    "O": [
        np.array([[1, 1], [1, 1]], dtype=np.int8),
    ],
    "T": [
        np.array([[0, 1, 0], [1, 1, 1]], dtype=np.int8),
        np.array([[1, 0], [1, 1], [1, 0]], dtype=np.int8),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=np.int8),
        np.array([[0, 1], [1, 1], [0, 1]], dtype=np.int8),
    ],
    "S": [
        np.array([[0, 1, 1], [1, 1, 0]], dtype=np.int8),
        np.array([[1, 0], [1, 1], [0, 1]], dtype=np.int8),
    ],
    "Z": [
        np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int8),
        np.array([[0, 1], [1, 1], [1, 0]], dtype=np.int8),
    ],
    "J": [
        np.array([[1, 0, 0], [1, 1, 1]], dtype=np.int8),
        np.array([[1, 1], [1, 0], [1, 0]], dtype=np.int8),
        np.array([[1, 1, 1], [0, 0, 1]], dtype=np.int8),
        np.array([[0, 1], [0, 1], [1, 1]], dtype=np.int8),
    ],
    "L": [
        np.array([[0, 0, 1], [1, 1, 1]], dtype=np.int8),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=np.int8),
        np.array([[1, 1, 1], [1, 0, 0]], dtype=np.int8),
        np.array([[1, 1], [0, 1], [0, 1]], dtype=np.int8),
    ],
}

# Orden fijo de las 7 piezas (índices usados por el generador sembrado).
NAMES: list[str] = ["I", "O", "T", "S", "Z", "J", "L"]


def num_rotations(name: str) -> int:
    return len(PIECES[name])


class Piece:
    """Pieza en caída con posición (x, y) y rotación.

    Convención de coordenadas: x = columna (crece a la derecha),
    y = fila (crece hacia abajo). El origen de la matriz de la pieza se ancla
    en (x, y) (esquina superior izquierda).
    """

    def __init__(self, name: str, x: int = 0, y: int = 0, rotation: int = 0):
        self.name = name
        self.x = x
        self.y = y
        self.rotation = rotation % num_rotations(name)

    def current_shape(self) -> np.ndarray:
        """Matriz de la rotación actual."""
        return PIECES[self.name][self.rotation]

    @property
    def rows(self) -> int:
        return self.current_shape().shape[0]

    @property
    def cols(self) -> int:
        return self.current_shape().shape[1]

    def cells(self) -> list[tuple[int, int]]:
        """Celdas ocupadas en coordenadas absolutas (y, x) del tablero."""
        shape = self.current_shape()
        ys, xs = np.nonzero(shape)
        return [(self.y + int(r), self.x + int(c)) for r, c in zip(ys, xs)]

    def move_down(self) -> None:
        self.y += 1

    def move_left(self) -> None:
        self.x -= 1

    def move_right(self) -> None:
        self.x += 1

    def rotate_right(self) -> None:
        self.rotation = (self.rotation + 1) % num_rotations(self.name)

    def rotate_left(self) -> None:
        self.rotation = (self.rotation - 1) % num_rotations(self.name)

    def copy(self) -> "Piece":
        return Piece(self.name, self.x, self.y, self.rotation)
