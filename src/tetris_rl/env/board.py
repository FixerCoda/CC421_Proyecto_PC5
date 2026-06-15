"""Tablero de Tetris.

El tablero es una matriz rows x cols de enteros (0 = celda vacía, cualquier
valor != 0 = celda ocupada).
"""

from __future__ import annotations

import numpy as np


class Board:
    """Tablero estático de Tetris (bloques ya fijados)."""

    def __init__(self, rows: int = 20, cols: int = 10, grid: np.ndarray | None = None):
        if grid is not None:
            self.grid = np.asarray(grid, dtype=np.int8).copy()
            self.rows, self.cols = self.grid.shape
        else:
            self.rows = rows
            self.cols = cols
            self.grid = np.zeros((rows, cols), dtype=np.int8)

    @classmethod
    def from_grid(cls, grid) -> "Board":
        """Construye un tablero desde una matriz dada."""
        return cls(grid=np.asarray(grid, dtype=np.int8))

    def clear_lines(self) -> int:
        """Elimina las filas completas, desplaza el resto hacia abajo.

        Devuelve el número de líneas eliminadas.
        """
        full = np.all(self.grid != 0, axis=1)
        n = int(full.sum())
        if n:
            kept = self.grid[~full]
            new = np.zeros_like(self.grid)
            new[n:] = kept  # las filas conservadas caen al fondo
            self.grid = new
        return n

    def lock_block(self, x: int, y: int, value: int = 1) -> None:
        """Fija un bloque en la columna x, fila y."""
        self.grid[y, x] = value

    def cell(self, x: int, y: int) -> int:
        """Estado de la celda en columna x, fila y."""
        return int(self.grid[y, x])

    def is_empty(self, x: int, y: int) -> bool:
        return self.grid[y, x] == 0

    def copy(self) -> "Board":
        return Board(grid=self.grid)

    def __eq__(self, other) -> bool:
        return isinstance(other, Board) and np.array_equal(self.grid, other.grid)

    def __repr__(self) -> str:
        return f"Board(rows={self.rows}, cols={self.cols})"
