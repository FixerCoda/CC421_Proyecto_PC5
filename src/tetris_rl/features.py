"""6 features del afterstate de Tetris para la función de valor lineal del CEM.

Tablero como en env/: ndarray (rows, cols), 0 vacío / != 0 ocupado, fila 0
arriba. Las líneas vienen en lines_cleared porque el afterstate ya está limpio.

Variantes elegidas (para el informe): bumpiness = suma de |h_j - h_{j+1}| entre
columnas adyacentes; wells = suma de max(0, min(vecinas) - h_j) tratando las
paredes como vecino infinito, de modo que el pozo de borde también cuenta.
"""

from __future__ import annotations

import numpy as np

# Orden fijo: el CEM, los baselines y los encabezados de CSV dependen de él.
FEATURE_NAMES: tuple[str, ...] = (
    "aggregate_height",
    "holes",
    "bumpiness",
    "lines_cleared",
    "max_height",
    "wells",
)


def column_heights(occupied: np.ndarray) -> np.ndarray:
    """Altura de cada columna desde la máscara de ocupación (vacía -> 0)."""
    rows = occupied.shape[0]
    any_occ = occupied.any(axis=0)
    first_occ = occupied.argmax(axis=0)  # primera fila ocupada, 0 si no hay
    return np.where(any_occ, rows - first_occ, 0).astype(np.int64)


def extract_features(afterstate: np.ndarray, lines_cleared: int) -> np.ndarray:
    """Vector float64 de 6 features en el orden de FEATURE_NAMES."""
    grid = np.asarray(afterstate)
    occupied = grid != 0
    heights = column_heights(occupied)

    aggregate_height = int(heights.sum())
    max_height = int(heights.max()) if heights.size else 0

    # Hueco: celda vacía con alguna ocupada por encima en su columna.
    covered = np.maximum.accumulate(occupied, axis=0)
    holes = int((~occupied & covered).sum())

    bumpiness = int(np.abs(np.diff(heights)).sum()) if heights.size > 1 else 0

    # Pozos: paredes = vecino de altura infinita.
    inf = np.iinfo(np.int64).max
    left = np.concatenate(([inf], heights[:-1])) if heights.size else heights
    right = np.concatenate((heights[1:], [inf])) if heights.size else heights
    neighbor_min = np.minimum(left, right)
    wells = int(np.maximum(0, neighbor_min - heights).sum()) if heights.size else 0

    return np.array(
        [aggregate_height, holes, bumpiness, int(lines_cleared), max_height, wells],
        dtype=np.float64,
    )
