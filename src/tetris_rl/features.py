"""6 features del afterstate de Tetris para la función de valor lineal del CEM.

Tablero como en env/: ndarray (rows, cols), 0 vacío / != 0 ocupado, fila 0
arriba. Las líneas vienen en lines_cleared porque el afterstate ya está limpio.

Variantes elegidas (para el informe): bumpiness = suma de |h_j - h_{j+1}| entre
columnas adyacentes; wells = suma de max(0, min(vecinas) - h_j) tratando las
paredes como vecino infinito, de modo que el pozo de borde también cuenta.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Orden fijo: el CEM, los baselines y los encabezados de CSV dependen de él.
FEATURE_NAMES: tuple[str, ...] = (
    "aggregate_height",
    "holes",
    "bumpiness",
    "lines_cleared",
    "max_height",
    "wells",
)

@njit(cache=True, fastmath=True)
def _extract_features_numba(grid: np.ndarray, lines_cleared: int) -> np.ndarray:
    """Extrae las 6 características en una sola pasada usando bucles nativos (C)."""
    rows, cols = grid.shape
    
    # Pre-reservar variables
    heights = np.zeros(cols, dtype=np.int32)
    holes = 0
    aggregate_height = 0
    max_height = 0
    
    # 1. Alturas, Huecos, Max Height y Aggregate Height
    for c in range(cols):
        block_found = False
        for r in range(rows):
            if grid[r, c] != 0:
                if not block_found:
                    h = rows - r
                    heights[c] = h
                    aggregate_height += h
                    if h > max_height:
                        max_height = h
                    block_found = True
            else:
                if block_found:
                    holes += 1
                    
    # 2. Bumpiness (Irregularidad)
    bumpiness = 0
    for c in range(cols - 1):
        bumpiness += abs(heights[c] - heights[c + 1])
        
    # 3. Wells (Pozos)
    wells = 0
    for c in range(cols):
        # Paredes laterales se consideran de altura infinita (usamos 999999)
        left = 999999 if c == 0 else heights[c - 1]
        right = 999999 if c == cols - 1 else heights[c + 1]
        
        min_neighbor = left if left < right else right
        if min_neighbor > heights[c]:
            wells += min_neighbor - heights[c]
            
    # 4. Construir y retornar el vector final
    out = np.zeros(6, dtype=np.float64)
    out[0] = aggregate_height
    out[1] = holes
    out[2] = bumpiness
    out[3] = lines_cleared
    out[4] = max_height
    out[5] = wells
    
    return out


def extract_features(afterstate: np.ndarray, lines_cleared: int) -> np.ndarray:
    """Vector float64 de 6 features en el orden de FEATURE_NAMES.
    
    Actúa como puente hacia la función optimizada JIT.
    """
    return _extract_features_numba(afterstate, lines_cleared)