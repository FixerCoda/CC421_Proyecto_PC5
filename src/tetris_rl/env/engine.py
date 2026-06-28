import numpy as np
from numba import njit

@njit(cache=True, fastmath=True)
def is_valid_move(grid, shape, x, y):
    """Verifica colisiones iterando sobre primitivas (compilado a C)."""
    h, w = shape.shape
    rows, cols = grid.shape
    for r in range(h):
        for c in range(w):
            if shape[r, c] != 0:
                board_x = x + c
                board_y = y + r
                if board_x < 0 or board_x >= cols or board_y < 0 or board_y >= rows:
                    return False
                if grid[board_y, board_x] != 0:
                    return False
    return True

@njit(cache=True)
def drop(grid, shape, x, start_y):
    """Calcula el y final (hard-drop) sin instanciar piezas."""
    y = start_y
    while is_valid_move(grid, shape, x, y + 1):
        y += 1
    return y

@njit(cache=True)
def lock_and_clear(grid, shape, x, y):
    """Fija la pieza y limpia líneas en una sola pasada, sin np.zeros_like."""
    # Clonamos el estado primitivo una sola vez
    new_grid = grid.copy()
    h, w = shape.shape
    
    # Fijar bloque
    for r in range(h):
        for c in range(w):
            if shape[r, c] != 0:
                new_grid[y + r, x + c] = 1

    # Limpiar líneas (in-place shift, mucho más rápido que np.all)
    rows, cols = new_grid.shape
    lines_cleared = 0
    write_row = rows - 1
    
    for read_row in range(rows - 1, -1, -1):
        is_full = True
        for c in range(cols):
            if new_grid[read_row, c] == 0:
                is_full = False
                break
        
        if is_full:
            lines_cleared += 1
        else:
            if write_row != read_row:
                for c in range(cols):
                    new_grid[write_row, c] = new_grid[read_row, c]
            write_row -= 1
            
    # Llenar el tope con ceros
    while write_row >= 0:
        for c in range(cols):
            new_grid[write_row, c] = 0
        write_row -= 1
        
    return new_grid, lines_cleared