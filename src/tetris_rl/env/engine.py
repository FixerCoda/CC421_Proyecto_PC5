"""Motor de colisión.

Comprueba si una pieza puede moverse/rotar sin salir del tablero ni solapar
bloques fijados. Es la única autoridad sobre la validez de los movimientos.
"""

from __future__ import annotations

from .board import Board
from .pieces import Piece


class CollisionEngine:
    """Validación de movimientos y rotaciones de una pieza sobre el tablero."""

    def is_valid_move(self, piece: Piece, board: Board, dx: int, dy: int) -> bool:
        """Valida el desplazamiento (dx, dy) de una pieza.
        
        Considera bordes del tablero y solapamiento con bloques ya fijados.
        """
        shape = piece.current_shape()
        h, w = shape.shape
        for r in range(h):
            for c in range(w):
                if shape[r, c] == 0:
                    continue
                x = piece.x + dx + c
                y = piece.y + dy + r
                if x < 0 or x >= board.cols or y < 0 or y >= board.rows:
                    return False
                if board.grid[y, x] != 0:
                    return False
        return True

    def is_valid_rotation(self, piece: Piece, board: Board, direction: int) -> bool:
        """Valida la rotación (+1 derecha, -1 izquierda).

        Rota temporalmente, valida en el sitio y revierte.
        """
        if direction == 1:
            piece.rotate_right()
            ok = self.is_valid_move(piece, board, 0, 0)
            piece.rotate_left()
        elif direction == -1:
            piece.rotate_left()
            ok = self.is_valid_move(piece, board, 0, 0)
            piece.rotate_right()
        else:
            ok = False
        return ok
