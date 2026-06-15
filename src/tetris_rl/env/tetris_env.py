"""Entorno de Tetris de 1 jugador con API estilo Gymnasium.

Versión inicial por frame:
- En cada step se aplica una acción de control (mover/rotar) y luego un 
tick de gravedad.
- La pieza se fija al aterrizar y se genera la siguiente desde un RNG 
sembrado (reproducible).
"""

from __future__ import annotations

from .board import Board
from .engine import CollisionEngine
from .pieces import NAMES, PIECES, Piece

import numpy as np

# Acciones de control por frame.
NOOP, LEFT, RIGHT, ROTATE_CW, ROTATE_CCW, SOFT_DROP = range(6)

Observation = dict[str, object]
StepResult = tuple[Observation, int, bool, bool, dict[str, int]]


class TetrisEnv:
    """Entorno episódico de Tetris (1 agente, tablero rows x cols)."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, rows: int = 20, cols: int = 10, seed: int | None = None):
        self.rows = rows
        self.cols = cols
        self.engine = CollisionEngine()
        self._rng = np.random.default_rng(seed)
        # El tablero siempre existe (se reasigna en reset) y la pieza puede ser
        # None tras un top-out.
        self.board: Board = Board(rows, cols)
        self.piece: Piece | None = None
        self.lines_cleared = 0
        self.done = False

    # Ciclo de vida 
    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.board = Board(self.rows, self.cols)
        self.lines_cleared = 0
        self.done = False
        self.piece = self._spawn_piece()
        if self.piece is None:
            self.done = True
        return self._observation()

    def step(self, action: int) -> StepResult:
        if self.done or self.piece is None:
            return self._observation(), 0, True, False, {"lines": 0}

        piece = self.piece  # local no-None: estable ante llamadas a métodos
        self._apply_control(piece, action)
        reward = self._gravity_tick(piece)
        return self._observation(), reward, self.done, False, {"lines": reward}

    # Mecánica interna
    def _spawn_piece(self) -> Piece | None:
        name = NAMES[int(self._rng.integers(len(NAMES)))]
        shape_cols = PIECES[name][0].shape[1]
        x = (self.cols - shape_cols) // 2
        piece = Piece(name, x=x, y=0, rotation=0)
        if not self.engine.is_valid_move(piece, self.board, 0, 0):
            return None  # top-out: no cabe ni al aparecer
        return piece

    def _apply_control(self, piece: Piece, action: int) -> None:
        board, engine = self.board, self.engine
        if action == LEFT and engine.is_valid_move(piece, board, -1, 0):
            piece.move_left()
        elif action == RIGHT and engine.is_valid_move(piece, board, 1, 0):
            piece.move_right()
        elif action == ROTATE_CW and engine.is_valid_rotation(piece, board, 1):
            piece.rotate_right()
        elif action == ROTATE_CCW and engine.is_valid_rotation(piece, board, -1):
            piece.rotate_left()
        elif action == SOFT_DROP and engine.is_valid_move(piece, board, 0, 1):
            piece.move_down()

    def _gravity_tick(self, piece: Piece) -> int:
        """Baja la pieza un paso; si no puede, la fija y limpia líneas."""
        if self.engine.is_valid_move(piece, self.board, 0, 1):
            piece.move_down()
            return 0
        # aterriza: fijar bloques y limpiar líneas
        for (y, x) in piece.cells():
            self.board.lock_block(x, y, 1)
        cleared = self.board.clear_lines()
        self.lines_cleared += cleared
        self.piece = self._spawn_piece()
        if self.piece is None:
            self.done = True
        return cleared

    # Observación / Render
    def _observation(self) -> Observation:
        piece = self.piece
        return {
            "board": self.board.grid.copy(),
            "piece": None if piece is None else {
                "name": piece.name,
                "rotation": piece.rotation,
                "x": piece.x,
                "y": piece.y,
            },
        }

    def render(self) -> str:
        grid = self.board.grid.copy()
        piece = self.piece
        if piece is not None:
            for (y, x) in piece.cells():
                grid[y, x] = 2
        rows = ["|" + "".join("[]" if v else " ." for v in row) + "|" for row in grid]
        return "\n".join(rows) + f"\nLíneas: {self.lines_cleared}"
