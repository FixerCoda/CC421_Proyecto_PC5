"""Entorno de Tetris de 1 jugador, formulación por colocación.

Refactor desde la versión por-frame:
- En lugar de controlar la pieza frame a frame, el agente elige una 
colocación (rotación, columna).
- La pieza cae en hard-drop, se fija y se limpian las líneas. 
- Cada decisión produce un afterstate (el tablero resultante), que es lo 
que evalúa la política RL. Es la formulación natural y eficiente para 
Tetris basado en características (una decisión por pieza).
"""

from __future__ import annotations

from dataclasses import dataclass

from .board import Board
from .engine import is_valid_move, drop, lock_and_clear
from .pieces import NAMES, PIECES, Piece, num_rotations

import numpy as np


@dataclass(frozen=True)
class Placement:
    """Una colocación posible de la pieza actual y su afterstate resultante."""

    rotation: int
    x: int
    landing_y: int
    afterstate: np.ndarray  # tablero tras fijar la pieza y limpiar líneas
    lines: int              # líneas eliminadas por esta colocación


State = dict[str, object]
StepResult = tuple[State, float, bool, dict[str, int]]


class TetrisEnv:
    """Entorno episódico de Tetris (1 agente, tablero rows x cols)."""

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, rows: int = 20, cols: int = 10, seed: int | None = None):
        self.rows = rows
        self.cols = cols
        self._rng = np.random.default_rng(seed)
        # El tablero siempre existe (se reasigna en reset) y la pieza puede ser
        # None tras un top-out.
        self.board: Board = Board(rows, cols)
        self.piece: Piece | None = None
        self.lines_cleared = 0
        self.done = False

    # Ciclo de vida
    def reset(self, seed: int | None = None) -> State:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.board = Board(self.rows, self.cols)
        self.lines_cleared = 0
        self.done = False
        self.piece = self._spawn_piece()
        if self.piece is None or not self.legal_placements():
            self.done = True
        return self._state()

    def legal_placements(self) -> list[Placement]:
        """Colocaciones válidas de la pieza actual, con su afterstate y líneas.

        Para cada rotación y columna factible, deja caer la pieza en hard-drop,
        la fija sobre una copia del tablero y limpia líneas. El resultado es el
        afterstate que la política RL evaluará con φ.
        """
        piece = self.piece
        if piece is None:
            return []
        
        placements: list[Placement] = []
        name = piece.name
        board_grid = self.board.grid
        
        for rotation in range(num_rotations(name)):
            shape = PIECES[name][rotation]
            width = shape.shape[1]
            
            for x in range(self.cols - width + 1):
                # 1. Chequeo de validez inicial vía Numba
                if not is_valid_move(board_grid, shape, x, 0):
                    continue
                
                # 2. Hard-drop delegando el bucle a Numba
                landing_y = drop(board_grid, shape, x, 0)
                
                # 3. Fijar y limpiar delegando la mutación de arrays a Numba
                afterstate, lines = lock_and_clear(board_grid, shape, x, landing_y)
                
                placements.append(
                    Placement(rotation=rotation, x=x, landing_y=landing_y,
                              afterstate=afterstate, lines=lines)
                )
        return placements

    def step(self, placement: Placement) -> StepResult:
        """Aplica una colocación: adopta su afterstate y genera la pieza siguiente."""
        if self.done:
            return self._state(), 0.0, True, {"lines": 0}
        self.board = Board(grid=placement.afterstate)
        self.lines_cleared += placement.lines
        self.piece = self._spawn_piece()
        if self.piece is None or not self.legal_placements():
            self.done = True
        # Recompensa = líneas eliminadas por la colocación: R(s, a) = L(s, a).
        reward = float(placement.lines)
        return self._state(), reward, self.done, {"lines": placement.lines}

    # Mecánica interna
    def _spawn_piece(self) -> Piece | None:
        name = NAMES[int(self._rng.integers(len(NAMES)))]
        shape = PIECES[name][0]
        width = shape.shape[1]
        x = (self.cols - width) // 2
        
        # Validación inicial usando Numba sobre la matriz primitiva
        if not is_valid_move(self.board.grid, shape, x, 0):
            return None  # top-out: no cabe ni al aparecer
            
        return Piece(name, x=x, y=0, rotation=0)

    # Observación / Render
    def _state(self) -> State:
        piece = self.piece
        return {
            "board": self.board.grid.copy(),
            "piece": None if piece is None else piece.name,
            "done": self.done,
            "lines_cleared": self.lines_cleared,
        }

    def render(self) -> str:
        grid = self.board.grid
        header = f"Pieza: {self.piece.name if self.piece is not None else '-'}"
        rows = ["|" + "".join("[]" if v else " ." for v in row) + "|" for row in grid]
        return header + "\n" + "\n".join(rows) + f"\nLíneas: {self.lines_cleared}"