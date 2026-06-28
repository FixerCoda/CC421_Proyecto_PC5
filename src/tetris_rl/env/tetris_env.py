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
from .engine import CollisionEngine
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
        self.engine = CollisionEngine()
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
        for rotation in range(num_rotations(name)):
            width = PIECES[name][rotation].shape[1]
            for x in range(self.cols - width + 1):
                candidate = Piece(name, x=x, y=0, rotation=rotation)
                if not self.engine.is_valid_move(candidate, self.board, 0, 0):
                    continue  # la pila llega arriba en esa columna
                while self.engine.is_valid_move(candidate, self.board, 0, 1):
                    candidate.move_down()
                after = self.board.copy()
                for (yy, xx) in candidate.cells():
                    after.lock_block(xx, yy, 1)
                lines = after.clear_lines()
                placements.append(
                    Placement(rotation=rotation, x=x, landing_y=candidate.y,
                              afterstate=after.grid, lines=lines)
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
        
        # --- Cimiento 1: Función de Recompensa R(s, a, s') ---
        # Recompensa por limpiar líneas (la convertimos a float para el descenso de gradiente)
        reward = float(placement.lines**2)
        
        # Penalización crítica: si la acción lo lleva a un estado terminal (game over)
        if self.done:
            reward -= 100.0
            
        return self._state(), reward, self.done, {"lines": placement.lines}

    # Mecánica interna
    def _spawn_piece(self) -> Piece | None:
        name = NAMES[int(self._rng.integers(len(NAMES)))]
        width = PIECES[name][0].shape[1]
        x = (self.cols - width) // 2
        piece = Piece(name, x=x, y=0, rotation=0)
        if not self.engine.is_valid_move(piece, self.board, 0, 0):
            return None  # top-out: no cabe ni al aparecer
        return piece

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
