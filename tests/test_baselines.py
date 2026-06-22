"""Tests de los agentes baseline."""

import numpy as np

from tetris_rl.agents import HeuristicAgent, RandomAgent
from tetris_rl.env.tetris_env import Placement


def _placement(board, lines=0, x=0):
    # landing_y/rotation no influyen en los baselines; valores de relleno.
    return Placement(rotation=0, x=x, landing_y=0,
                     afterstate=np.asarray(board, dtype=np.int8), lines=lines)


def _empty(rows=4, cols=3):
    return np.zeros((rows, cols), dtype=np.int8)


def test_random_empty_returns_none():
    assert RandomAgent(seed=0).select([]) is None


def test_random_returns_member():
    ps = [_placement(_empty(), x=i) for i in range(5)]
    chosen = RandomAgent(seed=0).select(ps)
    assert chosen in ps


def test_random_reproducible_with_seed():
    ps = [_placement(_empty(), x=i) for i in range(5)]
    a, b = RandomAgent(seed=42), RandomAgent(seed=42)
    seq_a = [a.select(ps).x for _ in range(10)]
    seq_b = [b.select(ps).x for _ in range(10)]
    assert seq_a == seq_b


def test_heuristic_empty_returns_none():
    assert HeuristicAgent().select([]) is None


def test_heuristic_prefers_clean_low_board():
    # Bueno: un bloque abajo (h=1, sin huecos). Malo: columna alta con un hueco.
    good = _placement([[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0],
                       [1, 0, 0]])
    bad = _placement([[1, 0, 0],
                      [0, 0, 0],   # hueco tapado
                      [1, 0, 0],
                      [1, 0, 0]])
    assert HeuristicAgent().select([bad, good]) is good


def test_heuristic_rewards_line_clears():
    # Mismo afterstate; solo cambia lines. El peso negativo de lines_cleared
    # debe hacer que gane la que limpió líneas.
    no_clear = _placement(_empty(), lines=0)
    cleared = _placement(_empty(), lines=2)
    assert HeuristicAgent().select([no_clear, cleared]) is cleared
