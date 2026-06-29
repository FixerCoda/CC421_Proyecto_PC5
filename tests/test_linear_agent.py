"""Tests del LinearAgent (política greedy sobre w·φ)."""

import numpy as np
import pytest

from tetris_rl.agents import LinearAgent
from tetris_rl.env.tetris_env import Placement
from tetris_rl.features import FEATURE_NAMES


def _placement(board, lines=0, x=0):
    return Placement(rotation=0, x=x, landing_y=0,
                     afterstate=np.asarray(board, dtype=np.int8), lines=lines)


def _empty(rows=4, cols=3):
    return np.zeros((rows, cols), dtype=np.int8)


def test_empty_returns_none():
    assert LinearAgent(np.zeros(len(FEATURE_NAMES))).select([]) is None


def test_bad_weight_length_raises():
    with pytest.raises(ValueError):
        LinearAgent(np.zeros(3))


def test_maximizes_lines_when_rewarded():
    # Peso solo en lines_cleared (índice 3): prefiere la jugada que limpia líneas.
    w = np.zeros(len(FEATURE_NAMES))
    w[3] = 1.0
    no_clear = _placement(_empty(), lines=0)
    cleared = _placement(_empty(), lines=2)
    assert LinearAgent(w).select([no_clear, cleared]) is cleared


def test_prefers_lower_board_when_height_penalized():
    # Peso negativo en aggregate_height (índice 0): prefiere el tablero más bajo.
    w = np.zeros(len(FEATURE_NAMES))
    w[0] = -1.0
    low = _placement([[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0],
                      [1, 0, 0]])          # aggregate_height = 1
    high = _placement([[1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0],
                       [1, 0, 0]])         # aggregate_height = 4
    assert LinearAgent(w).select([high, low]) is low


def test_deterministic():
    w = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])
    ps = [_placement(_empty(), x=i, lines=i % 2) for i in range(4)]
    agent = LinearAgent(w)
    assert agent.select(ps) is agent.select(ps)
