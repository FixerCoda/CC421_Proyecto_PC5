"""Tests del entrenador CEM (tablero chico y params mínimos para que sea rápido)."""

import numpy as np

from tetris_rl.agents import LinearAgent
from tetris_rl.agents.cem import evaluate, play_episode, train_cem
from tetris_rl.env.tetris_env import TetrisEnv
from tetris_rl.features import FEATURE_NAMES

SMALL = dict(generations=2, population=6, episodes_per_eval=2, rows=6, cols=4)


def test_returns_valid_weights():
    weights, history = train_cem(seed=0, **SMALL)
    assert weights.shape == (len(FEATURE_NAMES),)
    assert np.all(np.isfinite(weights))
    assert len(history) == SMALL["generations"]


def test_reproducible():
    w1, _ = train_cem(seed=0, **SMALL)
    w2, _ = train_cem(seed=0, **SMALL)
    np.testing.assert_array_equal(w1, w2)


def test_different_seed_differs():
    w1, _ = train_cem(seed=0, **SMALL)
    w2, _ = train_cem(seed=1, **SMALL)
    assert not np.array_equal(w1, w2)


def test_cem_weights_can_play():
    # Smoke: los pesos entrenados producen un agente que juega sin romper.
    weights, _ = train_cem(seed=0, **SMALL)
    env = TetrisEnv(rows=6, cols=4)
    lines = play_episode(env, LinearAgent(weights), seed=0)
    assert lines >= 0


def test_evaluate_returns_float():
    env = TetrisEnv(rows=6, cols=4)
    fit = evaluate(np.zeros(len(FEATURE_NAMES)), env, seeds=[0, 1])
    assert isinstance(fit, float)
    assert fit >= 0


def test_max_steps_zero_clears_nothing():
    # Con tope 0 no se coloca ninguna pieza -> 0 líneas (verifica que el tope corta).
    env = TetrisEnv(rows=6, cols=4)
    lines = play_episode(env, LinearAgent(np.zeros(len(FEATURE_NAMES))), seed=0, max_steps=0)
    assert lines == 0
