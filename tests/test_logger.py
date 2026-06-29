"""Tests del logger de transiciones."""

import csv

import numpy as np

from tetris_rl.data.logger import FIELDS, TransitionLogger
from tetris_rl.env.tetris_env import Placement
from tetris_rl.features import FEATURE_NAMES, extract_features


def _placement(board, lines=0, rotation=0, x=0):
    return Placement(rotation=rotation, x=x, landing_y=0,
                     afterstate=np.asarray(board, dtype=np.int8), lines=lines)


def test_header_matches_fields(tmp_path):
    path = tmp_path / "run.csv"
    with TransitionLogger(path):
        pass
    with path.open() as f:
        header = next(csv.reader(f))
    assert header == FIELDS
    # lines_cleared aparece una sola vez (como feature, no duplicado).
    assert header.count("lines_cleared") == 1


def test_creates_parent_dir(tmp_path):
    path = tmp_path / "sub" / "dir" / "run.csv"
    with TransitionLogger(path):
        pass
    assert path.exists()


def test_logs_rows_with_features(tmp_path):
    board = [[0, 0, 0],
             [0, 0, 0],
             [0, 0, 0],
             [1, 0, 0]]
    p = _placement(board, lines=2, rotation=1, x=3)
    path = tmp_path / "run.csv"
    with TransitionLogger(path) as logger:
        logger.log(episode=0, step=5, piece="T", placement=p, reward=-1.5)

    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    row = rows[0]
    assert row["episode"] == "0"
    assert row["step"] == "5"
    assert row["piece"] == "T"
    assert row["rotation"] == "1"
    assert row["x"] == "3"
    assert float(row["reward"]) == -1.5

    # Las features del CSV coinciden con extract_features.
    phi = extract_features(np.asarray(board, dtype=np.int8), 2)
    for name, val in zip(FEATURE_NAMES, phi):
        assert float(row[name]) == val


def test_multiple_rows(tmp_path):
    p = _placement([[0, 0, 0], [1, 0, 0]])
    path = tmp_path / "run.csv"
    with TransitionLogger(path) as logger:
        for step in range(3):
            logger.log(episode=1, step=step, piece="I", placement=p, reward=0.0)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert [r["step"] for r in rows] == ["0", "1", "2"]
