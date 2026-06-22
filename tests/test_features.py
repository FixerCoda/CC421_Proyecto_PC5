"""Tests de extract_features con tableros chicos hechos a mano.

Convención: int8, fila 0 arriba, 0 = vacío, 1 = ocupado.
Vector: [aggregate_height, holes, bumpiness, lines_cleared, max_height, wells].
"""

import numpy as np

from tetris_rl.features import FEATURE_NAMES, extract_features


def test_feature_names_and_shape():
    assert FEATURE_NAMES == (
        "aggregate_height",
        "holes",
        "bumpiness",
        "lines_cleared",
        "max_height",
        "wells",
    )
    out = extract_features(np.zeros((3, 3), dtype=np.int8), 0)
    assert out.shape == (6,)
    assert out.dtype == np.float64


def test_empty_board_passthrough_lines():
    # Tablero vacío: todo 0, salvo lines_cleared que se pasa tal cual.
    board = np.zeros((3, 3), dtype=np.int8)
    out = extract_features(board, lines_cleared=2)
    np.testing.assert_array_equal(out, [0, 0, 0, 2, 0, 0])


def test_full_single_column():
    # Columna 0 llena (h=4). heights=[4,0,0]: bumpiness=4, wells=0.
    board = np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
            [1, 0, 0],
        ],
        dtype=np.int8,
    )
    out = extract_features(board, lines_cleared=0)
    np.testing.assert_array_equal(out, [4, 0, 4, 0, 4, 0])


def test_covered_hole():
    # Columna 1: ocupada/vacía/ocupada -> 1 hueco tapado. heights=[0,3,0].
    board = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 1, 0],
        ],
        dtype=np.int8,
    )
    # wells: ambos bordes hundidos respecto a col 1 (3 cada uno) -> 6.
    out = extract_features(board, lines_cleared=0)
    np.testing.assert_array_equal(out, [3, 1, 6, 0, 3, 6])


def test_central_well():
    # Cols 0 y 2 llenas, col 1 vacía: pozo central de profundidad 3.
    board = np.array(
        [
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
        ],
        dtype=np.int8,
    )
    out = extract_features(board, lines_cleared=0)
    np.testing.assert_array_equal(out, [6, 0, 6, 0, 3, 3])


def test_staircase_bumpiness_and_edge_well():
    # Alturas 1-2-3: bumpiness=2 y un pozo de borde de profundidad 1 en col 0.
    board = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )
    out = extract_features(board, lines_cleared=0)
    np.testing.assert_array_equal(out, [6, 0, 2, 0, 3, 1])
