"""Tests del pipeline de generación."""

import csv

from tetris_rl.agents import RandomAgent
from tetris_rl.data.generate import generate
from tetris_rl.data.logger import FIELDS


def test_generate_writes_valid_csv(tmp_path):
    path = generate(RandomAgent(seed=0), episodes=3, seed=0, out_dir=tmp_path,
                    rows=6, cols=4)
    assert path.exists()
    with path.open() as f:
        rows = list(csv.DictReader(f))
        f.seek(0)
        header = next(csv.reader(f))

    assert header == FIELDS
    assert len(rows) >= 1
    episodes = {int(r["episode"]) for r in rows}
    assert episodes <= {0, 1, 2}  # dentro del rango pedido


def test_generate_is_reproducible(tmp_path):
    a = generate(RandomAgent(seed=7), episodes=3, seed=7, out_dir=tmp_path,
                 run_name="a", rows=6, cols=4)
    b = generate(RandomAgent(seed=7), episodes=3, seed=7, out_dir=tmp_path,
                 run_name="b", rows=6, cols=4)
    assert a.read_text() == b.read_text()


def test_different_seed_changes_output(tmp_path):
    a = generate(RandomAgent(seed=1), episodes=3, seed=1, out_dir=tmp_path,
                 run_name="s1", rows=6, cols=4)
    b = generate(RandomAgent(seed=2), episodes=3, seed=2, out_dir=tmp_path,
                 run_name="s2", rows=6, cols=4)
    assert a.read_text() != b.read_text()
