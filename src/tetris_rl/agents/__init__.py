"""Agentes que juegan eligiendo una colocación de env.legal_placements()."""

from .baselines import DEFAULT_WEIGHTS, HeuristicAgent, RandomAgent

__all__ = ["RandomAgent", "HeuristicAgent", "DEFAULT_WEIGHTS"]
