"""Agentes que juegan eligiendo una colocación de env.legal_placements()."""

from .baselines import DEFAULT_WEIGHTS, HeuristicAgent, RandomAgent
from .linear_agent import LinearAgent
from .td_agent import TDAgent

__all__ = ["RandomAgent", "HeuristicAgent", "DEFAULT_WEIGHTS", "TDAgent", "LinearAgent"]
