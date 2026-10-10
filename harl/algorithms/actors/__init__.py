"""Actors used by the reproduction experiments."""
from harl.algorithms.actors.happo import HAPPO
from harl.algorithms.actors.hasac import HASAC

ALGO_REGISTRY = {"happo": HAPPO, "hasac": HASAC}
