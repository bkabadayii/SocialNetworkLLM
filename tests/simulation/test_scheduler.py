"""
Tests for all PairingStrategy implementations.
"""

from __future__ import annotations
import random

import pytest

from snllm.agents.core import Agent
from snllm.agents.models import AgentPersona
from snllm.environment.environment import NetworkEnvironment
from snllm.simulation.pairing import (
    RandomPairing,
    WeightedRandomPairing,
    AgentFriendshipPairing,
)
from snllm.config.network import EmptyNetworkConfig


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #
def make_agents(n: int) -> dict[str, Agent]:
    ags = []
    for i in range(n):
        p = AgentPersona(
            id=f"a{i}",
            name=f"Agent-{i}",
            age=30 + i,
            occupation="tester",
            political_view="none",
            interests=["pytest"],
        )
        ags.append(Agent(persona=p))
    return {a.id: a for a in ags}


def seeded_env(agents: dict[str, Agent]) -> NetworkEnvironment:
    """Environment with zero edges for deterministic tests."""
    env = NetworkEnvironment()
    env.initialize(agents.keys(), EmptyNetworkConfig())
    return env


# ------------------------------------------------------------------ #
@pytest.mark.parametrize("n_agents", [4, 5])
def test_random_pairing_no_overlap(n_agents):
    ags = make_agents(n_agents)
    env = seeded_env(ags)

    strategy = RandomPairing()
    pairs = strategy.select_pairs(ags, env, step=0)

    # pairs are disjoint
    flat = [x for p in pairs for x in p]
    assert len(flat) == len(set(flat))

    # size correct
    assert len(pairs) == len(ags) // 2


def test_weighted_pairing_bias():
    ags = make_agents(3)
    env = seeded_env(ags)

    # manually insert strong edge a0-a1
    env.update_edge("a0", "a1", 1.0)

    strategy = WeightedRandomPairing(epsilon=0.0)
    random.seed(42)  # deterministic choice

    pairs = strategy.select_pairs(ags, env, step=0)
    assert ("a0", "a1") in pairs or ("a1", "a0") in pairs


def test_agent_friendship_pairing_exploration():
    ags = make_agents(4)
    env = seeded_env(ags)

    strat = AgentFriendshipPairing(epsilon=1.0)  # pure exploration
    pairs = strat.select_pairs(ags, env, step=0)

    # everyone paired because exploration ignores strength
    assert len(pairs) == len(ags) // 2
