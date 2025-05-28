# tests/environment/test_environment.py

"""
Unit-tests for snllm.environment.environment.NetworkEnvironment
covering every NetworkEnvironmentConfig flavour.
"""

from __future__ import annotations

import networkx as nx
import pytest

from snllm.environment.environment import NetworkEnvironment
from snllm.config.network import (
    EmptyNetworkConfig,
    FullNetworkConfig,
    ErdosRenyiConfig,
    CustomNetworkConfig,
)

AGENTS = [f"a{i}" for i in range(5)]  # five dummy IDs


def test_empty_config():
    env = NetworkEnvironment()
    env.initialize(AGENTS, EmptyNetworkConfig())
    g = env.as_networkx()
    assert g.number_of_nodes() == len(AGENTS)
    assert g.number_of_edges() == 0


def test_full_config():
    """FullNetworkConfig should produce a complete graph with uniform strength."""
    cfg = FullNetworkConfig(default_strength=0.42)
    env = NetworkEnvironment()
    env.initialize(AGENTS, cfg)

    g: nx.Graph = env.as_networkx()
    n = len(AGENTS)
    # complete graph ⇒ n·(n-1)/2 edges
    assert g.number_of_edges() == n * (n - 1) // 2

    # every edge has identical weight = 0.42
    weights = [data["strength"] for _, _, data in g.edges(data=True)]
    assert all(w == pytest.approx(0.42) for w in weights)


def test_er_config_seeded():
    """ErdosRenyiConfig(p=1) collapses to full graph; p=0 gives empty."""
    # p=1.0 → full graph
    cfg_full = ErdosRenyiConfig(p=1.0, default_strength=0.7, seed=123)
    env = NetworkEnvironment()
    env.initialize(AGENTS, cfg_full)
    g = env.as_networkx()
    n = len(AGENTS)
    assert g.number_of_edges() == n * (n - 1) // 2
    weights = [data["strength"] for _, _, data in g.edges(data=True)]
    assert all(w == pytest.approx(0.7) for w in weights)

    # p=0.0 → no edges
    cfg_empty = ErdosRenyiConfig(p=0.0, default_strength=0.5, seed=123)
    env.initialize(AGENTS, cfg_empty)
    assert env.as_networkx().number_of_edges() == 0


def test_custom_config_round_trip():
    """CustomNetworkConfig should reproduce exactly the user-specified edges."""
    # Provide an order-insensitive dict
    pairs = {
        (AGENTS[0], AGENTS[1]): 0.9,
        (AGENTS[3], AGENTS[2]): 0.4,
    }
    cfg = CustomNetworkConfig(edges=pairs)

    env = NetworkEnvironment()
    env.initialize(AGENTS, cfg)
    g = env.as_networkx()

    # Each specified pair should appear (in either order) with correct strength
    for (u, v), expected in pairs.items():
        assert expected == pytest.approx(env.strength(u, v))
        assert expected == pytest.approx(env.strength(v, u))

    # No extra edges
    assert g.number_of_edges() == len(pairs)
