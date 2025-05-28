"""
snllm.environment.environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Friendship graph layer (undirected, weight = friendship strength).

* Only deals with network data
* Initial network comes from a `NetworkEnvironmentConfig`.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import networkx as nx

from snllm.config.network import (
    NetworkEnvironmentConfig,
    EmptyNetworkConfig,
    FullNetworkConfig,
    ErdosRenyiConfig,
    CustomNetworkConfig,
)


class NetworkEnvironment:
    """Thin façade over NetworkX keeping strengths in [0,1]."""

    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        self._g: nx.Graph = nx.Graph()

    # ------------------------------------------------------------------ #
    # Initialization (called exactly once by SimulationRunner)
    # ------------------------------------------------------------------ #
    def initialize(
        self,
        agent_ids: Iterable[str],
        cfg: NetworkEnvironmentConfig,
    ) -> None:
        """
        Populate the graph from *cfg*.

        The method is idempotent and clears any pre-existing edges.
        """
        self._g.clear()
        self._g.add_nodes_from(agent_ids)

        ids = list(agent_ids)

        # ------------- flavour dispatch --------------------------------
        if isinstance(cfg, EmptyNetworkConfig):  # no edges
            return

        if isinstance(cfg, FullNetworkConfig):  # complete graph
            for i, u in enumerate(ids):
                for v in ids[i + 1 :]:
                    self.update_edge(u, v, cfg.default_strength)
            return

        if isinstance(cfg, ErdosRenyiConfig):  # G(n,p)
            rng = random.Random(cfg.seed)
            for i, u in enumerate(ids):
                for v in ids[i + 1 :]:
                    if rng.random() < cfg.p:
                        self.update_edge(u, v, cfg.default_strength)
            return

        if isinstance(cfg, CustomNetworkConfig):
            for (u, v), s in cfg.edges.items():
                # order-independent keys
                a, b = sorted((u, v))
                self.update_edge(a, b, s)
            return

        # ----------------------------------------------------------------
        raise ValueError(f"Unsupported config type: {type(cfg)!r}")

    # ------------------------------------------------------------------ #
    # CRUD helpers (runner uses these during simulation)
    # ------------------------------------------------------------------ #
    def update_edge(self, u: str, v: str, strength: float) -> None:
        self._g.add_edge(u, v, weight=max(0.0, min(1.0, strength)))

    def strength(self, u: str, v: str) -> float:
        return self._g[u][v]["strength"] if self._g.has_edge(u, v) else 0.0

    def edges(self):
        """Yield `(u, v, strength)` tuples."""
        for u, v, data in self._g.edges(data=True):
            yield u, v, data["strength"]

    # ------------------------------------------------------------------ #
    # Export helpers
    # ------------------------------------------------------------------ #
    def snapshot(self, path: Path) -> None:
        # ensure the parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        nx.write_weighted_edgelist(self._g, path)
        nx.write_weighted_edgelist(self._g, path)

    def as_networkx(self) -> nx.Graph:  # read-only handle
        return self._g
