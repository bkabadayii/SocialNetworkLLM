"""
snllm.simulation.pairing
~~~~~~~~~~~~~~~~~~~~~~~~
Pluggable partner-selection strategies.
"""

from __future__ import annotations
import random
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import warnings

from snllm.agents.core import Agent
from snllm.environment.environment import NetworkEnvironment

from snllm.config.config import FRIENDSHIP_EXPLORATION_EPSILON


class PairingStrategy(ABC):
    """Return disjoint pairs (u,v) for the current step."""

    @abstractmethod
    def select_pairs(
        self,
        agents: Dict[str, Agent],
        env: NetworkEnvironment,
        *,
        step: int,
    ) -> List[Tuple[str, str]]: ...


# --------------------------------------------------------------------- #
class RandomPairing(PairingStrategy):
    """Uniform random matching without replacement."""

    def select_pairs(
        self,
        agents: Dict[str, Agent],
        env: NetworkEnvironment,
        *,
        step: int,
    ) -> List[Tuple[str, str]]:
        ids = list(agents.keys())
        random.shuffle(ids)
        # drop last if odd
        if len(ids) % 2:
            ids.pop()
        return [(ids[i], ids[i + 1]) for i in range(0, len(ids), 2)]


# --------------------------------------------------------------------- #
class WeightedRandomPairing(PairingStrategy):
    """
    Default strategy:  P(u <-> v) ~ friendship_strength(u,v) + ε
    (ε small so isolated nodes still get paired).
    """

    def __init__(self, epsilon: float = 0.05):
        self.eps = epsilon

    def select_pairs(
        self,
        agents: Dict[str, Agent],
        env: NetworkEnvironment,
        *,
        step: int,
    ) -> List[Tuple[str, str]]:
        ids = list(agents.keys())
        random.shuffle(ids)  # random tie-breaking
        pairs: List[Tuple[str, str]] = []
        used: set[str] = set()

        for u in ids:
            if u in used:
                continue
            # build candidate list
            weights, candidates = [], []
            for v in ids:
                if v == u or v in used:
                    continue
                w = env.strength(u, v) + self.eps
                candidates.append(v)
                weights.append(w)
            if not candidates:
                continue
            v = random.choices(candidates, weights=weights, k=1)[0]
            pairs.append((u, v))
            used.update((u, v))
        return pairs


class AgentFriendshipPairing(PairingStrategy):
    """
    Pair agents based on their SNAgent state.
    This is a placeholder for future strategies that might
    consider more complex agent states or attributes.
    """

    def __init__(self, epsilon: float = FRIENDSHIP_EXPLORATION_EPSILON):
        """
        AgentFriendshipPairing
        Initialize with an exploration rate (epsilon).
        This controls the exploration vs exploitation trade-off.
        Parameters
        ----------
        epsilon: float
            Exploration rate for selecting partners. Must be in [0, 1].
            Higher values mean more exploration.
        """
        if not (0 <= epsilon <= 1):
            raise ValueError(
                "Friendship Exploration Epsilon must be in [0, 1]. Either set it properly in the config or pass a valid value constructor."
            )
        self.epsilon = epsilon

    def select_pairs(
        self,
        agents: Dict[str, Agent],
        env: NetworkEnvironment,
        *,
        step: int,
    ) -> List[Tuple[str, str]]:
        # For now, just use random pairing
        ids = list(agents.keys())
        pairs: List[Tuple[str, str]] = []
        used: set[str] = set()
        remaining = set(ids)

        for agent in agents.values():
            if agent.id in used:
                continue
            # get sample from agent.friends
            v = agent.friends.sample_partner(
                all_agent_ids=list(remaining),
                current_step=step,
                epsilon=self.epsilon,  # exploration rate
            )

            if not v:
                continue

            pairs.append((agent.id, v))  # (u, v) pair
            used.update((agent.id, v))  # mark both as used
            remaining.discard(v)  # remove v from remaining
            remaining.discard(agent.id)  # remove u from remaining

        # Raise warnings for unpaired agents
        if len(remaining) > 0:
            warnings.warn(
                f"Unpaired agents remaining: {', '.join(remaining)}.\n"
                "If this is unexpected behavior, Consider adjusting the pairing strategy or epsilon."
            )

        return pairs
