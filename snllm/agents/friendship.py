"""
snllm.agents.friendship
=======================

Stateful wrapper around `AgentState.friends` that:

1. Stores tie strength ∈ [0, 1] and the last_interaction_step (int).
2. Applies exponential decay when agents ignore each other.
3. Lets the LLM alter friendships via `adjust()` (delta ∈ [-1, 1]).
4. Provides `sample_partner()` for the Planner:
      •  weighted roulette-wheel among neighbours (1-ε)
      •  ε chance to explore a random non-neighbour  (“weak-tie”)

All time units are simulation steps — see `snllm.config`.
"""

from __future__ import annotations

import math
from random import choices, random
from typing import Dict, List, Optional
import warnings

from snllm.agents.models import AgentState, Friendship
from snllm.config.config import FRIENDSHIP_HALF_LIFE_STEPS

# Pre-compute μ so that  strength → strength/2  after HALF_LIFE steps
_DECAY_MU = math.log(2) / FRIENDSHIP_HALF_LIFE_STEPS


class FriendshipManager:
    """Operations on a single agent’s friendship dict (step-based)."""

    def __init__(self, state: AgentState):
        self.state = state

    # ─────────────────────── read helpers ────────────────────────── #
    def strength(self, target_id: str) -> float:
        """Return current tie strength to target_id (0.0 if unknown)."""
        return self.state.friends.get(target_id, Friendship(strength=0.0)).strength

    def neighbors(self) -> Dict[str, Friendship]:
        """Direct access to the underlying mapping."""
        return self.state.friends

    # ─────────────────────── write helpers ───────────────────────── #
    def set_strength(
        self, target_id: str, strength: float, current_step: int = 0
    ) -> None:
        """
        Set friendship strength to strength (0.0 ≤ strength ≤ 1.0).
        """
        effective_strength = max(0.0, min(1.0, strength))
        if not (0.0 <= strength <= 1.0):
            warnings.warn(
                f"Invalid friendship strength {strength} for {target_id}. "
                "Must be in [0.0, 1.0]. Setting to "
                f"{effective_strength} instead.",
                UserWarning,
            )

        edge = self.state.friends.get(target_id, Friendship(strength=0.0))
        edge.strength = effective_strength
        self.state.friends[target_id] = edge

    def adjust(self, target_id: str, delta: float, current_step: int) -> None:
        """
        Increase (or decrease) the friendship with target_id by delta.
        Called by Gemini tool-function `adjust_friendship`.
        """
        edge = self.state.friends.get(target_id, Friendship(strength=0.0))
        edge.strength = max(0.0, min(1.0, edge.strength + delta))
        self.state.friends[target_id] = edge

    def touch(self, target_id: str, current_step: int) -> None:
        """
        Update last_interaction_step without changing strength.
        Call after each successful conversation turn.
        """
        if target_id in self.state.friends:
            self.state.friends[target_id].last_interaction_step = current_step

    # ────────────────── passive decay (run each timestep) ─────────── #
    def decay(self, current_step: int) -> None:
        """
        Fade friendships that haven't been contacted.
        Strength halves every `FRIENDSHIP_HALF_LIFE_STEPS`.
        Remove edge when strength < 0.01.
        """
        for fid, edge in list(self.state.friends.items()):
            delta = current_step - edge.last_interaction_step
            edge.strength *= math.exp(-_DECAY_MU * delta)
            edge.last_interaction_step = current_step
            if edge.strength < 0.01:
                self.state.friends.pop(fid)

    # ──────────────────── partner sampling for Planner ──────────── #
    def sample_partner(
        self,
        all_agent_ids: List[str],
        current_step: int,
        epsilon: float = 0.1,
    ) -> Optional[str]:
        """
        Select a partner ID.

        Parameters
        ----------
        all_agent_ids : list[str]
            Universe of agents (including self).
        current_step : int
            Needed only if we later add time-aware heuristics here.
        epsilon : float
            Exploration rate: chance to pick a random non-neighbour.

        Returns
        -------
        str | None
            Partner ID or None if no candidates (isolated agent).
        """
        # Exploit existing neighbours  (prob = 1-ε)
        if self.state.friends and random() > epsilon:
            ids, weights = zip(
                *[(fid, edge.strength) for fid, edge in self.state.friends.items()]
            )
            return choices(ids, weights=weights, k=1)[0]

        # Explore a non-neighbour
        non_neigh = [
            aid
            for aid in all_agent_ids
            if aid not in self.state.friends and aid != self.state.persona.id
        ]
        return choices(non_neigh, k=1)[0] if non_neigh else None
