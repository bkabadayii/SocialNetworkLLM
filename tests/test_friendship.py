"""
Unit-tests for snllm.agents.friendship.FriendshipManager

Coverage
--------
✓ strength() returns 0 for unknown ids
✓ adjust() clamps to [0,1] and updates last_interaction_step
✓ touch() refreshes timestamp without strength change
✓ decay() multiplies by exp(-μ·Δ) and drops tiny ties
✓ sample_partner():
     • exploitation path (random() > ε)
     • exploration path (random() ≤ ε)
"""

from __future__ import annotations

from random import Random

import pytest

from snllm.agents.friendship import FriendshipManager
from snllm.agents.models import AgentPersona, AgentState, Friendship
from snllm.config.config import FRIENDSHIP_HALF_LIFE_STEPS


# ------------------------------------------------------------------ #
# deterministic RNG helper
# ------------------------------------------------------------------ #
class DummyRand(Random):
    """Predictable RNG: .random() returns constant, .choices() returns first."""

    def __init__(self, const: float = 0.5):
        super().__init__()
        self.const = const

    def random(self):  # type: ignore[override]
        return self.const

    def choices(self, population, weights=None, k=1):  # type: ignore[override]
        return [population[0]]


# ------------------------------------------------------------------ #
@pytest.fixture()
def fm() -> FriendshipManager:
    """Fresh agent with one friend 'B' @ strength 0.8."""
    persona = AgentPersona(
        id="A",
        name="Alice",
        age=30,
        occupation="Dev",
        political_view="center",
        interests=["jazz"],
    )
    state = AgentState(persona=persona)
    state.friends["B"] = Friendship(strength=0.8, last_interaction_step=0)
    return FriendshipManager(state)


# ------------------------------------------------------------------ #
def test_strength_and_adjust(fm: FriendshipManager):
    assert fm.strength("Z") == 0.0  # unknown

    fm.adjust("B", delta=0.3, current_step=5)
    assert 0.0 <= fm.strength("B") <= 1.0
    assert fm.state.friends["B"].last_interaction_step == 5

    # clamp overflows
    fm.adjust("B", delta=10, current_step=6)
    assert fm.strength("B") == 1.0


def test_touch_updates_timestamp(fm: FriendshipManager):
    fm.touch("B", current_step=4)
    assert fm.state.friends["B"].last_interaction_step == 4


def test_decay_half_life(fm: FriendshipManager):
    # strength should halve after FRIENDSHIP_HALF_LIFE_STEPS with no touch
    initial = fm.strength("B")
    fm.decay(current_step=FRIENDSHIP_HALF_LIFE_STEPS)
    halved = fm.strength("B")
    assert pytest.approx(halved, rel=1e-2) == initial * 0.5

    # push many steps → pruned
    fm.decay(current_step=FRIENDSHIP_HALF_LIFE_STEPS * 10)
    assert "B" not in fm.neighbors()


def test_sample_partner_exploit_explore(monkeypatch, fm: FriendshipManager):
    universe = ["A", "B", "C"]  # C is non-neighbour
    # ----- exploitation path (random() > ε) -----
    rng = DummyRand(const=0.9)  # > ε default 0.1
    monkeypatch.setattr("snllm.agents.friendship.random", rng.random)
    monkeypatch.setattr("snllm.agents.friendship.choices", rng.choices)

    partner = fm.sample_partner(universe, current_step=1, epsilon=0.1)
    assert partner == "B"  # picks existing neighbour

    # ----- exploration path (random() <= ε) -----
    rng.const = 0.05  # <= ε; forces explore
    partner = fm.sample_partner(universe, current_step=2, epsilon=0.1)
    assert partner == "C"  # picks non-neighbour
