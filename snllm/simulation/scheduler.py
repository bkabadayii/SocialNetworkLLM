"""
snllm.simulation.scheduler
~~~~~~~~~~~~~~~~~~~~~~~~~~
Light wrapper: given agents + env + strategy → list of pairs.
"""

from __future__ import annotations
from typing import Dict, Tuple, List

from snllm.agents.core import Agent
from snllm.environment.environment import NetworkEnvironment
from snllm.simulation.pairing import PairingStrategy, AgentFriendshipPairing


class Scheduler:
    """
    Holds a PairingStrategy; can be swapped at runtime.
    """

    def __init__(self, strategy: PairingStrategy | None = None):
        self.strategy = strategy or AgentFriendshipPairing()

    def set_strategy(self, strategy: PairingStrategy) -> None:
        """
        Change the pairing strategy used by this scheduler.
        """
        self.strategy = strategy

    def pairs(
        self,
        agents: Dict[str, Agent],
        env: NetworkEnvironment,
        *,
        step: int,
    ) -> List[Tuple[str, str]]:
        return self.strategy.select_pairs(agents, env, step=step)
