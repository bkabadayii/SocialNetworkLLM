# tests/simulation/test_simulation_runner_live.py
"""
Live end-to-end test for SimulationRunner.

⚠️  Requires real Google Gemini calls — will incur token costs.
    Runs only when:
        SNLLM_RUN_GEMINI_TESTS=true
        GOOGLE_API_KEY set
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import networkx as nx
import pytest

from snllm.agents.core import Agent
from snllm.agents.models import AgentPersona
from snllm.simulation.runner import SimulationRunner
from snllm.simulation.pairing import RandomPairing
from snllm.simulation.scheduler import Scheduler
from snllm.config.network import EmptyNetworkConfig

# ------------------------------------------------------------------ #
RUN = os.getenv("SNLLM_RUN_GEMINI_TESTS", "").lower() == "true" and os.getenv(
    "GOOGLE_API_KEY"
)


@pytest.mark.skipif(not RUN, reason="Gemini live test disabled")
def test_simulation_two_steps_live() -> None:
    # ---------- temp log dir ----------------------------------------
    log_dir = Path("tests/test_logs")

    # ---------- realistic personas ----------------------------------
    personas = [
        AgentPersona(
            id="alice",
            name="Alice Johnson",
            age=29,
            occupation="Software Engineer",
            political_view="Centrist",
            interests=["art museums", "rock-climbing", "sci-fi novels"],
        ),
        AgentPersona(
            id="bob",
            name="Robert Kim",
            age=34,
            occupation="UX Designer",
            political_view="Green",
            interests=["photography", "urban gardening"],
        ),
        AgentPersona(
            id="carla",
            name="Carla Diaz",
            age=27,
            occupation="Data Scientist",
            political_view="Liberal",
            interests=["jazz piano", "cycling", "machine learning"],
        ),
        AgentPersona(
            id="dan",
            name="Daniele Rossi",
            age=48,
            occupation="Civil Engineer",
            political_view="Social-democrat",
            interests=["football", "Italian cooking"],
        ),
    ]
    agents = [Agent(p) for p in personas]

    # deterministic partner shuffle for reproducibility
    random.seed(42)
    scheduler = Scheduler(strategy=RandomPairing())

    runner = SimulationRunner(
        agents=agents,
        graph_cfg=EmptyNetworkConfig(),  # start with no edges
        scheduler=scheduler,
        max_steps=2,  # keep token use low
        log_dir=log_dir,
    )
    runner.run()

    # ---------- assertions ------------------------------------------
    step_logs = runner.step_logs()
    assert len(step_logs) == 2

    # graph snapshots exist & load
    for sl in step_logs:
        assert sl.graph_path and sl.graph_path.exists()
        g = nx.read_edgelist(sl.graph_path)
        assert isinstance(g, nx.Graph)

    # at least one conversation with content
    convs = [c for sl in step_logs for c in sl.conversations]
    assert any(c.turns for c in convs)

    # every interest trace well-formed
    for c in convs:
        assert c.interest  # non-empty
        assert 0.0 <= min(c.interest) <= max(c.interest) <= 1.0
