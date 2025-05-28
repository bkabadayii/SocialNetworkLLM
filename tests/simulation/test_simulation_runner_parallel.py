"""
Live end-to-end test for the *parallel* SimulationRunner.

⚠️  Requires real Google Gemini calls — will incur token costs.
    Runs only when BOTH
        SNLLM_RUN_GEMINI_TESTS=true
        GOOGLE_API_KEY          are present in the environment.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from snllm.agents.core import Agent
from snllm.agents.models import AgentPersona
from snllm.simulation.runner import SimulationRunner
from snllm.simulation.pairing import RandomPairing
from snllm.simulation.scheduler import Scheduler
from snllm.config.network import EmptyNetworkConfig

# --------------------------------------------------------------------- #
RUN = os.getenv("SNLLM_RUN_GEMINI_TESTS", "").lower() == "true" and os.getenv(
    "GOOGLE_API_KEY"
)

LOG_DIR = Path("tests/test_logs_parallel")
LOG_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------- #
@pytest.mark.skipif(not RUN, reason="Gemini live test disabled")
def test_parallel_simulation_live() -> None:
    # ---------- build 10 realistic personas ------------------------- #
    personas = [
        AgentPersona(
            id="alice",
            name="Alice Martin",
            age=28,
            occupation="Front-end Dev",
            political_view="Centrist",
            interests=["indie games", "bouldering", "street food"],
        ),
        AgentPersona(
            id="bob",
            name="Bob Lee",
            age=35,
            occupation="Product Manager",
            political_view="Liberal",
            interests=["travelling", "board games", "coffee brewing"],
        ),
        AgentPersona(
            id="clara",
            name="Clara Meier",
            age=31,
            occupation="Architect",
            political_view="Green",
            interests=["urban sketching", "salsa dancing"],
        ),
        AgentPersona(
            id="dan",
            name="Dan O’Donnell",
            age=42,
            occupation="High-school Teacher",
            political_view="Social-democrat",
            interests=["history podcasts", "cycling"],
        ),
        AgentPersona(
            id="eva",
            name="Eva Rossi",
            age=27,
            occupation="Biomedical Engineer",
            political_view="Progressive",
            interests=["yoga", "vegan cooking", "sci-fi films"],
        ),
        AgentPersona(
            id="felix",
            name="Felix Zhang",
            age=30,
            occupation="Data Analyst",
            political_view="Libertarian",
            interests=["basketball", "electronic music"],
        ),
        AgentPersona(
            id="gia",
            name="Gianna Conti",
            age=24,
            occupation="Graphic Designer",
            political_view="Green",
            interests=["illustration", "fashion thrifting"],
        ),
        AgentPersona(
            id="hugo",
            name="Hugo Sørensen",
            age=38,
            occupation="Marine Biologist",
            political_view="Environmental-left",
            interests=["diving", "photography", "sailing"],
        ),
        AgentPersona(
            id="iris",
            name="Iris Khan",
            age=33,
            occupation="Clinical Psychologist",
            political_view="Liberal",
            interests=["mindfulness", "poetry slams"],
        ),
        AgentPersona(
            id="jack",
            name="Jack Wilson",
            age=45,
            occupation="Chef",
            political_view="Moderate",
            interests=["fermentation", "trail running"],
        ),
    ]
    agents = [Agent(p) for p in personas]

    # deterministic partner shuffle for reproducibility
    random.seed(123)
    scheduler = Scheduler(strategy=RandomPairing())

    max_steps = 50

    runner = SimulationRunner(
        agents=agents,
        graph_cfg=EmptyNetworkConfig(),  # start with zero edges
        scheduler=scheduler,
        max_steps=max_steps,
        log_dir=LOG_DIR,
    )

    # ---------- PARALLEL run ---------------------------------------- #
    runner.run_parallel()  # default thread pool

    # ---------- assertions ------------------------------------------ #
    step_logs = runner.step_logs()
    assert len(step_logs) == max_steps, f"Should have {max_steps} StepLog entries"

    # every snapshot exists
    for slog in step_logs:
        assert slog.graph_path and slog.graph_path.exists()

    # at least one conversation with spoken turns
    conversations = [c for sl in step_logs for c in sl.conversations]
    assert any(c.turns for c in conversations), "No dialogue turns logged!"

    # interest traces stay in [0,1]
    for c in conversations:
        assert 0.0 <= min(c.interest) <= max(c.interest) <= 1.0

    # final graph has >= 1 edge with non-zero strength (friendship update happened)
    final_g = runner.graph()
    assert any(
        data["weight"] > 0 for _, _, data in final_g.edges(data=True)
    ), "No friendship edges created after 20 steps"
