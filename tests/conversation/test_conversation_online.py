# tests/test_conversation_live.py

"""
Live integration test for ConversationEngine using real Gemini API.

❗ WARNING ❗
These tests make actual LLM calls and will consume quota. They run only if:
  - SNLLM_RUN_GEMINI_TESTS=true
  - GOOGLE_API_KEY

Example:
    export SNLLM_RUN_GEMINI_TESTS=true
    export GOOGLE_API_KEY="your_key"
    pytest tests/test_conversation_live.py -q
"""

import os
import pytest
from pathlib import Path

from snllm.agents.models import AgentPersona
from snllm.agents.core import Agent
from snllm.conversation.conversation import ConversationEngine
from snllm.conversation.interest import InterestModel

# Determine whether to run live tests
RUN_LIVE = os.getenv("SNLLM_RUN_GEMINI_TESTS", "").lower() == "true" and (
    os.getenv("GOOGLE_API_KEY")
)

# ensure logs directory
LOG_DIR = Path("tests/test_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)


@pytest.mark.skipif(not RUN_LIVE, reason="Live Gemini conversation test disabled")
def test_conversation_live_short():
    """
    Run a short two-turn conversation between two real Agents.
    Verify that:
      - At least one utterance occurs.
      - transcript() and turn_count() agree.
      - interest_trace() values stay within [0,1].
      - friendship strength is updated to a valid float in [0,1].
    """
    # --- create two simple personas ---
    alice_p = AgentPersona(
        id="mark_live",
        name="Mark Jones",
        age=56,
        occupation="Author",
        political_view="Neutral",
        interests=["writing", "history", "philosophy"],
    )
    bob_p = AgentPersona(
        id="bob_live",
        name="BobLive",
        age=31,
        occupation="Artist",
        political_view="Libertarian",
        interests=["music", "travel"],
    )

    # --- instantiate agents (no initial memories) ---
    alice = Agent(persona=alice_p)
    bob = Agent(persona=bob_p)

    # --- set up ConversationEngine with minimal constraints ---
    # Use default InterestModel so interest may drop naturally
    engine = ConversationEngine(
        agent_a=alice,
        agent_b=bob,
        interest_model=InterestModel(),
        max_turns=10,  # cap at 10 turns
        interest_threshold=0.0,  # allow up to max_turns
    )

    # --- run a single dialogue at step=1 ---
    engine.run(step=1)

    transcript = engine.transcript()
    turns = engine.turn_count()
    interest = engine.interest_trace()
    # friendship strengths after average delta
    f_ab = alice.friends.strength(bob.id)
    f_ba = bob.friends.strength(alice.id)

    # --- assertions ---
    assert turns > 0, "Expected at least one utterance"
    assert len(transcript) == turns
    assert all(isinstance(s, str) and isinstance(u, str) for s, u in transcript)
    assert all(
        0.0 <= val <= 1.0 for val in interest
    ), "Interest values must be in [0,1]"
    assert isinstance(f_ab, float) and 0.0 <= f_ab <= 1.0
    assert isinstance(f_ba, float) and 0.0 <= f_ba <= 1.0

    # --- write logs out ---
    full_log = engine.log()
    with open(LOG_DIR / "conversation_log.json", "w") as f:
        f.write(full_log.model_dump_json(indent=2))
