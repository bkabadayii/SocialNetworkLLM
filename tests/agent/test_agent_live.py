"""
Live two-round dialogue test for Agent.react().

Run only if:
  SNLLM_RUN_GEMINI_TESTS=true
  GOOGLE_API_KEY is set
"""

import os
import pytest
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List

from snllm.agents.core import Agent
from snllm.agents.models import AgentPersona

RUN = os.getenv("SNLLM_RUN_GEMINI_TESTS", "").lower() == "true" and os.getenv(
    "GOOGLE_API_KEY"
)


@pytest.mark.skipif(not RUN, reason="Gemini live test disabled")
def test_two_round_conversation_live():
    # --- personas ----------------------------------------------------
    alice_p = AgentPersona(
        id="alice",
        name="Alice",
        age=30,
        occupation="Dev",
        political_view="Centrist",
        interests=["chess", "jazz"],
    )
    bob_p = AgentPersona(
        id="bob",
        name="Bob",
        age=32,
        occupation="Designer",
        political_view="Green",
        interests=["hiking"],
    )

    # --- agents ------------------------------------------------------
    alice = Agent(persona=alice_p)
    bob = Agent(persona=bob_p)

    # --- round 1 -----------------------------------------------------
    bob_initial = "Hi Alice! I'm Bob."

    history_a: List[BaseMessage] = [HumanMessage(content=bob_initial)]
    history_b: List[BaseMessage] = [AIMessage(content=bob_initial)]

    a_reply = alice.react(partner_persona=bob_p, history=history_a, current_step=1)

    # Accept either spoken text or a tool-only step
    if a_reply:
        # feed Alice's reply to Bob
        history_b.append(HumanMessage(content=a_reply.messages[-1].content))
    else:
        history_b.append(HumanMessage(content="(Alice performed a tool.)"))

    # --- round 2 -----------------------------------------------------
    b_reply = bob.react(partner_persona=alice_p, history=history_b, current_step=1)

    # Basic sanity: at least one of the two replies is text
    assert a_reply or b_reply
