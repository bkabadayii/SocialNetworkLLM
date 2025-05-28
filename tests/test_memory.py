"""
Integration test that exercises MemoryManager against a *real* Pinecone
index.  Requires PINECONE_API_KEY in the environment.

Steps
-----
1.  Creates a fresh random namespace (agent_id = UUID).
2.  Adds one memory and verifies it exists remotely.
3.  Retrieves it, checking rehearsal boosts importance.
4.  Advances simulation steps until decay prunes the vector.
5.  Cleans up the namespace.

If anything fails (network, credentials) the test is skipped.
"""

from __future__ import annotations

import os
import time
import uuid
import math

import pytest
from pinecone import Pinecone

from snllm.agents.memory import MemoryManager

WAIT_TIME = 5  # seconds to wait for Pinecone to update


@pytest.mark.skipif(
    "PINECONE_API_KEY" not in os.environ,
    reason="PINECONE_API_KEY env var not set; skipping online Pinecone test.",
)
def test_memory_manager_live_roundtrip():
    # ------------------------------------------------------------------ #
    # 0.  Set-up Pinecone client + temp namespace
    # ------------------------------------------------------------------ #
    pc = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
        environment="gcp-starter",
    )
    index_name = "snllm"
    index = pc.Index(index_name)

    agent_ns = f"test-{uuid.uuid4().hex[:8]}"
    mm = MemoryManager(agent_id=agent_ns)

    # ------------------------------------------------------------------ #
    # 1.  Add memory
    # ------------------------------------------------------------------ #
    mm.add("Integration test fact", importance=0.6)
    time.sleep(WAIT_TIME)  # tiny wait to ensure eventual consistency
    stats = index.describe_index_stats()
    assert stats["namespaces"].get(agent_ns, {}).get("vector_count", 0) == 1

    # ------------------------------------------------------------------ #
    # 2.  Retrieve & check rehearsal boost
    # ------------------------------------------------------------------ #
    retrieved = mm.retrieve("fact", k=1, current_step=1)
    assert len(retrieved) == 1
    boosted_imp = retrieved[0].item.importance
    assert math.isclose(boosted_imp, 0.7, abs_tol=1e-3) or boosted_imp > 0.6

    # ------------------------------------------------------------------ #
    # 3.  Decay until pruned
    # ------------------------------------------------------------------ #
    #  current_step far in the future -> importance should fall < threshold
    mm.decay_and_prune(current_step=99)
    time.sleep(WAIT_TIME)
    stats_after = index.describe_index_stats()
    assert stats_after["namespaces"].get(agent_ns, {}).get("vector_count", 0) == 0

    # ------------------------------------------------------------------ #
    # 4.  Clean up namespace (defensive)
    # ------------------------------------------------------------------ #
    index.delete(delete_all=True, namespace=agent_ns)
