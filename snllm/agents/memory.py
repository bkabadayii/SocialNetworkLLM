"""
snllm.agents.memory
~~~~~~~~~~~~~~~~~~~
Hybrid long-term memory:

*  Local store → rich Python `MemoryItem`s with all metadata
*  Pinecone v3 → sparse vector index id -> embedding

Retrieval:
    1. Query Pinecone - get ids + cosine scores
    2. Look-up ids locally - compute hybrid score
    3. Apply rehearsal bonus & return top-k `RetrievedMemory`

Decay & prune:
    * Exponential decay is computed over the local dict
    * Items falling below IMPORTANCE_THRESHOLD are deleted locally and
      removed from Pinecone in one batch call
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

from loguru import logger
from pydantic import BaseModel, ConfigDict, PrivateAttr
from pinecone import Pinecone, ServerlessSpec

from snllm.agents.models import MemoryItem
from snllm.config.config import (
    FORGET_TARGET,
    STEPS_PER_SIM_DAY,
    EMBEDDING_DIM,
    IMPORTANCE_THRESHOLD,
    REHEARSAL_BONUS,
)
from snllm.utils.embeddings import embed_text


# ────────────────────────── Pinecone bootstrap ───────────────────────
def _get_index():
    pc = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"],
        environment="gcp-starter",
    )
    if "snllm" not in [idx.name for idx in pc.list_indexes()]:
        pc.create_index(
            name="snllm",
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
    return pc.Index("snllm")


_INDEX = _get_index()


@dataclass
class RetrievedMemory:
    item: MemoryItem
    score: float


# ──────────────────────────── MemoryManager ──────────────────────────
class MemoryManager(BaseModel):
    """
    Agent-level wrapper that owns a local dict AND a Pinecone namespace.
    """

    agent_id: str
    model_config = ConfigDict(extra="forbid")

    # private attrs
    _lambda: float = PrivateAttr(default=0.0)
    _store: Dict[str, MemoryItem] = PrivateAttr(default_factory=dict)

    # ------------------------------------------------------------------
    def __init__(self, **data):
        super().__init__(**data)
        # lambda such that importance halves after STEPS_PER_SIM_DAY / ln(1/target)
        object.__setattr__(
            self,
            "_lambda",
            math.log(1 / FORGET_TARGET) / STEPS_PER_SIM_DAY,
        )

    # convenience
    @property
    def _index(self):
        return _INDEX

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    def add(
        self,
        text: str,
        *,
        importance: float = 0.5,
        current_step: int = 0,
    ) -> MemoryItem:
        emb = embed_text(text)
        vec_id = f"{self.agent_id}|{hash(text)}|{len(text)}"

        # push to Pinecone (embedding only — cheap)
        self._index.upsert(
            vectors=[(vec_id, emb, {"text": text})],
            namespace=self.agent_id,
        )

        # cache locally
        item = MemoryItem(
            vec_id=vec_id,
            text=text,
            first_added_step=current_step,
            last_access_step=current_step,
            importance=importance,
        )
        self._store[vec_id] = item
        return item

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        k: int = 5,
        current_step: int,
    ) -> List[RetrievedMemory]:
        q_emb = embed_text(query)

        # 1) cosine search on Pinecone
        resp: Any = self._index.query(
            vector=q_emb,
            namespace=self.agent_id,
            top_k=k * 3,  # 3x oversample before re-scoring
        )

        rescored: List[Tuple[str, float]] = []
        for match in resp["matches"]:
            vid = match["id"]
            cosine = match["score"]

            item = self._store.get(vid)
            if not item:  # desync safeguard
                continue

            rec = 1.0 / (1.0 + (current_step - item.last_access_step))
            score = 0.6 * cosine + 0.3 * rec + 0.1 * item.importance
            rescored.append((vid, score))

        rescored.sort(key=lambda t: t[1], reverse=True)
        selected = rescored[:k]

        out: List[RetrievedMemory] = []
        for vid, score in selected:
            item = self._store[vid]

            # rehearsal: small importance boost
            item.importance = min(1.0, item.importance + REHEARSAL_BONUS)
            item.last_access_step = current_step

            out.append(RetrievedMemory(item=item, score=score))

        return out

    # ------------------------------------------------------------------
    def decay_and_prune(self, *, current_step: int) -> None:
        """
        Exponential decay locally; remove vectors only when pruned.
        """
        to_delete: List[str] = []

        for vid, item in list(self._store.items()):
            delta = current_step - item.last_access_step
            item.importance *= math.exp(-self._lambda * delta)

            if item.importance < IMPORTANCE_THRESHOLD:
                to_delete.append(vid)
                self._store.pop(vid)

        if to_delete:
            self._index.delete(ids=to_delete, namespace=self.agent_id)
            logger.debug(
                f"[{self.agent_id}] pruned {len(to_delete)} memories @step {current_step}"
            )

    # ------------------------------------------------------------------
    # Introspection helpers (used by metrics / debug)
    # ------------------------------------------------------------------
    def all_items(self) -> List[MemoryItem]:
        """Return a copy of all local MemoryItems."""
        return list(self._store.values())
