"""
snllm.agents.models
~~~~~~~~~~~~~~~~~~~
Pydantic (v2) data-models for one LLM agent.

These classes are pure data containers - no business logic - so they
can be (de)serialised cheaply and passed between multiprocessing workers
if we later parallelise the simulation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator
from langchain_core.messages import BaseMessage


# ─────────────────────────────────────────────────────────────────────────────
# AgentPersona
# ─────────────────────────────────────────────────────────────────────────────
class AgentPersona(BaseModel):
    """
    Stable identity information injected into every LLM prompt.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    id: str
    name: str
    age: int = Field(ge=0, le=120)
    occupation: str
    political_view: str
    interests: List[str] = Field(min_length=1)
    extra: Optional[str] = None

    @property
    def interest_sentence(self) -> str:
        """Join interests for embedding."""
        return ", ".join(self.interests)


# ─────────────────────────────────────────────────────────────────────────────
# MemoryItem
# ─────────────────────────────────────────────────────────────────────────────
class MemoryItem(BaseModel):  # TODO: Add embeddings if we want to log them
    """
    Full metadata for one memory stored locally.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    vec_id: str  # Unique ID for this memory, e.g. "agent_id|hash(text)|len(text)"
    text: str
    first_added_step: int
    last_access_step: int
    importance: float = Field(0.5, ge=0.0, le=1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Friendship
# ─────────────────────────────────────────────────────────────────────────────
class Friendship(BaseModel):
    """
    Weighted edge between this agent and a peer.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    strength: float = Field(0.0, ge=0.0, le=1.0)
    last_interaction_step: int = 0

    # pydantic-v2 style validator
    @field_validator("strength")
    @classmethod
    def _clip_strength(cls, v: float) -> float:
        """Clamp due to possible floating-point drift."""
        return max(0.0, min(1.0, v))


# ─────────────────────────────────────────────────────────────────────────────
# AgentState
# ─────────────────────────────────────────────────────────────────────────────
class AgentState(BaseModel):
    """
    Complete mutable state of an agent.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,  # to accept NumPy ndarray
    )

    persona: AgentPersona
    memories: List[MemoryItem] = Field(default_factory=list)
    friends: Dict[str, Friendship] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# AgentReply
# ─────────────────────────────────────────────────────────────────────────────
class AgentReply(BaseModel):
    """
    The result of an Agent.react() call.

    `messages` is the full sequence of messages the agent emits:
      - one or more AIMessage instances,
      - zero or more ToolMessage instances,
      - ending in the final AIMessage.

    `tool_calls` is the raw list of tool-call specs (dicts) the agent requested and used.
    """

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,  # to accept Message subclasses
    )

    messages: List[BaseMessage]
    tool_calls: List[dict]
