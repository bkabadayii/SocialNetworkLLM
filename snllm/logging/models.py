"""
snllm.logging.models
~~~~~~~~~~~~~~~~~~~~~~~~~
Typed on-disk record of everything that happened in one sim-step.


They’re deliberately independent from Agent internals so that
changes to agent state will not ripple into historical run logs.
"""

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional, Literal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


# ------------------------------------------------------------------ #
class ToolCallRecord(BaseModel):
    """One function-call emitted by an agent during the dialogue."""

    speaker_id: str
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ConversationLog(BaseModel):
    """
    Complete, JSON-serialisable trace of a single dialogue episode.
    """

    # ---------------------------------------------------------------- #
    # Static meta
    agents: Tuple[str, str]  # (a_id, b_id)
    end_reason: Literal["end_tool", "interest_drop", "max_turns"]

    # ---------------------------------------------------------------- #
    # Turn-level traces
    turns: List[Tuple[str, str]]  # (speaker_id, text)
    tool_calls: List[ToolCallRecord]

    # ---------------------------------------------------------------- #
    # Continuous signals
    interest: List[float]  # I0 ... In

    # ---------------------------------------------------------------- #
    # Post-hoc side-effects
    friendship_deltas: Dict[str, float]  # per agent requested delta
    applied_delta: float  # average actually applied
    memory_commits: Dict[str, Optional[str]]  # id → consolidated text | None

    # ---------------------------------------------------------------- #
    model_config = ConfigDict(extra="forbid")


class StepLog(BaseModel):
    """
    Summary of a single simulation step, containing all conversations
    and the current state of the friendship graph.
    """

    step: int
    conversations: List[ConversationLog]
    graph_path: Optional[Path] = Field(
        default=None,
        description="Filesystem location of NetworkX snapshot",
    )
