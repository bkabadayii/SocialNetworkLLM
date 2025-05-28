"""
snllm.config.network
~~~~~~~~~~~~~~~~~~~~
Typed configurations for initial friendship graphs.

All configs inherit from `NetworkEnvironmentConfig` so the runner can
use `isinstance()` dispatch without caring about the concrete flavour.
"""

from __future__ import annotations

from typing import Dict, Tuple, Literal
from pydantic import BaseModel, Field, ConfigDict


class NetworkEnvironmentConfig(BaseModel):
    """
    Abstract base-class. Concrete subclasses must implement
    `kind` as a Literal so we can pattern-match.
    """

    kind: str = Field(..., description="Discriminator set by subclasses")

    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ #
class EmptyNetworkConfig(NetworkEnvironmentConfig):
    """No edges to start with."""

    kind: str = Field(
        default="empty",
        description="Discriminator for an empty network",
    )


# ------------------------------------------------------------------ #
class FullNetworkConfig(NetworkEnvironmentConfig):
    """Complete graph with identical weight on every edge."""

    kind: str = Field(
        default="full",
        description="Discriminator for a full network",
    )
    default_strength: float = Field(
        0.3, description="Uniform edge strength for the complete graph"
    )


# ------------------------------------------------------------------ #
class ErdosRenyiConfig(NetworkEnvironmentConfig):
    """G(n, p) Erdős–Rényi random graph."""

    kind: str = Field(
        default="erdos_renyi",
        description="Discriminator for an Erdos-Renyi graph",
    )
    p: float = Field(
        0.1, ge=0.0, le=1.0, description="Probability of including each edge"
    )
    default_strength: float = Field(
        0.3, description="Uniform edge strength for edges that exist"
    )
    seed: int | None = Field(
        None, description="Optional random seed for reproducibility"
    )


# ------------------------------------------------------------------ #
class CustomNetworkConfig(NetworkEnvironmentConfig):
    """Caller provides an explicit mapping of strengths."""

    kind: str = Field(
        default="custom",
        description="Discriminator for a custom network",
    )
    edges: Dict[Tuple[str, str], float] = Field(
        ...,
        description="Edge strengths keyed by (u, v) tuples (order ignored)",
    )
