"""
# TODO: ACTUAL LITERATURE BASED IMPLEMENTATION
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
from snllm.utils.embeddings import embed_text


def _sentiment_stub(text: str) -> float:
    """
    Very light sentiment proxy ∈ [-1,1].
    Replace with a proper model if desired.
    """
    lo = text.lower()
    if any(w in lo for w in ("great", "thanks", "glad", "happy")):
        return 1.0
    if any(w in lo for w in ("hate", "angry", "bad", "sad")):
        return -1.0
    return 0.0


def _novelty(u1: str, u0: str) -> float:
    """
    1 - cosine similarity between two utterances, in [0,1].
    """
    v1 = np.array(embed_text(u1))
    v0 = np.array(embed_text(u0))
    cos = float(v1 @ v0 / (np.linalg.norm(v1) * np.linalg.norm(v0) + 1e-9))
    return 1.0 - max(0.0, min(1.0, cos))


@dataclass
class InterestModel:
    """
    EWMA interest tracker Iₜ  ∈ [0,1].

        Iₜ = (1-λ)·Iₜ₋₁ + λ·( wₙ·novelty + wₐ·affect )

    Parameters are exposed so researchers can ablate.
    """

    λ: float = 0.20  # recency weight   (≈ half-life 3 turns)
    w_novelty: float = 0.80  # weight for novelty term
    w_affect: float = 0.20  # weight for sentiment term
    I0: float = 1.0  # initial interest

    # internal trace (first element = I0)
    _trace: list[float] = None  # type: ignore[assignment]

    # ---------------------------------------------------------------- #
    def __post_init__(self) -> None:
        self._trace = [self.I0]

    # ---------------------------------------------------------------- #
    def update(self, new_utt: str, prev_utt: str) -> float:
        """
        Update interest with a new utterance new_utt
        following prev_utt and return the new value.
        """
        return self.I0  # TODO: TEMP
        I_prev = self._trace[-1]
        nov = _novelty(new_utt, prev_utt)
        aff = (_sentiment_stub(new_utt) + 1) / 2  # → [0,1]
        I_new = (1 - self.λ) * I_prev + self.λ * (
            self.w_novelty * nov + self.w_affect * aff
        )
        I_new = max(0.0, min(1.0, I_new))
        self._trace.append(I_new)
        return I_new

    # ---------------------------------------------------------------- #
    def current(self) -> float:
        return self._trace[-1]

    def trace(self) -> list[float]:
        return self._trace[:]
