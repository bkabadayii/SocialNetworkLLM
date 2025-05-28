"""
snllm.utils.embeddings
~~~~~~~~~~~~~~~~~~~~~~

Batch-aware wrapper for Google / Gemini embeddings API

Key ideas
---------
*  Calls arriving within a short window (200 ms) are coalesced into one
   batch (`/embeddings.create` supports list input), counting as a single
   quota unit.
*  A token-bucket throttles the number of batch requests to ≤ 150/min.
*  Results are cached (LRU ✕ 10 000) so repeated strings are free.

The batching thread is entirely internal; other modules remain oblivious.
"""

from __future__ import annotations

import os
import time
import threading
from collections import deque
from functools import lru_cache
from typing import List, Tuple

import openai

from snllm.config.config import EMBEDDING_MODEL, GEMINI_BASE_URL

# ──────────────────────────────────────────────────────────────────────
# Constants & tuneables
# ──────────────────────────────────────────────────────────────────────
_API_KEY = os.getenv("GOOGLE_API_KEY")
if _API_KEY is None:
    raise EnvironmentError("GOOGLE_API_KEY must be set for embeddings.")

# TODO: make these configurable via snllm.config
BATCH_INTERVAL_MS = 200  # collect requests for 0.2 s
BATCH_SIZE = 96  # Google currently allows 100
MAX_REQUESTS_PER_MIN = 150  # Quota for batch calls
CACHE_SIZE = 10_000  # LRU cache for duplicate text


# ──────────────────────────────────────────────────────────────────────
# Lazy OpenAI-compatible client
# ──────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _client() -> openai.OpenAI:
    return openai.OpenAI(api_key=_API_KEY, base_url=GEMINI_BASE_URL)


# ──────────────────────────────────────────────────────────────────────
# Simple token-bucket rate-limiter for batch requests
# ──────────────────────────────────────────────────────────────────────
class _RateLimiter:
    def __init__(self, rate_per_min: int):
        self.capacity = rate_per_min
        self.tokens = rate_per_min
        self.refill_ts = time.time()
        self.lock = threading.Lock()

    def consume(self) -> None:
        with self.lock:
            now = time.time()
            # refill tokens
            delta = now - self.refill_ts
            self.refill_ts = now
            self.tokens = min(self.capacity, self.tokens + delta * self.capacity / 60.0)
            if self.tokens < 1:
                # wait for next token
                sleep_for = (1 - self.tokens) * 60.0 / self.capacity
                time.sleep(sleep_for)
                self.tokens += 1
            else:
                self.tokens -= 1


_RATE_LIMITER = _RateLimiter(MAX_REQUESTS_PER_MIN)

# ──────────────────────────────────────────────────────────────────────
# Request / response queue infrastructure
# ──────────────────────────────────────────────────────────────────────
# Each enqueued item: (text, threading.Event, output_holder)
_REQ_Q: "deque[Tuple[str, threading.Event, list]]" = deque()
_Q_LOCK = threading.Lock()

# start-up flag so we only launch the worker once
_started = threading.Event()


def _start_worker() -> None:
    if _started.is_set():
        return
    _started.set()
    t = threading.Thread(target=_batch_worker, name="EmbBatcher", daemon=True)
    t.start()


def _batch_worker() -> None:
    """
    Infinite loop:  flush queue every BATCH_INTERVAL_MS or when it
    reaches BATCH_SIZE, obeying the rate-limiter.
    """
    while True:
        time.sleep(BATCH_INTERVAL_MS / 1000.0)
        with _Q_LOCK:
            if not _REQ_Q:
                continue
            batch: List[Tuple[str, threading.Event, list]] = [
                _REQ_Q.popleft() for _ in range(min(BATCH_SIZE, len(_REQ_Q)))
            ]

        # --- API call (may block on rate-limit) ----------------------
        texts = [t for t, _, _ in batch]
        try:
            _RATE_LIMITER.consume()
            resp = _client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
            vectors = [d.embedding for d in resp.data]
        except Exception as exc:  # pragma: no cover – network errors
            vectors = [[0.0] * 768] * len(batch)  # fallback zero-vecs
            print("Embedding batch failed:", exc)

        # --- fulfil promises ----------------------------------------
        for (_, event, holder), vec in zip(batch, vectors):
            holder.append(vec)  # write result
            event.set()  # unblock caller


# ──────────────────────────────────────────────────────────────────────
# Public helper
# ──────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=CACHE_SIZE)
def _embed_cached(text: str) -> List[float]:
    """Real API embedding, routed through batching queue."""
    _start_worker()

    ready = threading.Event()
    out: list = []

    with _Q_LOCK:
        _REQ_Q.append((text, ready, out))

    ready.wait()  # block until worker sets the result
    return out[0]  # type: ignore[index]


def embed_text(text: str) -> List[float]:  # noqa: D401
    """Return a 768-d Gemini embedding, batched transparently."""
    return list(_embed_cached(text))
