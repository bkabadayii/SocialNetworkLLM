"""
snllm.config
============

Central knobs shared across modules.
Override via environment variables or during testing.
"""

import os

# ---------------- simulation cadence -------------------------------- #
# how many simulation steps ≈ one “social day”
STEPS_PER_SIM_DAY = int(os.getenv("SNLLM_STEPS_PER_DAY", "24"))

# friendship exploration probability (epsilon)
FRIENDSHIP_EXPLORATION_EPSILON = float(
    os.getenv("SNLLM_FRIEND_EXPLORATION_EPSILON", "0.1")
)

# tie strength halves after this many steps without contact
FRIENDSHIP_HALF_LIFE_STEPS = int(os.getenv("SNLLM_FRIEND_HALF_LIFE", "30"))

# ---------------- Gemini defaults ----------------------------------- #
GEMINI_MODEL_NAME = os.getenv("SNLLM_GEMINI_MODEL", "gemini-2.0-flash-lite")
GEMINI_TEMPERATURE = float(os.getenv("SNLLM_GEMINI_TEMPERATURE", 0.7))
GEMINI_MAX_TOKENS = int(os.getenv("SNLLM_GEMINI_MAX_TOKENS", 512))
GEMINI_BASE_URL = os.getenv(
    "SNLLM_GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"
)

# ---------------- embedding model ----------------------------------- #
EMBEDDING_MODEL = os.getenv("SNLLM_EMBEDDING_MODEL", "models/text-embedding-004")
EMBEDDING_DIM = int(
    os.getenv("SNLLM_EMBEDDING_DIM", "768")
)  # embedding dimension for EMBEDDING_MODEL

# ---------------- Memory config -------------------------------- #
# importance threshold for memory retention
IMPORTANCE_THRESHOLD = float(os.getenv("SNLLM_IMPORTANCE_THRESHOLD", "0.05"))
# rehearsal bonus for memory importance
REHEARSAL_BONUS = float(os.getenv("SNLLM_REHEARSAL_BONUS", "0.10"))
# memory target: 30 % importance left after one day, this default number is according to the forgettin curve theory.
FORGET_TARGET = float(os.getenv("SNLLM_FORGET_TARGET", "0.3"))
