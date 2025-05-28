"""
snllm.tools.specs
~~~~~~~~~~~~~~~~~~~
Built-in ToolSpec definitions shared by all agents.
Users can import these or create their own.
"""

from snllm.agents.llm_backbone import ToolSpec

ADJUST_FRIENDSHIP = ToolSpec(
    name="adjust_friendship",
    description="Increase or decrease friendship strength.",
    parameters={
        "delta": {
            "type": "number",
            "minimum": -1,
            "maximum": 1,
            "description": "Positive to strengthen, negative to weaken.",
        },
    },
)

REMEMBER_FACT = ToolSpec(
    name="remember_fact",
    description="Store a fact in long-term memory.",
    parameters={
        "text": {"type": "string"},
        "importance": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
    },
)

END_CONVERSATION = ToolSpec(
    name="end_conversation",
    description="Politely terminate the dialogue.",
    parameters={},
)

BUILTIN_TOOLS = [ADJUST_FRIENDSHIP, REMEMBER_FACT, END_CONVERSATION]
