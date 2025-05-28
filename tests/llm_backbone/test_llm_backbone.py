"""
tests/test_llm_backbone.py

Unit tests for snllm.agents.llm_backbone.LLMBackbone, covering:
  - system message formatting
  - single‐ and multi‐turn conversation
  - function‐calling (ToolSpec → bind_tools → tool_call)
  - dynamic tool registration and rebinding

We cast the `llm` attribute to DummyChat in each test to access stub‐specific
fields (`bound_schemas`, `last_messages`) without static‐typing errors.
"""

from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional, cast

import pytest
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)

import snllm.agents.llm_backbone as lb
from snllm.config.config import GEMINI_MODEL_NAME, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS


# ---------------------------------------------------------------------------- #
# DummyChat stub to replace ChatGoogleGenerativeAI
# ---------------------------------------------------------------------------- #
class DummyChat:
    """
    Stub for ChatGoogleGenerativeAI:
    - records bound function schemas in .bound_schemas
    - captures the last invoke() messages in .last_messages
    - returns the class‐level AIMessage from .next_msg
    """

    next_msg: AIMessage | None = None

    def __init__(
        self,
        *,
        model: str,
        temperature: float,
        max_output_tokens: int,
        google_api_key: str | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.google_api_key = google_api_key

        # fields used by the tests:
        self.bound_schemas: List[Dict[str, Any]] = []
        self.last_messages: List[BaseMessage] = []

    def bind_tools(self, schemas: List[Dict[str, Any]], tool_choice: str = "auto"):
        """Simulate LangChain’s bind_tools API."""
        self.bound_schemas = schemas
        return self

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """Simulate a single LLM turn."""
        # record the incoming conversation
        self.last_messages = messages
        assert DummyChat.next_msg is not None, "Test must set DummyChat.next_msg"
        msg = DummyChat.next_msg
        DummyChat.next_msg = None
        return msg


# ---------------------------------------------------------------------------- #
# Fixture to patch out the real Gemini client
# ---------------------------------------------------------------------------- #
@pytest.fixture(autouse=True)
def patch_chat(monkeypatch):
    # ensure an API key is present (avoiding ADC errors)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    # patch the class in our module
    monkeypatch.setattr(lb, "ChatGoogleGenerativeAI", DummyChat)
    yield


# ---------------------------------------------------------------------------- #
def default_tools():
    return [
        lb.ToolSpec(
            name="adjust_friendship",
            description="Adjust friendship strength.",
            parameters={
                "target_id": {"type": "string"},
                "delta": {"type": "number", "minimum": -1, "maximum": 1},
            },
        )
    ]


# ---------------------------------------------------------------------------- #
def test_system_message_format():
    """build_system_message should include persona, memories, and extra instructions."""
    bb = lb.LLMBackbone(tools=None)
    persona = "Alice, 30, loves chess"
    memories = "- Met Bob yesterday\n- Likes jazz"
    extra = "Always stay in character."
    sys_msg = bb.build_system_message(persona, memories, extra)

    assert "Alice, 30, loves chess" in sys_msg.content
    assert "- Met Bob yesterday" in sys_msg.content
    assert "Always stay in character." in sys_msg.content


# ---------------------------------------------------------------------------- #
def test_single_turn_ask_no_history():
    """
    .ask() with no prior history except one HumanMessage should:
    - invoke LLM with [system_msg, human_msg]
    - return the DummyChat.next_msg content
    """
    bb = lb.LLMBackbone(tools=None)
    sys_msg = bb.build_system_message("P", "M")
    human = HumanMessage(content="Hello, world!")

    DummyChat.next_msg = AIMessage(content="Hi there!")

    # call ask and capture result
    result = bb.ask(sys_msg, history=[human])

    # cast to DummyChat to inspect stub internals
    chat = cast(DummyChat, bb.llm)
    assert chat.last_messages == [sys_msg, human]
    assert result.text == "Hi there!"
    assert result.tool_call is None


# ---------------------------------------------------------------------------- #
def test_multi_turn_history_passed_through():
    """
    .ask() with multi-turn history should forward all messages:
    [system_msg, human1, ai1, human2]
    """
    bb = lb.LLMBackbone(tools=None)
    sys_msg = bb.build_system_message("P", "M")
    h1 = HumanMessage(content="First message")
    a1 = AIMessage(content="Reply 1")
    h2 = HumanMessage(content="Second message")

    DummyChat.next_msg = AIMessage(content="Reply 2")
    result = bb.ask(sys_msg, history=[h1, a1, h2])

    chat = cast(DummyChat, bb.llm)
    assert chat.last_messages == [sys_msg, h1, a1, h2]
    assert result.text == "Reply 2"
    assert result.tool_call is None


# ---------------------------------------------------------------------------- #
def test_tool_call_parsing_and_binding():
    """
    When the model invokes a function:
    - DummyChat.next_msg carries additional_kwargs.function_call
    - result.tool_call captures name & arguments
    - bind_tools() was called exactly once on init if tools provided
    """
    bb = lb.LLMBackbone(tools=default_tools())
    chat = cast(DummyChat, bb.llm)

    # initial bind_tools call
    assert chat.bound_schemas, "Expected initial tools to be bound"
    assert chat.bound_schemas[0]["name"] == "adjust_friendship"

    # simulate a function-call response
    DummyChat.next_msg = AIMessage(
        content="",
        additional_kwargs={
            "function_call": {
                "name": "adjust_friendship",
                "arguments": {"target_id": "Bob", "delta": 0.75},
            }
        },
    )
    sys_msg = bb.build_system_message("P", "M")
    user = HumanMessage(content="Please adjust friendship")
    result = bb.ask(sys_msg, history=[user])

    assert result.text == ""
    assert result.tool_call == {
        "name": "adjust_friendship",
        "arguments": {"target_id": "Bob", "delta": 0.75},
    }


# ---------------------------------------------------------------------------- #
def test_register_tool_and_rebind():
    """
    register_tool() should:
    - append to bb.tools
    - re-bind the client so bound_schemas grows
    """
    bb = lb.LLMBackbone(tools=default_tools())
    chat = cast(DummyChat, bb.llm)
    initial = len(chat.bound_schemas)

    new_tool = lb.ToolSpec(
        name="drop_connection",
        description="End friendship",
        parameters={"target_id": {"type": "string"}},
    )

    bb.register_tool(new_tool)
    chat2 = cast(DummyChat, bb.llm)

    # confirm the new tool is in bb.tools
    assert any(t.name == "drop_connection" for t in bb.tools)

    # confirm the stub client was rebound with an extra schema
    assert len(chat2.bound_schemas) == initial + 1
