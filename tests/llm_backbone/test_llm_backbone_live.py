# tests/test_llm_backbone_live.py

"""
Live integration tests for snllm.agents.llm_backbone.LLMBackbone against
the real Google Gemini API.

❗ WARNING ❗
These tests use actual API calls and will consume quota. They run only if
you set the following environment variables:

    SNLLM_RUN_GEMINI_TESTS=true
    GOOGLE_API_KEY

Example:
    export SNLLM_RUN_GEMINI_TESTS=true
    export GOOGLE_API_KEY="your_real_key"
    pytest tests/test_llm_backbone_live.py -q
"""

import os
import json
from typing import List

import pytest
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage

from snllm.agents.llm_backbone import LLMBackbone, ToolSpec


def should_run() -> bool:
    return bool(
        os.getenv("SNLLM_RUN_GEMINI_TESTS", "").lower() == "true"
        and os.getenv("GOOGLE_API_KEY")
    )


@pytest.mark.skipif(not should_run(), reason="Live Gemini tests disabled")
def test_gemini_round_trip_live():
    """
    Send a minimal prompt to Gemini and assert we get
    either text or a tool_call back.
    """
    tools = [
        ToolSpec(
            name="noop_tool",
            description="A no-op function for testing.",
            parameters={"dummy": {"type": "string"}},
        )
    ]
    llm = LLMBackbone(tools=tools)

    # Build the stable context
    sys_msg = SystemMessage(
        "TestAgent, a friendly assistant.\nKeep your responses brief."
    )

    # Single-turn greeting
    history: List[BaseMessage] = [HumanMessage(content="Hello, Gemini!")]
    result = llm.ask(sys_msg, history=history)
    assert result.text or result.tool_call, "Expected non-empty text or a tool_call"

    # Now prompt specifically for a tool call
    tool_request = HumanMessage(content="Please invoke noop_tool with dummy='yes'.")
    result2 = llm.ask(sys_msg, history=[tool_request])

    # If the model does call the tool, ensure args parse correctly.
    if result2.tool_call:
        assert result2.tool_call["name"] == "noop_tool"
        raw_args = result2.tool_call["arguments"]

        # GenAI sometimes returns the arguments as a JSON-encoded string
        if isinstance(raw_args, str):
            args = json.loads(raw_args)
        else:
            args = raw_args

        assert "dummy" in args, "Expect 'dummy' key in function arguments"
        assert isinstance(args["dummy"], str), "Expect dummy to be a string"


@pytest.mark.skipif(not should_run(), reason="Live Gemini tests disabled")
def test_gemini_explicit_tool_prompt():
    """
    Ask the model *explicitly* to perform a tool call. If it cooperates,
    we verify the function name; otherwise we skip.
    """
    tools = [
        ToolSpec(
            name="noop_tool",
            description="A no-op function for testing.",
            parameters={"dummy": {"type": "string"}},
        )
    ]
    llm = LLMBackbone(tools=tools)

    sys_msg = SystemMessage(
        "PromptTester, checking function-calling.\nIf asked, invoke the provided function."
    )

    # Very explicit instruction to call the function
    explicit_request = HumanMessage(
        content=(
            "I want you to call the function noop_tool now. "
            "Pass dummy='triggered' as the argument."
        )
    )
    result = llm.ask(sys_msg, history=[explicit_request])

    if not result.tool_call:
        pytest.skip("Model did not issue a tool_call for explicit prompt")
    # Otherwise verify
    assert result.tool_call["name"] == "noop_tool"
    raw_args = result.tool_call["arguments"]
    if isinstance(raw_args, str):
        args = json.loads(raw_args)
    else:
        args = raw_args

    assert args.get("dummy") == "triggered"
