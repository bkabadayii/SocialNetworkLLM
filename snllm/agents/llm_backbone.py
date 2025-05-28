"""
snllm.agents.llm_backbone
~~~~~~~~~~~~~~~~~~~~~~~~~

Flexible Gemini chat integration with function-calling and multi-turn support.

Key Features
------------
1. Auto-prompt construction
   • Persona & retrieved memories injected via `build_system_message()`
   • Optional extra instructions (e.g. interest scores, global rules)

2. Function-calling (ReAct / Toolformer style)
   • Declare JSON-schema `ToolSpec` objects
   • Bind to Gemini via `bind_tools(..., tool_choice="auto")`
   • Parsed into `LLMResult.tool_call`

3. Single-entry `.ask()`
   • Accepts `SystemMessage` + arbitrary history of `BaseMessage` (human or AI)
   • Returns either free-form text or a tool invocation

4. Extensible
   • Use `.activate_tools()` to narrow active tools by name
   • Use `.deactivate_tools()` to clear all active tools
   • Use `.ask()` to invoke the LLM with the current tool set
   • Hot-load new `ToolSpec` at runtime with `.register_tool()`
   • No downstream code changes required

Environment Variables
---------------------
• `GOOGLE_API_KEY`
• `SNLLM_GEMINI_MODEL`, `SNLLM_GEMINI_TEMPERATURE`, `SNLLM_GEMINI_MAX_TOKENS`
  (defaults in `snllm.config`)

References
----------
- ReAct (Yao et al., 2023)
- Toolformer (Schick et al., 2023)
- “Function-calling” via Google Gemini & LangChain
"""

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_google_genai import ChatGoogleGenerativeAI

from snllm.config.config import (
    GEMINI_MODEL_NAME,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_TOKENS,
)


@dataclass
class LLMResult:
    """
    Wrapper for a single LLM turn.

    Attributes
    ----------
    text : str
        Assistant reply (empty string if a tool was invoked).
    tool_call : dict | None
        Parsed JSON payload when the model calls a declared function.
    raw_message : AIMessage
        Original LangChain AIMessage for logging or debugging.
    """

    text: str
    tool_call: Optional[Dict[str, Any]]
    raw_message: AIMessage


@dataclass
class ToolSpec:
    """
    Declarative JSON-schema for one function the LLM can call.

    Parameters
    ----------
    name : str
        Unique identifier for the function.
    description : str
        Short human-readable explanation.
    parameters : dict
        JSON-schema properties with types, ranges, and descriptions.
    """

    name: str
    description: str
    parameters: Dict[str, Any]

    def as_openai_schema(self) -> Dict[str, Any]:
        """
        Convert this spec into the OpenAI/Google-GenAI function descriptor.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": list(self.parameters.keys()),
            },
        }


class LLMBackbone:
    """
    High-level wrapper over Google Gemini chat with function-calling.

    Usage
    -----
        llm = LLMBackbone(tools=[...])
        sys_msg = llm.build_system_message(persona_block, memory_block)
        history = [HumanMessage(...), AIMessage(...)]
        result = llm.ask(sys_msg, history=history)
    """

    def __init__(
        self,
        tools: Optional[List[ToolSpec]] = None,
        *,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ):
        """
        Initialize the Gemini client and bind any provided tools.

        Parameters
        ----------
        tools : list[ToolSpec] | None
            JSON-schema functions the model may invoke.
        model_name : str | None
            Gemini model (overrides config.GEMINI_MODEL_NAME).
        temperature : float | None
            Sampling temperature (overrides config.GEMINI_TEMPERATURE).
        max_output_tokens : int | None
            Max tokens in response (overrides config.GEMINI_MAX_TOKENS).
        """
        # The superset of all possible tools
        self.all_tools: List[ToolSpec] = tools or []
        # The subset currently bound into the client
        self.active_tools: List[ToolSpec] = []

        self._model_name = model_name or GEMINI_MODEL_NAME
        self._temperature = (
            temperature if temperature is not None else GEMINI_TEMPERATURE
        )
        self._max_tokens = max_output_tokens or GEMINI_MAX_TOKENS
        self.llm: Runnable[LanguageModelInput, BaseMessage] | ChatGoogleGenerativeAI = (
            self._make_llm()
        )

    def _make_llm(
        self,
    ) -> ChatGoogleGenerativeAI:
        """
        Instantiate the raw Gemini chat client and bind tools via bind_tools().

        Returns
        -------
        Runnable or ChatGoogleGenerativeAI
            A callable client exposing `.invoke(messages: List[BaseMessage])`.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        client = ChatGoogleGenerativeAI(
            model=self._model_name,
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
            google_api_key=api_key,
        )
        return client

    def register_tool(self, tool: ToolSpec) -> None:
        """
        Dynamically add a new function and rebind the client.

        Parameters
        ----------
        tool : ToolSpec
            New JSON-schema to expose to the LLM.
        """
        self.all_tools.append(tool)

    def activate_tools(self, tool_names: Optional[List[str]]) -> None:
        """
        Narrow active_tools to only those whose name is in `tool_names`,
        then re-bind the client.
        Parameters
        ----------
        tool_names : list[str]
            Names of tools to activate. If empty, no tools are activated.
            If None, all tools are activated.
        """
        tools_to_activate = (
            tool_names if tool_names else [t.name for t in self.all_tools]
        )
        name_set = set(tools_to_activate)
        self.active_tools = [t for t in self.all_tools if t.name in name_set]
        if self.active_tools:
            client = self._make_llm()  # re-initialize the client
            schemas = [t.as_openai_schema() for t in self.active_tools]
            client = client.bind_tools(schemas, tool_choice="auto")

            self.llm = client

    def deactivate_tools(self) -> None:
        """
        Clear all active tools (unbind every function)
        """
        self.active_tools = []
        self.llm = self._make_llm()

    def ask(
        self,
        system_msg: SystemMessage,
        history: Optional[List[BaseMessage]] = None,
        use_tools: Optional[List[str]] = None,
    ) -> LLMResult:
        """
        Execute one LLM turn over the given conversation context.

        1. Concatenate [system_msg] + history.
        2. Invoke Gemini via `.invoke(messages)`.
        3. Parse either free-form text or a tool-call.

        Parameters
        ----------
        system_msg : SystemMessage
            The persona + memory + instructions context.
        history : list[BaseMessage] | None
            Past messages (HumanMessage or AIMessage), including the
            latest user utterance as a HumanMessage.
        use_tools : list[str] | None
            If provided, only use tools with these names.
            If None, use currently active tools.

        Returns
        -------
        LLMResult
            Contains either `text` or `tool_call` and the raw AIMessage.
        """
        # Prepare conv history
        convo: List[BaseMessage] = [system_msg]
        if history:
            convo.extend(history)

        # Activate tools if specified
        prior_tools = [t.name for t in self.active_tools]
        if use_tools:
            self.activate_tools(use_tools)

        ai_msg: AIMessage = self.llm.invoke(convo)  # type: ignore[arg-type]

        if use_tools:
            # Deactivate tools after the call
            self.deactivate_tools()
            # Activate the prior tools again
            self.activate_tools(prior_tools)

        fn = ai_msg.additional_kwargs.get("function_call")
        if fn:
            return LLMResult(
                text=str(ai_msg.content) or "",
                tool_call={
                    "name": fn["name"],
                    "arguments": fn.get("arguments", {}),
                },
                raw_message=ai_msg,
            )

        content: Union[str, list[Any], dict[str, Any]] = ai_msg.content
        if not isinstance(content, str):
            content = json.dumps(content)
        return LLMResult(text=content, tool_call=None, raw_message=ai_msg)
