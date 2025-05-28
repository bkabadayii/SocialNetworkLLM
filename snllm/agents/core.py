"""
snllm.agents.core
~~~~~~~~~~~~~~~~~
Bundle state + managers + LLM backbone for one agent.
"""

# TODO: ?Add HumanMessage fallback for LLMs that don't support ToolMessage

from __future__ import annotations
from typing import List, Optional, Dict, Any
import json


from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from snllm.agents.models import AgentPersona, AgentReply, AgentState
from snllm.agents.memory import MemoryManager
from snllm.agents.friendship import FriendshipManager
from snllm.agents.llm_backbone import LLMBackbone, ToolSpec, LLMResult
import snllm.tools.specs as tool_specs
from snllm.tools.router import ToolRouter
from snllm.tools.return_signals import CONVERSATION_END


class Agent:
    """
    Facade bundling persona + sub-modules for simulation use.
    """

    # ---------------------------------------------------------------- #
    def __init__(
        self,
        persona: AgentPersona,
        *,
        initial_memories: Optional[List[str]] = None,
        extra_tools: Optional[List[ToolSpec]] = None,
    ):
        self.state = AgentState(persona=persona)
        self.memory = MemoryManager(agent_id=persona.id)
        self.friends = FriendshipManager(self.state)

        tools = tool_specs.BUILTIN_TOOLS + (extra_tools or [])
        self.llm = LLMBackbone(tools=tools)
        self._router = ToolRouter(self.memory, self.friends)

        for txt in initial_memories or []:
            self.memory.add(txt, importance=0.7)

    # ---------------------------------------------------------------- #
    # TODO: Handle concurrent tool calls properly.
    def react(
        self,
        partner_persona: AgentPersona,
        history: List[BaseMessage],
        *,
        current_step: int,
        retrieved_k: int = 5,
    ) -> AgentReply:
        """
        Produce next utterance (or execute tool) in dialogue with partner_persona.

        Parameters
        ----------
        partner_persona : AgentPersona
            Full profile of the interlocutor.
        history : list[BaseMessage]
            Turns so far, ending with partner’s latest HumanMessage.
        current_step : int
            Simulation tick.
        retrieved_k : int
            How many memories to inject.

        Returns
        -------
        AIMessage
        """
        # -- latest user text --------------------------------------------------
        latest_human = history[-1]
        assert isinstance(
            latest_human, HumanMessage
        ), "history must end with HumanMessage"
        query_text = (
            latest_human.content
            if isinstance(latest_human.content, str)
            else str(latest_human.content)
        )

        # -- memory retrieval ---------------------------------------------------
        top_mem = self.memory.retrieve(
            query=query_text, k=retrieved_k, current_step=current_step
        )
        mem_block = "\n".join(f"- {m.item.text}" for m in top_mem)

        # -- social context -----------------------------------------------------
        strength = self.friends.strength(partner_persona.id)
        partner_block = f"## Partner profile\n{partner_persona.model_dump_json()}"
        extra = f"{partner_block}\nCurrent friendship strength: {strength:.2f}"

        # -- build prompt & ask -------------------------------------------------
        sys_msg = self.build_system_message(
            persona_block=str(self.state.persona.model_dump()),
            memory_block=mem_block,
            extra_instructions=extra,
        )
        result: LLMResult = self.llm.ask(
            system_msg=sys_msg,
            history=history,
            use_tools=[tool_specs.END_CONVERSATION.name],
        )

        # -- If there's no tool call, just wrap the single AIMessage: -----------
        if not result.tool_call:
            ai = AIMessage(content=result.text)
            return AgentReply(messages=[ai], tool_calls=[])

        # TODO: Handle multiple tool calls in a single turn.
        # Otherwise:
        #   record the agent’s initial “I want to call a tool” reply
        initial = result.raw_message

        #   actually run the tool
        args = (
            json.loads(result.tool_call["arguments"])
            if isinstance(result.tool_call["arguments"], str)
            else result.tool_call["arguments"]
        )
        #   add the target ID to the tool call extra arguments
        extra_args = {
            "target_id": partner_persona.id,
        }

        tool_output = self._router.route(
            result.tool_call["name"],
            arguments=args,
            step=current_step,
            extra_args=extra_args,
        )

        #   if CONVERSATION_END, return immediately
        if tool_output == CONVERSATION_END:
            return AgentReply(
                messages=[initial],
                tool_calls=[result.tool_call],
            )

        #   wrap that tool output in a ToolMessage
        tool_msg = ToolMessage(
            tool_call_id=initial.tool_calls[0]["id"], content=tool_output
        )

        #   re-ask the model for its free-form follow-up reply
        #   now that the tool’s result is “in context”
        #   (we send the same system message and a history extended
        #   by [initial, tool_msg] so the LLM sees what happened)
        follow = self.llm.ask(
            system_msg=sys_msg,
            history=history + [initial, tool_msg],
            use_tools=[tool_specs.END_CONVERSATION.name],
        )
        final = AIMessage(content=follow.text)

        # 4) return the whole sequence and the tool-call spec
        return AgentReply(
            messages=[initial, tool_msg, final],
            tool_calls=[result.tool_call],
        )

    def request_friendship_delta(
        self,
        partner_persona: AgentPersona,
        transcript: list[tuple[str, str]],
        current_step: int,  # not used but required for routing
    ) -> float:
        """
        Prompt the agent
        to self-report how much it wants to change its friendship.
        Returns 0.0 if the tool isn’t used.
        """
        # mini-backbone with a single tool
        convo_text = "\n".join(f"{sid}: {txt}" for sid, txt in transcript)
        sys = self.build_system_message(
            persona_block=str(self.state.persona.model_dump()),
            memory_block="(omitted)",
            extra_instructions=(
                f"You just spoke with {partner_persona.name}.\n"
                "Below is the conversation transcript. "
                f"If your feelings changed, call {tool_specs.ADJUST_FRIENDSHIP.name} with the delta "
                "you desire. If unchanged, reply normally."
            ),
        )
        ask = HumanMessage(content=convo_text)
        result = self.llm.ask(
            system_msg=sys, history=[ask], use_tools=[tool_specs.ADJUST_FRIENDSHIP.name]
        )

        if result.tool_call:
            args = (
                json.loads(result.tool_call["arguments"])
                if isinstance(result.tool_call["arguments"], str)
                else result.tool_call["arguments"]
            )
            extra_args = {"target_id": partner_persona.id}  # add target ID
            self._router.route(
                result.tool_call["name"],
                arguments=args,
                step=current_step,
                extra_args=extra_args,
            )
            return args.get("delta", 0.0)
        return 0.0

    def request_memory_commit(
        self,
        partner_persona: AgentPersona,
        transcript: list[tuple[str, str]],
        current_step: int,
        *,
        default_importance: float = 0.5,
    ) -> Optional[str]:
        """
        Ask the LLM to wrap all key takeaways from the conversation
        into one memory entry via a single REMEMBER_FACT call.

        Returns the consolidated fact-string, or None if none was invoked.
        """
        # 1) Flatten the transcript
        convo_text = "\n".join(f"{sid}: {txt}" for sid, txt in transcript)

        # 2) System prompt telling the model to bundle everything
        sys = self.build_system_message(
            persona_block=str(self.state.persona.model_dump()),
            memory_block="(omitted)",
            extra_instructions=(
                f"You just spoke with {partner_persona.model_dump()}.\n"
                "Below is the conversation transcript you had."
                "Please summarize all key points you want to remember "
                "in a single text string, and call the "
                f"`{tool_specs.REMEMBER_FACT.name}` tool *once* with that string"
                "and the importance of the information you think."
            ),
        )

        # 3) One-turn request, allowing only REMEMBER_FACT
        ask = HumanMessage(content=convo_text)
        result = self.llm.ask(
            system_msg=sys,
            history=[ask],
            use_tools=[tool_specs.REMEMBER_FACT.name],
        )

        # 4) If the model did not call your tool, bail out
        if (
            not result.tool_call
            or result.tool_call["name"] != tool_specs.REMEMBER_FACT.name
        ):
            return None

        # 5) Parse its single argument
        raw_args = result.tool_call.get("arguments", {})
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        consolidated = args.get("text", "").strip()
        if not consolidated:
            return None

        # 6) Route that one entry into memory
        importance = float(args.get("importance", default_importance))
        self._router.route(
            tool_specs.REMEMBER_FACT.name,
            arguments={"text": consolidated, "importance": importance},
            step=current_step,
        )

        return consolidated

    # ---------------------------------------------------------------- #
    def decay(
        self, current_step: int
    ) -> None:  # TODO: to be used in simulation_runner.py
        self.memory.decay_and_prune(current_step=current_step)
        self.friends.decay(current_step=current_step)

    # ---------------------------------------------------------------- #
    # Helper
    # ---------------------------------------------------------------- #
    def build_system_message(
        self,
        persona_block: str,
        memory_block: str,
        extra_instructions: str = "",
    ) -> SystemMessage:
        """
        Format the stable context for each turn.

        Parameters
        ----------
        persona_block : str
            Text summary of the agent’s persona.
        memory_block : str
            Bullet list of retrieved memories.
        extra_instructions : str
            Additional rules or state (e.g. interest scores).

        Returns
        -------
        SystemMessage
            To prepend before any history.
        """
        content = (
            "You are role-playing the persona below.\n\n"
            f"## Persona\n{persona_block}\n\n"
            f"## Retrieved memories\n{memory_block}\n\n"
            f"{extra_instructions}".strip()
        )
        return SystemMessage(content=content)

    # ---------------------------------------------------------------- #
    # Introspection
    # ---------------------------------------------------------------- #
    def memories(self) -> List[str]:
        return [m.item.text for m in self.memory.retrieve("", k=100, current_step=0)]

    def friendships(self) -> Dict[str, float]:
        return {fid: f.strength for fid, f in self.state.friends.items()}

    def persona_json(self) -> Dict[str, Any]:
        return self.state.persona.model_dump()

    # ---------------------------------------------------------------- #
    @property
    def id(self) -> str:  # noqa: D401
        return self.state.persona.id
