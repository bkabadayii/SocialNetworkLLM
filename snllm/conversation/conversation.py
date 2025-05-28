"""
snllm.conversation.conversation
~~~~~~~~~~~~~~~~~~~~~~~~~

ConversationEngine coordinates one dialogue between two `Agent`s.

Stopping rules
--------------
* interest < `interest_threshold`
* ≥ `max_turns`
* `end_conversation` tool fired by either side

Afterwards each agent is asked (via its own LLM) to
* adjust friendship strength with the other agent
* commit memories (if it has a memory manager)

A single call to `run(step)` constitutes one simulation step.
"""

from __future__ import annotations
from typing import Tuple, Optional
import json
import time

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, ToolMessage

from snllm.agents.core import Agent
from snllm.conversation.interest import InterestModel
from snllm.logging.models import ConversationLog, ToolCallRecord
from snllm.tools.specs import END_CONVERSATION


class ConversationEngine:
    # -------------------------------------------------------------- #
    def __init__(
        self,
        agent_a: Agent,
        agent_b: Agent,
        *,
        interest_model: InterestModel | None = None,
        max_turns: int = 8,
        interest_threshold: float = 0.25,
    ):
        self.a = agent_a
        self.b = agent_b
        self.max_turns = max_turns
        self.theta = interest_threshold
        self.I = interest_model or InterestModel()

        # ── log ─────────────────────────────── #
        self._log = ConversationLog(
            agents=(self.a.id, self.b.id),
            end_reason="max_turns",  # default, may change later
            turns=[],
            tool_calls=[],
            interest=[self.I.current()],
            friendship_deltas={},  # filled in _apply_friendship…
            applied_delta=0.0,
            memory_commits={},  # filled in _apply_memory…
        )

        self._ended = False

    # -------------------------------------------------------------- #
    def run(self, step: int, delay: Optional[int] = None) -> None:
        """
        Alternate speakers until stopping criteria met.
        step is forwarded to Agent.react
        Parameters:
        - step: the current simulation step (for logging and agent state)
        - delay: optional delay before each agent's request to match LLM api limits.
        """
        hist_a: list[BaseMessage] = []
        hist_b: list[BaseMessage] = []

        speaker, listener = self.a, self.b
        hist_s, hist_l = hist_a, hist_b

        turn = 0
        while (
            turn < self.max_turns and self.I.current() >= self.theta and not self._ended
        ):
            # -- delay if requested ---------------------------------------
            if delay:
                time.sleep(delay)

            # -------- Agent speaks ----------------------------------
            reply = speaker.react(
                partner_persona=listener.state.persona,
                history=hist_s or [HumanMessage(content="Hi!")],
                current_step=step,
            )

            # ---------------- tool-call branch ----------------------- #
            if reply.tool_calls:
                # If there are tool calls, we add them to the log and check for END_CONVERSATION.
                for tc in reply.tool_calls:
                    tc_name = tc["name"]
                    tc_args = tc.get("arguments", {})
                    # cast to dict if not already
                    if isinstance(tc_args, str):
                        tc_args = json.loads(tc_args)

                    self._log.tool_calls.append(
                        ToolCallRecord(
                            speaker_id=speaker.id,
                            name=tc_name,
                            arguments=tc_args,
                        )
                    )
                    # if END_CONVERSATION, bail out immediately
                    if tc_name == END_CONVERSATION.name:
                        self._ended = True
                        self._log.end_reason = "end_tool"
                        break

            # Otherwise record each message in order
            for msg in reply.messages:
                if isinstance(msg, AIMessage):
                    # always add to the speaker’s own history
                    hist_s.append(msg)

                    # only the _final_ AIMessage gets shown to the other agent
                    #    (ToolMessages are never sent onward)
                    if msg is reply.messages[-1]:
                        msg_str = (
                            msg.content
                            if isinstance(msg.content, str)
                            else str(msg.content)
                        )
                        hist_l.append(HumanMessage(content=msg_str))

                        self._log.turns.append((speaker.id, msg_str))

                        # update interest if we have at least one prior turn
                        if len(self._log.turns) >= 2:
                            prev = self._log.turns[-2][1]
                            self.I.update(msg_str, prev)
                            self._log.interest.append(self.I.current())

                elif isinstance(msg, ToolMessage):
                    # ToolMessages only live in the speaker’s history
                    hist_s.append(msg)

            # swap roles
            speaker, listener = listener, speaker
            hist_s, hist_l = hist_l, hist_s
            turn += 1

        # if loop broke due to interest drop
        if not self._ended and self.I.current() < self.theta:
            self._log.end_reason = "interest_drop"

        # -- end of conversation ----------------------------------------
        self._end_run(step)

    # ------------------------------------------------------------------ #
    # private helpers
    # ------------------------------------------------------------------ #
    def _end_run(self, step) -> None:
        """
        End the current conversation run, applying friendship deltas
        and committing memories.
        This is called at the end of each conversation run,
        regardless of whether it ended naturally or via a tool call.
        """
        # -- mark conversation as ended ----------------------------------
        self._ended = True

        # -- touch friends -----------------------------------------------
        self.a.friends.touch(self.b.id, step)
        self.a.friends.touch(self.b.id, step)

        # -- friendship adjustment after dialogue ------------------------
        self._apply_friendship_deltas(step)

        # -- commit memories after dialogue ------------------------------
        self._apply_memory_commits(step)

    def _apply_friendship_deltas(self, step: int) -> None:
        """
        Ask each agent to adjust friendship strength with the other
        based on the conversation outcome.
        """
        delta_a = self.a.request_friendship_delta(
            self.b.state.persona, self._log.turns, current_step=step
        )
        delta_b = self.b.request_friendship_delta(
            self.a.state.persona, self._log.turns, current_step=step
        )

        # average the deltas to get the final applied delta
        applied_delta = (delta_a + delta_b) / 2.0

        # logg the deltas
        self._log.applied_delta = applied_delta
        self._log.friendship_deltas = {
            self.a.id: delta_a,
            self.b.id: delta_b,
        }

    def _apply_memory_commits(self, step: int) -> None:
        """
        Ask each agent to commit their memories after the conversation.
        This is a no-op if the agent does not have a memory manager.
        """
        mem_a = self.a.request_memory_commit(
            self.b.state.persona, self._log.turns, current_step=step
        )
        mem_b = self.b.request_memory_commit(
            self.a.state.persona, self._log.turns, current_step=step
        )

        # log the memory commits
        self._log.memory_commits = {self.a.id: mem_a, self.b.id: mem_b}

    # ------------------------------------------------------------------ #
    # public getters
    # ------------------------------------------------------------------ #
    def transcript(self) -> list[Tuple[str, str]]:
        return self._log.turns.copy()

    def interest_trace(self) -> list[float]:
        return self._log.interest.copy()

    def turn_count(self) -> int:
        return len(self._log.turns)

    def log(self) -> ConversationLog:
        """Return the full, validated log object (deep-copy safe)."""
        return self._log.model_copy()
