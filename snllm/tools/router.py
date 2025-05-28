"""
snllm.tools.router
~~~~~~~~~~~~~~~~~~

Name → side-effect dispatcher for function-calls.

• Each handler is a regular function for clarity.
• All handlers return None so they conform to `DispatchFn`.
"""

from __future__ import annotations
from typing import Dict, Callable, Any

import snllm.tools.specs as tool_specs
import snllm.tools.return_signals as RETURN_SIGNALS
from snllm.agents.memory import MemoryManager
from snllm.agents.friendship import FriendshipManager


# TODO: Add dict, list return types for router.
DispatchFn = Callable[[Dict[str, Any], int], str]


class ToolRouter:
    """Route tool-call names to concrete handler functions."""

    # ---------------------------------------------------------- #
    def __init__(self, mem: MemoryManager, friends: FriendshipManager):
        self.mem = mem
        self.friends = friends

        # register built-in tools
        self._routes: dict[str, DispatchFn] = {
            tool_specs.ADJUST_FRIENDSHIP.name: self._handle_adjust_friendship,
            tool_specs.REMEMBER_FACT.name: self._handle_remember_fact,
            tool_specs.END_CONVERSATION.name: self._handle_end_conversation,
        }

    # ---------------------------------------------------------- #
    # Handlers
    # ---------------------------------------------------------- #
    def _handle_adjust_friendship(self, args: Dict[str, Any], step: int) -> str:
        """Delta ∈ [-1, 1] applied to tie strength."""
        self.friends.adjust(
            target_id=args["target_id"],
            delta=args["delta"] / 2,  # divide by 2 to represent a single agent's view
            current_step=step,
        )

        return f"Friendship with {args['target_id']} adjusted by {args['delta']}. You can continue chatting."

    def _handle_remember_fact(self, args: Dict[str, Any], step: int) -> str:
        """Adds text to vector store; ignore returned MemoryItem."""
        self.mem.add(
            text=args["text"],
            importance=args.get("importance", 0.5),
            current_step=step,
        )
        return f"Remembered fact: {args['text']}\nYou can continue chatting."

    def _handle_end_conversation(self, args: Dict[str, Any], step: int) -> str:
        """Ends the conversation."""
        return RETURN_SIGNALS.CONVERSATION_END

    # ---------------------------------------------------------- #
    # Public dispatch
    # ---------------------------------------------------------- #
    def route(
        self,
        name: str,
        arguments: Dict[str, Any],
        step: int,
        extra_args: Dict[str, Any] = {},
    ) -> str:
        handler = self._routes.get(name)
        if handler:
            # Get extra args based on tool spec
            if name == tool_specs.ADJUST_FRIENDSHIP.name:
                arguments["target_id"] = extra_args.get("target_id", "")

            # Call the handler with the provided arguments and step
            return handler(arguments, step)
        return f"Unknown tool: {name}. You can continue chatting."
