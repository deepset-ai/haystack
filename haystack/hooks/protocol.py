# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, Protocol, get_args

from haystack.components.agents.state.state import State

# Lifecycle points at which the Agent invokes hooks.
HookEvent = Literal["before_llm", "before_tool", "on_exit"]

BEFORE_LLM: HookEvent = "before_llm"
BEFORE_TOOL: HookEvent = "before_tool"
ON_EXIT: HookEvent = "on_exit"
VALID_HOOK_EVENTS: tuple[HookEvent, ...] = get_args(HookEvent)


class Hook(Protocol):
    """
    A callable the Agent invokes at a point in its run loop, receiving the live `State`.

    A hook influences the run only by mutating `State` in place. At least `messages` (the conversation),
    `step_count`, `token_usage` and `tool_call_counts` are available; any additional keys defined in the Agent's
    `state_schema` are available too. The same hook object can be registered under multiple events.

    Implement this protocol directly for stateful hooks (e.g. one wrapping a component), or use the `@hook`
    decorator to wrap a plain `(State) -> None` function.

    A hook may additionally define `async def run_async(self, state: State) -> None` for true async behavior; when
    absent, the Agent calls `run` during async runs. It is left off this protocol on purpose so sync-only hooks
    don't have to implement it.

    A hook may also implement the optional lifecycle methods `warm_up` / `warm_up_async` and `close` / `close_async`.
    The Agent calls them from its own `warm_up` / `warm_up_async` and `close` / `close_async`, so a hook can defer
    opening clients or reading credentials until warm-up and release them on close.
    """

    def run(self, state: State) -> None:
        """Run the hook against the live `State`, mutating it in place."""
        ...

    def to_dict(self) -> dict[str, Any]:
        """Serialize the hook to a dictionary."""
        ...

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Hook":
        """Deserialize the hook from a dictionary."""
        ...
