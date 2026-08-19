# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.components.agents.state.state import State
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.utils.experimental import _experimental


@_experimental
class TokenBudgetHook:
    """
    Stop an Agent run when its token usage reaches a configured budget.

    The hook runs at the `after_tool` hook point and checks the cumulative `total_tokens` recorded in the Agent state.
    When the configured budget is reached, the Agent stops with the exit reason `"token_budget_exceeded"`.

    <!-- test-ignore -->
    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.hooks.budget import TokenBudgetHook

    agent = Agent(
        chat_generator=OpenAIChatGenerator(),
        tools=[web_search],
        hooks={"after_tool": [TokenBudgetHook(max_total_tokens=100_000)]},
    )

    result = agent.run(messages=[...])
    ```

    The budget is checked after a tool step completes, so the final token usage may exceed it. Only calls made by the
    Agent's chat generator contribute to `token_usage`; calls made by tools or other hooks are not included.
    """

    allowed_hook_points = ("after_tool",)

    def __init__(self, *, max_total_tokens: int) -> None:
        """
        Create a token budget hook.

        :param max_total_tokens: Maximum cumulative token usage before the Agent is stopped.
        :raises ValueError: If `max_total_tokens` is less than 1.
        """
        if max_total_tokens < 1:
            raise ValueError(f"`max_total_tokens` must be a positive number of tokens, got {max_total_tokens}.")
        self.max_total_tokens = max_total_tokens

    def run(self, state: State) -> None:
        """
        Stop the Agent if its cumulative token usage has reached the budget.

        :param state: Agent state containing the cumulative token usage.
        """
        total_tokens = state.data.get("token_usage", {}).get("total_tokens", 0)
        if total_tokens >= self.max_total_tokens:
            state.set("stop_run", "token_budget_exceeded")

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this hook to a dictionary.

        :returns: Serialized representation of the hook.
        """
        return default_to_dict(self, max_total_tokens=self.max_total_tokens)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenBudgetHook":
        """
        Create a hook from its serialized representation.

        :param data: Serialized hook data.
        :returns: The deserialized hook.
        """
        return default_from_dict(cls, data=data)
