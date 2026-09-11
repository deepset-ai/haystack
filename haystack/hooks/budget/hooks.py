# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import logging
from haystack.components.agents.state.state import State
from haystack.components.agents.utils import _INPUT_TOKEN_KEYS, _OUTPUT_TOKEN_KEYS, _first_numeric
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.utils.experimental import _experimental

logger = logging.getLogger(__name__)


_FINAL_MESSAGE_TEXT = "The Agent stopped because the token budget was exceeded."


@_experimental
class TokenBudgetHook:
    """
    Stop an Agent run when its token usage reaches a configured budget.

    The hook runs at the `before_llm` hook point and checks the cumulative token usage recorded in the Agent state.
    When the budget is reached, the run ends before the next LLM call with the exit reason `"token_budget_exceeded"`.

    Only calls made by the Agent's chat generator contribute to `token_usage`; calls made by tools or other hooks are
    not included.

    <!-- test-ignore -->
    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.hooks.budget import TokenBudgetHook

    agent = Agent(
        chat_generator=OpenAIChatGenerator(),
        tools=[web_search],
        hooks={"before_llm": [TokenBudgetHook(max_total_tokens=100_000)]},
    )

    result = agent.run(messages=[...])
    ```
    """

    allowed_hook_points = ("before_llm",)

    def __init__(self, *, max_total_tokens: int, add_final_message: bool = False) -> None:
        """
        Create a token budget hook.

        :param max_total_tokens: Maximum cumulative token usage before the Agent is stopped.
        :param add_final_message: Whether to append an assistant message explaining why the Agent stopped.
        :raises ValueError: If `max_total_tokens` is less than 1.
        """
        if max_total_tokens < 1:
            raise ValueError(f"`max_total_tokens` must be a positive number of tokens, got {max_total_tokens}.")
        self.max_total_tokens = max_total_tokens
        self.add_final_message = add_final_message

    def run(self, state: State) -> None:
        """
        Stop the Agent if its cumulative token usage has reached the budget.

        :param state: Agent state containing the cumulative token usage.
        """
        usage = state.data.get("token_usage") or {}
        # Not every chat generator reports `total_tokens`, so fall back to summing the input and output keys across
        # the known naming conventions.
        total_tokens = _first_numeric(usage, ("total_tokens",))
        if not total_tokens:
            total_tokens = _first_numeric(usage, _INPUT_TOKEN_KEYS) + _first_numeric(usage, _OUTPUT_TOKEN_KEYS)
        if total_tokens >= self.max_total_tokens:
            logger.warning(
                "Agent reached its token budget of {max_total_tokens} ({total_tokens} used); requesting a stop.",
                max_total_tokens=self.max_total_tokens,
                total_tokens=total_tokens,
            )
            state.set("stop_run", "token_budget_exceeded")
            if self.add_final_message:
                state.set("messages", [ChatMessage.from_assistant(_FINAL_MESSAGE_TEXT)])

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this hook to a dictionary.

        :returns: Serialized representation of the hook.
        """
        return default_to_dict(self, max_total_tokens=self.max_total_tokens, add_final_message=self.add_final_message)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TokenBudgetHook":
        """
        Create a hook from its serialized representation.

        :param data: Serialized hook data.
        :returns: The deserialized hook.
        """
        return default_from_dict(cls, data=data)
