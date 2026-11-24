# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import logging
from haystack.components.agents.state.state import State
from haystack.components.agents.utils import _INPUT_TOKEN_KEYS, _OUTPUT_TOKEN_KEYS, _first_numeric
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.hooks.utils import _SYSTEM_REMINDER_META_KEY
from haystack.utils.experimental import _experimental

logger = logging.getLogger(__name__)


_FINAL_MESSAGE_TEXT = "The Agent stopped because the token budget was exceeded."

_STEPS_REMAINING_TEMPLATE = (
    "You have {steps_remaining} of {max_agent_steps} steps remaining before this run is stopped. Start working "
    "toward your final answer."
)
_LAST_STEP_TEMPLATE = (
    "This is your last step of {max_agent_steps}. Give your final answer now: you will not see the results of any "
    "further tool calls."
)


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
    from haystack.components.generators.chat import OpenAIResponsesChatGenerator
    from haystack.hooks.budget import TokenBudgetHook

    agent = Agent(
        chat_generator=OpenAIResponsesChatGenerator(),
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


@_experimental
class StepBudgetReminderHook:
    """
    Tells the Agent how many steps it has left once it gets close to `max_agent_steps`.

    This `before_llm` hook adds a reminder before each of the Agent's last `remind_within_steps` LLM calls, telling
    the model how many steps are left and asking it to wrap up. On the last step, it asks for the final answer
    directly.

    The reminder is a user message wrapped in `<system-reminder>` tags and marked with
    `meta={"system_reminder": True}`, so compaction hooks don't mistake it for the user's task. Reminders stay in the
    conversation and appear in the Agent's `messages` output.

    The model can still ignore the reminder. Pair it with an `after_run` hook if a run that ends on `max_agent_steps`
    needs a final answer.

    <!-- test-ignore -->
    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIResponsesChatGenerator
    from haystack.hooks.budget import StepBudgetReminderHook

    agent = Agent(
        chat_generator=OpenAIResponsesChatGenerator(model="gpt-5.4-nano"),
        tools=[web_search],
        max_agent_steps=10,
        hooks={"before_llm": [StepBudgetReminderHook()]},
    )
    ```
    """

    allowed_hook_points = ("before_llm",)

    def __init__(self, *, remind_within_steps: int = 3) -> None:
        """
        Create a step budget reminder hook.

        :param remind_within_steps: Number of remaining steps at which reminders start.
        :raises ValueError: If `remind_within_steps` is less than 1.
        """
        if remind_within_steps < 1:
            raise ValueError(f"`remind_within_steps` must be a positive number of steps, got {remind_within_steps}.")
        self.remind_within_steps = remind_within_steps

    def run(self, state: State) -> None:
        """
        Add a reminder if no more than `remind_within_steps` steps are left.

        :param state: Agent state containing `step_count` and `max_agent_steps`.
        """
        max_agent_steps = state.data["max_agent_steps"]
        # Count the step about to run, so this is 1 on the Agent's last LLM call.
        steps_remaining = max_agent_steps - state.data["step_count"]
        if steps_remaining > self.remind_within_steps:
            return
        if steps_remaining == 1:
            text = _LAST_STEP_TEMPLATE.format(max_agent_steps=max_agent_steps)
        else:
            text = _STEPS_REMAINING_TEMPLATE.format(steps_remaining=steps_remaining, max_agent_steps=max_agent_steps)
        reminder = ChatMessage.from_user(
            text=f"<system-reminder>\n{text}\n</system-reminder>", meta={_SYSTEM_REMINDER_META_KEY: True}
        )
        state.set("messages", [reminder])

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this hook to a dictionary.

        :returns: Serialized representation of the hook.
        """
        return default_to_dict(self, remind_within_steps=self.remind_within_steps)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StepBudgetReminderHook":
        """
        Create a hook from its serialized representation.

        :param data: Serialized hook data.
        :returns: The deserialized hook.
        """
        return default_from_dict(cls, data=data)
