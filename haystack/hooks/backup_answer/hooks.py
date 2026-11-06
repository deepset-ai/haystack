# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

from haystack import logging
from haystack.components.agents.state.state import State
from haystack.components.generators.chat.types import ChatGenerator
from haystack.components.generators.utils import _trace_chat_generator_run
from haystack.core.serialization import component_to_dict, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.utils.async_utils import _execute_component_async
from haystack.utils.deserialization import deserialize_component_inplace
from haystack.utils.experimental import _experimental

logger = logging.getLogger(__name__)

_DEFAULT_SYSTEM_PROMPT = (
    "You are given a conversation in which an assistant was working on the user's request using tools, but was "
    "stopped before it could write a final answer. Write the final answer to the user's request now, using only "
    "the information gathered in the conversation so far. Do not call tools. If the gathered information is "
    "incomplete, answer as well as you can and say clearly which parts could not be answered."
)


@_experimental
class BackupAnswerHook:
    """
    Write a final answer when an Agent run stops without one.

    The hook runs at the `after_run` hook point. When the run's `exit_reason` is one of `exit_reasons` (by default
    `"max_agent_steps"`), the run ended on a tool call or tool result instead of a text answer. The hook then makes
    one extra call to its own chat generator, with no tools, over the conversation so far and appends the reply to
    `messages`. The Agent's `last_message` is therefore the backup answer.

    The system messages of the conversation are replaced by `system_prompt`. The `exit_reason` is left unchanged, so
    downstream components can tell a backup answer from a normal one.

    If the chat generator raises or returns no text, a warning is logged and `messages` is left unchanged, unless
    `raise_on_failure` is set.

    The hook's chat generator is warmed up and closed together with the Agent.

    <!-- test-ignore -->
    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIResponsesChatGenerator
    from haystack.hooks.backup_answer import BackupAnswerHook

    agent = Agent(
        chat_generator=OpenAIResponsesChatGenerator(),
        tools=[web_search],
        max_agent_steps=10,
        hooks={"after_run": [BackupAnswerHook(chat_generator=OpenAIResponsesChatGenerator())]},
    )

    result = agent.run(messages=[...])
    print(result["exit_reason"])
    # >> max_agent_steps
    print(result["last_message"].text)
    # >> The best-effort answer based on the evidence gathered so far...
    ```
    """

    allowed_hook_points = ("after_run",)

    def __init__(
        self,
        chat_generator: ChatGenerator,
        *,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        exit_reasons: Sequence[str] = ("max_agent_steps",),
        raise_on_failure: bool = False,
    ) -> None:
        """
        Create a backup answer hook.

        :param chat_generator: The chat generator that writes the backup answer. It is called without tools.
        :param system_prompt: The system prompt for the backup answer call. It replaces the conversation's own system
            messages. Defaults to a generic prompt that asks for a best-effort final answer from the gathered
            information.
        :param exit_reasons: The `exit_reason` values that trigger a backup answer. Defaults to `("max_agent_steps",)`.
            Add other values, such as `"token_budget_exceeded"` from `TokenBudgetHook`, to also cover those exits.
        :param raise_on_failure: Whether to raise an exception if the chat generator fails or returns no text. By
            default the failure is logged and `messages` is left unchanged.
        """
        self.chat_generator = chat_generator
        self.system_prompt = system_prompt
        self.exit_reasons = tuple(exit_reasons)
        self.raise_on_failure = raise_on_failure

    def warm_up(self) -> None:
        """Warm up the hook's chat generator. The Agent calls this from its own `warm_up`."""
        if hasattr(self.chat_generator, "warm_up"):
            self.chat_generator.warm_up()

    async def warm_up_async(self) -> None:
        """Warm up the hook's chat generator asynchronously. The Agent calls this from its own `warm_up_async`."""
        warm_up_async = getattr(self.chat_generator, "warm_up_async", None)
        if warm_up_async is not None:
            await warm_up_async()
        elif hasattr(self.chat_generator, "warm_up"):
            self.chat_generator.warm_up()

    def close(self) -> None:
        """Release the hook's chat generator resources. The Agent calls this from its own `close`."""
        if hasattr(self.chat_generator, "close"):
            self.chat_generator.close()

    async def close_async(self) -> None:
        """Release the hook's chat generator resources asynchronously. The Agent calls this from `close_async`."""
        close_async = getattr(self.chat_generator, "close_async", None)
        if close_async is not None:
            await close_async()
        elif hasattr(self.chat_generator, "close"):
            self.chat_generator.close()

    def run(self, state: State) -> None:
        """
        Append a backup answer to the conversation when the run stopped with one of the configured exit reasons.

        :param state: Agent state holding the conversation and the `exit_reason`.
        :raises RuntimeError: If `raise_on_failure` is set and the chat generator returns no text. Errors raised by
            the chat generator itself are re-raised when `raise_on_failure` is set.
        """
        prompt = self._build_prompt(state=state)
        if prompt is None:
            return
        try:
            with _trace_chat_generator_run(self.chat_generator, {"messages": prompt}) as span:
                result = self.chat_generator.run(messages=prompt)
                span.set_content_tag("haystack.component.output", result)
            reply = _reply_from_result(result=result)
        except Exception as error:
            self._report_failure(error=error)
            return
        # The `messages` handler appends the reply, so it becomes the last message
        state.set("messages", [reply])

    async def run_async(self, state: State) -> None:
        """
        Append a backup answer to the conversation when the run stopped with one of the configured exit reasons.

        Uses the chat generator's `run_async` when it has one, and otherwise runs its `run` in a worker thread.

        :param state: Agent state holding the conversation and the `exit_reason`.
        :raises RuntimeError: If `raise_on_failure` is set and the chat generator returns no text. Errors raised by
            the chat generator itself are re-raised when `raise_on_failure` is set.
        """
        prompt = self._build_prompt(state=state)
        if prompt is None:
            return
        try:
            with _trace_chat_generator_run(self.chat_generator, {"messages": prompt}) as span:
                result = await _execute_component_async(component_instance=self.chat_generator, messages=prompt)
                span.set_content_tag("haystack.component.output", result)
            reply = _reply_from_result(result=result)
        except Exception as error:
            self._report_failure(error=error)
            return
        # The `messages` handler appends the reply, so it becomes the last message
        state.set("messages", [reply])

    def _build_prompt(self, state: State) -> list[ChatMessage] | None:
        """Return the messages for the backup answer call, or `None` when no backup answer is needed."""
        exit_reason = state.data.get("exit_reason")
        if exit_reason not in self.exit_reasons:
            return None
        transcript = [m for m in state.data.get("messages") or [] if not m.is_from(ChatRole.SYSTEM)]
        if not transcript:
            return None
        logger.info(
            "The Agent stopped with exit reason '{exit_reason}', writing a backup answer.", exit_reason=exit_reason
        )
        return [ChatMessage.from_system(self.system_prompt), *transcript]

    def _report_failure(self, error: Exception) -> None:
        """Re-raise a failed backup answer call or log it, so the Agent still returns the conversation so far."""
        if self.raise_on_failure:
            raise error
        logger.warning(
            "Writing a backup answer failed; returning the conversation without one. Error: {error}", error=error
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this hook to a dictionary.

        :returns: Serialized representation of the hook.
        """
        return default_to_dict(
            self,
            chat_generator=component_to_dict(obj=self.chat_generator, name="chat_generator"),
            system_prompt=self.system_prompt,
            exit_reasons=list(self.exit_reasons),
            raise_on_failure=self.raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BackupAnswerHook":
        """
        Create a hook from its serialized representation.

        :param data: Serialized hook data.
        :returns: The deserialized hook.
        """
        init_params = {**data.get("init_parameters", {})}
        deserialize_component_inplace(data=init_params, key="chat_generator")
        data = {**data, "init_parameters": init_params}
        return default_from_dict(cls, data=data)


def _reply_from_result(result: dict[str, Any]) -> ChatMessage:
    """
    Return the first reply of a chat generator result.

    :raises RuntimeError: If the result has no reply or the reply has no text.
    """
    replies = result.get("replies") or []
    if not replies or not (replies[0].text or "").strip():
        raise RuntimeError(f"The Chat Generator returned no usable text for the backup answer. Output: {result}.")
    return replies[0]
