# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import logging
from haystack.components.agents.state.state import State
from haystack.components.agents.state.state_utils import replace_values
from haystack.core.serialization import component_to_dict, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction.types import Compactor
from haystack.utils.deserialization import deserialize_component_inplace

logger = logging.getLogger(__name__)


class ContextCompactionHook:
    """
    Compacts an Agent's conversation once it grows past a threshold, so a long run does not exhaust the context window.

    This `before_llm` Agent hook compares the `context_tokens` state key against `threshold_tokens` before each
    chat-generator call and, when it is over, hands the conversation to a `Compactor` to rewrite. Register it on an
    `Agent` under the `before_llm` hook point:

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.hooks.compaction import ContextCompactionHook, SlidingWindowCompactor

    hook = ContextCompactionHook(
        compactor=SlidingWindowCompactor(keep_last_n_messages=20), threshold_tokens=100_000
    )
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-5.4-nano"),
        tools=[web_search],
        hooks={"before_llm": [hook]},
        max_agent_steps=50,
    )
    ```

    **The Agent's Chat Generator must report token usage.** `context_tokens` is refreshed after each call from the
    reply's `meta["usage"]`, and stays at `0` for a generator that does not report it - in which case the threshold is
    never reached and the hook never compacts. The hook logs a warning once per run when it detects this. Most Chat
    Generators report usage; a custom or mock one may not.

    Set the threshold well below the model's context window: it is checked before the call rather than after, so the
    reply and the tool results it triggers are added on top of what was measured.

    Compaction is lossy by nature, so the Agent works from a shorter record of the run afterwards. What survives is up
    to the compactor.
    """

    allowed_hook_points = ("before_llm",)

    def __init__(self, compactor: Compactor, *, threshold_tokens: int) -> None:
        """
        Initialize the hook with a compactor and the context size that triggers it.

        :param compactor: The `Compactor` that rewrites the conversation, for example a `SlidingWindowCompactor`.
        :param threshold_tokens: Compact once the `context_tokens` state key reaches this value. The Agent refreshes
            that key after each chat-generator call from the reply's reported token usage, so the Agent's Chat
            Generator must report usage for this to ever fire.
        :raises ValueError: If `threshold_tokens` is less than 1, which would compact on every step.
        """
        if threshold_tokens < 1:
            raise ValueError(
                f"`threshold_tokens` must be at least 1, got {threshold_tokens}. A threshold of 0 is reached before "
                f"the Agent has done anything, so every step would attempt compaction."
            )
        self.compactor = compactor
        self.threshold_tokens = threshold_tokens

    def run(self, state: State) -> None:
        """
        Compact `state.data["messages"]` if the context is over the threshold.

        :param state: The Agent's live `State`. Read to decide whether to compact, and rewritten in place when the
            compactor returns a compacted conversation.
        :returns: None. The hook mutates `state` in place.
        """
        if not self._over_threshold(state):
            return
        self._apply(state, self.compactor.compact(state))

    async def run_async(self, state: State) -> None:
        """
        Asynchronously compact `state.data["messages"]` if the context is over the threshold.

        :param state: The Agent's live `State`. Read to decide whether to compact, and rewritten in place when the
            compactor returns a compacted conversation.
        :returns: None. The hook mutates `state` in place.
        """
        if not self._over_threshold(state):
            return
        self._apply(state, await self.compactor.compact_async(state))

    def _over_threshold(self, state: State) -> bool:
        """
        Return whether the context has reached `threshold_tokens`.

        :param state: The Agent's `State`.
        :returns: True when compaction should be attempted.
        """
        context_tokens = state.data.get("context_tokens", 0)
        if context_tokens == 0:
            self._warn_if_usage_is_never_reported(state)
            return False
        return context_tokens >= self.threshold_tokens

    def _warn_if_usage_is_never_reported(self, state: State) -> None:
        """
        Warn when `context_tokens` is still unset after the Agent's first chat-generator call.

        `context_tokens` is legitimately `0` before the first call. Still being `0` afterwards means the Chat Generator
        does not report token usage, so the threshold can never be reached and this hook will silently do nothing for
        the whole run.

        `step_count` equals 1 at exactly one `before_llm` call per run, so warning only then logs once without the hook
        having to remember anything - which keeps it safe to share across concurrent runs.

        :param state: The Agent's `State`.
        :returns: None.
        """
        if state.data.get("step_count") != 1:
            return
        logger.warning(
            "The Agent's `context_tokens` is still 0 after the first chat-generator call, which means the Chat "
            "Generator does not report token usage. `ContextCompactionHook` triggers on `context_tokens`, so it will "
            "never compact this run. Use a Chat Generator that reports usage in `meta['usage']`."
        )

    def _apply(self, state: State, compacted: list[ChatMessage] | None) -> None:
        """
        Write a compacted conversation back into `State`.

        A compactor returns None when it has nothing worth changing, so there is no second-guessing here: whatever it
        returns is applied.

        `context_tokens` is reset to `0`, its "not yet measured" value: the count it held describes the conversation
        that was just replaced, and the next chat-generator call refreshes it from real usage. Leaving the old value in
        place would re-trigger compaction on the next step.

        The cumulative run metadata (`token_usage`, `tool_call_counts`) is deliberately left alone - it records what
        the run has spent and done, which compaction does not change.

        :param state: The Agent's live `State`.
        :param compacted: The compactor's result, or None when it had nothing to change.
        :returns: None.
        """
        if compacted is None:
            return
        messages_before = len(state.data.get("messages") or [])
        state.set("messages", compacted, handler_override=replace_values)
        state.set("context_tokens", 0)
        logger.debug(
            "Compacted the Agent's conversation from {before} to {after} messages.",
            before=messages_before,
            after=len(compacted),
        )

    def warm_up(self) -> None:
        """Warm up the compactor, which may hold resources such as a Chat Generator."""
        if hasattr(self.compactor, "warm_up"):
            self.compactor.warm_up()

    async def warm_up_async(self) -> None:
        """Warm up the compactor on the serving event loop."""
        warm_up_async = getattr(self.compactor, "warm_up_async", None)
        if warm_up_async is not None:
            await warm_up_async()
        elif hasattr(self.compactor, "warm_up"):
            self.compactor.warm_up()

    def close(self) -> None:
        """Release the compactor's resources."""
        if hasattr(self.compactor, "close"):
            self.compactor.close()

    async def close_async(self) -> None:
        """Release the compactor's async resources."""
        close_async = getattr(self.compactor, "close_async", None)
        if close_async is not None:
            await close_async()
        elif hasattr(self.compactor, "close"):
            self.compactor.close()

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the hook, including its compactor.

        :returns: A dictionary representation of the hook.
        """
        return default_to_dict(
            self,
            compactor=component_to_dict(obj=self.compactor, name="compactor"),
            threshold_tokens=self.threshold_tokens,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextCompactionHook":
        """
        Deserialize the hook, reconstructing its compactor.

        :param data: A dictionary representation produced by `to_dict`.
        :returns: The deserialized `ContextCompactionHook`.
        """
        init_params = data.get("init_parameters", {})
        if init_params.get("compactor") is not None:
            deserialize_component_inplace(init_params, key="compactor")
        return default_from_dict(cls, data)
