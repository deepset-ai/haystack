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
from haystack.hooks.compaction.utils import _conversation_chars
from haystack.utils.deserialization import deserialize_component_inplace

logger = logging.getLogger(__name__)


class ContextCompactionHook:
    """
    Compacts an Agent's conversation once it grows past a threshold, so a long run does not exhaust the context window.

    This `before_llm` Agent hook checks the size of the conversation before each chat-generator call and, when it is
    over the threshold, hands it to a `Compactor` to rewrite. Register it on an `Agent` under the `before_llm` hook
    point:

    <!-- test-concept -->
    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.hooks.compaction import ContextCompactionHook, SlidingWindowCompactor

    hook = ContextCompactionHook(
        compactor=SlidingWindowCompactor(keep_last_n_messages=20),
        threshold_tokens=100_000,
        threshold_chars=400_000,
    )
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-5.4-nano"),
        tools=[web_search],
        hooks={"before_llm": [hook]},
        max_agent_steps=50,
    )
    ```

    Set the threshold well below the model's context window: it is checked before the call rather than after, and the
    reply plus the tool results it triggers are added on top of what was measured.

    Compaction is lossy by nature, so the Agent works from a shorter record of the run afterwards. What survives is up
    to the compactor.
    """

    allowed_hook_points = ("before_llm",)

    def __init__(
        self, compactor: Compactor, *, threshold_tokens: int | None = None, threshold_chars: int | None = None
    ) -> None:
        """
        Initialize the hook with a compactor and the thresholds that trigger it.

        At least one threshold must be set. Setting both is the most robust configuration: they measure different
        things and are checked independently, so whichever crosses first triggers compaction.

        :param compactor: The `Compactor` that rewrites the conversation, for example a `SummarizationCompactor`.
        :param threshold_tokens: Compact once the `context_tokens` state key reaches this value. That key is refreshed
            after each chat-generator call from the reply's reported token usage, so it is the more accurate trigger -
            but it stays at `0` for Chat Generators that do not report usage, in which case this threshold never fires.
            Pair it with `threshold_chars` unless you know your generator reports usage.
        :param threshold_chars: Compact once the conversation reaches this many characters. Counted from the messages
            themselves, so it works with any Chat Generator and accounts for content added since the last call. As a
            rough guide one token is about four characters.
        :raises ValueError: If neither threshold is set, since the hook would then never compact.
        """
        if threshold_tokens is None and threshold_chars is None:
            raise ValueError(
                "`ContextCompactionHook` requires at least one of `threshold_tokens` or `threshold_chars` to be set, "
                "otherwise it would never compact."
            )
        self.compactor = compactor
        self.threshold_tokens = threshold_tokens
        self.threshold_chars = threshold_chars

    def run(self, state: State) -> None:
        """
        Compact `state.data["messages"]` if the conversation is over the threshold.

        :param state: The Agent's live `State`. Read to decide whether to compact, and rewritten in place when the
            compactor returns a shorter conversation.
        :returns: None. The hook mutates `state` in place.
        """
        if not self._over_threshold(state):
            return
        self._apply(state, self.compactor.compact(state))

    async def run_async(self, state: State) -> None:
        """
        Asynchronously compact `state.data["messages"]` if the conversation is over the threshold.

        :param state: The Agent's live `State`. Read to decide whether to compact, and rewritten in place when the
            compactor returns a shorter conversation.
        :returns: None. The hook mutates `state` in place.
        """
        if not self._over_threshold(state):
            return
        self._apply(state, await self.compactor.compact_async(state))

    def _over_threshold(self, state: State) -> bool:
        """
        Return whether the conversation has reached either configured threshold.

        :param state: The Agent's `State`.
        :returns: True when compaction should be attempted.
        """
        if self.threshold_tokens is not None and state.data.get("context_tokens", 0) >= self.threshold_tokens:
            return True
        if self.threshold_chars is not None:
            return _conversation_chars(state.data.get("messages") or []) >= self.threshold_chars
        return False

    def _apply(self, state: State, compacted: list[ChatMessage] | None) -> None:
        """
        Write a compacted conversation back into `State`, if the compactor actually shrank it.

        `context_tokens` is reset to `0`, its "not yet measured" value: the count it held describes the conversation
        that was just replaced, and the next chat-generator call refreshes it from real usage. Leaving the old value in
        place would re-trigger compaction on the next step whenever that call reports no usage.

        The cumulative run metadata (`token_usage`, `tool_call_counts`) is deliberately left alone - it records what
        the run has spent and done, which compaction does not change.

        :param state: The Agent's live `State`.
        :param compacted: The compactor's result, or None when it had nothing to change.
        :returns: None.
        """
        if compacted is None:
            return
        messages = state.data.get("messages") or []
        size_before = _conversation_chars(messages)
        size_after = _conversation_chars(compacted)
        if size_after >= size_before:
            return

        state.set("messages", compacted, handler_override=replace_values)
        state.set("context_tokens", 0)
        logger.debug(
            "Compacted the Agent's conversation from {before} to {after} characters "
            "({messages_before} to {messages_after} messages).",
            before=size_before,
            after=size_after,
            messages_before=len(messages),
            messages_after=len(compacted),
        )

        # Only `threshold_chars` can be re-checked here: `context_tokens` was just reset and is not measured again
        # until the next chat-generator call, so a token-only configuration cannot detect this.
        if self.threshold_chars is not None and size_after >= self.threshold_chars:
            logger.warning(
                "The Agent's conversation is still {after} characters after compaction, at or above the "
                "`threshold_chars` of {threshold}. The threshold is likely below the size the compactor can reach - "
                "for example, smaller than the recent messages it keeps verbatim - so compaction will keep being "
                "attempted without ever getting under it. Raise `threshold_chars` or configure the compactor to "
                "retain less.",
                after=size_after,
                threshold=self.threshold_chars,
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
            threshold_chars=self.threshold_chars,
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
