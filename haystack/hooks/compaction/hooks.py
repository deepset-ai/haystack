# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import logging
from haystack.components.agents.state.state import State
from haystack.components.agents.state.state_utils import replace_values
from haystack.core.serialization import component_to_dict, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction.types import CompactionBudget, Compactor
from haystack.hooks.compaction.utils import _estimated_context_tokens, _last_assistant_end
from haystack.token_counters import TiktokenCounter, TokenCounter
from haystack.utils.deserialization import deserialize_component_inplace
from haystack.utils.experimental import _experimental

logger = logging.getLogger(__name__)


@_experimental
class ContextCompactionHook:
    """
    Compacts an Agent's conversation once it fills too much of the model's context window.

    This `before_llm` Agent hook estimates the size of the conversation before each chat-generator call and, once it
    reaches `compact_at` of the window, hands it to a `Compactor` to bring back down to `compact_to`. Register it on an
    `Agent` under the `before_llm` hook point:

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.hooks.compaction import ContextCompactionHook, SlidingWindowCompactor

    hook = ContextCompactionHook(
        compactor=SlidingWindowCompactor(),
        context_window=200_000,
        compact_at=0.7,
        compact_to=0.4,
    )
    agent = Agent(
        chat_generator=OpenAIChatGenerator(model="gpt-5.4-nano"),
        tools=[web_search],
        hooks={"before_llm": [hook]},
        max_agent_steps=50,
    )
    ```

    Size is measured by anchoring on the `context_tokens` state key - the chat generator's own count of the request it
    was sent plus its reply, which already covers the system prompt, the tool schemas, and the provider's chat-template
    overhead - and counting only the messages appended since that call. The estimate is therefore exact for the bulk of
    the conversation and approximate only for its most recent messages.

    Compaction is lossy by nature, so the Agent works from a shorter record of the run afterwards. What survives is up
    to the compactor.
    """

    allowed_hook_points = ("before_llm",)

    def __init__(
        self,
        compactor: Compactor,
        *,
        context_window: int,
        compact_at: float = 0.7,
        compact_to: float = 0.4,
        token_counter: TokenCounter | None = None,
    ) -> None:
        """
        Initialize the hook with a compactor and the window it has to fit in.

        :param compactor: The `Compactor` that rewrites the conversation.
        :param context_window: The model's context window in tokens. Everything else is a fraction of this, so moving to
            a different model means changing only this number.
        :param compact_at: The fraction of the window at which compaction starts. Leave room above it for the reply and
            the tool results it triggers, which land on top of what was measured.
        :param compact_to: The fraction of the window compaction aims to bring the conversation down to. Lower means
            compacting less often but losing more each time.
        :param token_counter: The `TokenCounter` used to size the messages the chat generator has not reported on yet.
            Defaults to `TiktokenCounter`, which needs `tiktoken` installed.
        :raises ValueError: If `context_window` is not positive, or the fractions are not
            `0 < compact_to < compact_at <= 1`.
        """
        if context_window < 1:
            raise ValueError(f"`context_window` must be a positive number of tokens, got {context_window}.")
        if not 0 < compact_to < compact_at <= 1:
            raise ValueError(
                f"`compact_at` and `compact_to` must satisfy 0 < compact_to < compact_at <= 1, got "
                f"compact_at={compact_at} and compact_to={compact_to}. A target at or above the trigger would leave "
                f"the conversation over the trigger after compacting, so it would be attempted again every step."
            )
        self.compactor = compactor
        self.context_window = context_window
        self.compact_at = compact_at
        self.compact_to = compact_to
        self.token_counter = token_counter or TiktokenCounter()

    @property
    def _target_tokens(self) -> int:
        """The size compaction aims to bring the conversation down to."""
        return int(self.context_window * self.compact_to)

    def run(self, state: State) -> None:
        """
        Compact `state.data["messages"]` if the conversation fills too much of the window.

        :param state: The Agent's live `State`. Read to decide whether to compact, and rewritten in place when the
            compactor returns a compacted conversation.
        :returns: None. The hook mutates `state` in place.
        """
        if not self._over_threshold(state):
            return
        self._apply(state, self.compactor.compact(self._messages(state), self._budget(state)))

    async def run_async(self, state: State) -> None:
        """
        Asynchronously compact `state.data["messages"]` if the conversation fills too much of the window.

        :param state: The Agent's live `State`. Read to decide whether to compact, and rewritten in place when the
            compactor returns a compacted conversation.
        :returns: None. The hook mutates `state` in place.
        """
        if not self._over_threshold(state):
            return
        self._apply(state, await self.compactor.compact_async(self._messages(state), self._budget(state)))

    @staticmethod
    def _messages(state: State) -> list[ChatMessage]:
        """The conversation held in `State`, or an empty list."""
        return state.data.get("messages") or []

    def _estimated_tokens(self, state: State) -> int:
        """The estimated size of the whole context, anchored on what the chat generator reported."""
        return _estimated_context_tokens(self._messages(state), state.data.get("context_tokens", 0), self.token_counter)

    def _budget(self, state: State) -> CompactionBudget:
        """The size the messages should come in under, and the counter to measure with."""
        # The target covers the whole context, but a compactor can only remove messages. Subtract what it cannot touch -
        # the tool schemas and chat-template overhead the reported count includes - or it would remove far too little.
        overhead = self._estimated_tokens(state) - self.token_counter.count(self._messages(state))
        return CompactionBudget(target_tokens=max(self._target_tokens - overhead, 0), counter=self.token_counter)

    def _over_threshold(self, state: State) -> bool:
        """Whether the context has reached `compact_at` of the window, so compaction should be attempted."""
        return self._estimated_tokens(state) >= self.context_window * self.compact_at

    def _apply(self, state: State, compacted: list[ChatMessage] | None) -> None:
        """Write a compacted conversation back into `State`, or do nothing if the compactor declined."""
        if compacted is None:
            return
        messages_before = len(self._messages(state))
        state.set("messages", compacted, handler_override=replace_values)
        # Re-estimate rather than reset to 0, which would claim the context is empty when its size is roughly known.
        # Counting only through the last assistant message keeps the key's meaning intact, so a later read does not
        # count the trailing messages a second time.
        state.set("context_tokens", self.token_counter.count(compacted[: _last_assistant_end(compacted)]))
        logger.debug(
            "Compacted the Agent's conversation at step {step} from {before} to {after} messages, targeting {target} "
            "tokens.",
            step=state.data.get("step_count", 0),
            before=messages_before,
            after=len(compacted),
            target=self._target_tokens,
        )

    def warm_up(self) -> None:
        """Warm up the token counter and the compactor, which may hold resources such as a Chat Generator."""
        if hasattr(self.token_counter, "warm_up"):
            self.token_counter.warm_up()
        if hasattr(self.compactor, "warm_up"):
            self.compactor.warm_up()

    async def warm_up_async(self) -> None:
        """Warm up the token counter and the compactor on the serving event loop."""
        if hasattr(self.token_counter, "warm_up"):
            self.token_counter.warm_up()
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
        Serialize the hook, including its compactor and token counter.

        :returns: A dictionary representation of the hook.
        """
        return default_to_dict(
            self,
            compactor=component_to_dict(obj=self.compactor, name="compactor"),
            context_window=self.context_window,
            compact_at=self.compact_at,
            compact_to=self.compact_to,
            token_counter=component_to_dict(obj=self.token_counter, name="token_counter"),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContextCompactionHook":
        """
        Deserialize the hook, reconstructing its compactor and token counter.

        :param data: A dictionary representation produced by `to_dict`.
        :returns: The deserialized `ContextCompactionHook`.
        """
        init_params = data.get("init_parameters", {})
        for key in ("compactor", "token_counter"):
            if init_params.get(key) is not None:
                deserialize_component_inplace(init_params, key=key)
        return default_from_dict(cls, data)
