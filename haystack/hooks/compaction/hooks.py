# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import logging, tracing
from haystack.components.agents.state.state import State
from haystack.components.agents.state.state_utils import replace_values
from haystack.core.serialization import (
    component_to_dict,
    default_from_dict,
    default_to_dict,
    generate_qualified_class_name,
)
from haystack.dataclasses import ChatMessage
from haystack.hooks.compaction.types import Compactor
from haystack.hooks.compaction.utils import _estimated_context_tokens, _last_assistant_index
from haystack.token_counters import ApproximateTokenCounter, TokenCounter
from haystack.tools import ToolsType
from haystack.utils.deserialization import deserialize_component_inplace
from haystack.utils.experimental import _experimental

logger = logging.getLogger(__name__)


@_experimental
class CompactionHook:
    """
    Compacts an Agent's conversation once it fills too much of the model's context window.

    This `before_llm` Agent hook estimates the size of the conversation before each chat-generator call and, once it
    reaches `compact_at` of the window, hands it to a `Compactor` to bring back down to `compact_to`. Register it on an
    `Agent` under the `before_llm` hook point:

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIResponsesChatGenerator
    from haystack.hooks.compaction import CompactionHook, SlidingWindowCompactor

    hook = CompactionHook(
        compactor=SlidingWindowCompactor(),
        context_window=400_000,
        compact_at=0.7,
        compact_to=0.4,
    )
    agent = Agent(
        chat_generator=OpenAIResponsesChatGenerator(model="gpt-5.4-nano"),
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
            Defaults to `ApproximateTokenCounter`, which needs no extra dependency.
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
        self.token_counter = token_counter or ApproximateTokenCounter()

    def run(self, state: State) -> None:
        """
        Compact `state.data["messages"]` if the conversation fills too much of the window.

        :param state: The Agent's live `State`. Read to decide whether to compact, and rewritten in place when the
            compactor returns a compacted conversation.
        :returns: None. The hook mutates `state` in place.
        """
        messages = state.data.get("messages", [])
        context_tokens = state.data.get("context_tokens", 0)
        target = self._target_tokens(messages=messages, context_tokens=context_tokens, tools=state.data.get("tools"))
        if target is None:
            return
        target_tokens, estimated_overhead = target
        compacted = self.compactor.compact(
            messages=messages, target_tokens=target_tokens, token_counter=self.token_counter
        )
        self._apply(
            state=state,
            compacted=compacted,
            before=len(messages),
            target_tokens=target_tokens,
            original_context_tokens=context_tokens,
            estimated_overhead=estimated_overhead,
        )

    async def run_async(self, state: State) -> None:
        """
        Asynchronously compact `state.data["messages"]` if the conversation fills too much of the window.

        :param state: The Agent's live `State`. Read to decide whether to compact, and rewritten in place when the
            compactor returns a compacted conversation.
        :returns: None. The hook mutates `state` in place.
        """
        messages = state.data.get("messages", [])
        context_tokens = state.data.get("context_tokens", 0)
        target = self._target_tokens(messages=messages, context_tokens=context_tokens, tools=state.data.get("tools"))
        if target is None:
            return
        target_tokens, estimated_overhead = target
        compacted = await self.compactor.compact_async(
            messages=messages, target_tokens=target_tokens, token_counter=self.token_counter
        )
        self._apply(
            state=state,
            compacted=compacted,
            before=len(messages),
            target_tokens=target_tokens,
            original_context_tokens=context_tokens,
            estimated_overhead=estimated_overhead,
        )

    def _target_tokens(
        self, messages: list[ChatMessage], context_tokens: int, tools: ToolsType | None = None
    ) -> tuple[int, int] | None:
        """
        The size a compactor should bring `messages` down to.

        :param messages: The conversation, oldest to newest.
        :param context_tokens: The `context_tokens` state key. It is the current token count of the conversation except
            for the most recent tool result messages.
        :param tools: Tools whose schemas are sent alongside the messages.
        :returns: The target token amount the messages should be compacted to and the estimated non-message overhead,
            or None when the conversation is not yet large enough to compact.
        """
        estimated = _estimated_context_tokens(
            messages=messages, context_tokens=context_tokens, token_counter=self.token_counter, tools=tools
        )
        triggered = estimated >= self.context_window * self.compact_at

        span = tracing.tracer.current_span()
        if span is not None:
            span.set_tags(
                tags={
                    "haystack.agent.hook.compaction.strategy": generate_qualified_class_name(cls=type(self.compactor)),
                    "haystack.agent.hook.compaction.estimated_context_tokens": estimated,
                    "haystack.agent.hook.compaction.triggered": triggered,
                }
            )

        # The conversation is not yet large enough to compact, so leave it alone.
        if not triggered:
            return None

        # Calculate the non-message overhead, such as tool schemas and the provider's chat-template overhead.
        message_tokens = self.token_counter.count(messages=messages)
        overhead = estimated - message_tokens
        if overhead < 0:
            logger.warning(
                "The TokenCounter estimated more tokens for the messages ({message_tokens}) than the estimated total "
                "context size ({estimated}). It may be overestimating; consider configuring a more accurate "
                "TokenCounter.",
                message_tokens=message_tokens,
                estimated=estimated,
            )
        target_tokens = max(int(self.context_window * self.compact_to) - overhead, 0)

        if span is not None:
            span.set_tag(key="haystack.agent.hook.compaction.target_tokens", value=target_tokens)

        # Return the target token amount the messages should be compacted to and the overhead needed to re-estimate the
        # context size after compaction.
        return target_tokens, overhead

    def _apply(
        self,
        state: State,
        compacted: list[ChatMessage] | None,
        before: int,
        target_tokens: int,
        original_context_tokens: int,
        estimated_overhead: int,
    ) -> None:
        """
        Write a compacted conversation back into `State`, or do nothing if the compactor declined.

        :param state: The Agent's live `State`, rewritten in place.
        :param compacted: What the compactor returned.
        :param before: How many messages the conversation held before compaction, for the log line.
        :param target_tokens: The target the compactor was given, for the log line.
        :param original_context_tokens: The provider-reported context count before compaction, or 0 when unavailable.
        :param estimated_overhead: The estimated non-message overhead to include in the updated context count.
        """
        span = tracing.tracer.current_span()
        if span is not None:
            span.set_tag(key="haystack.agent.hook.compaction.compacted", value=compacted is not None)

        # If the compactor returned None, it declined to compact. Leave the conversation alone.
        if compacted is None:
            return

        state.set("messages", compacted, handler_override=replace_values)
        # If the original value was 0, leave it unchanged so later steps keep recounting the full request locally.
        if original_context_tokens != 0:
            # Re-estimate the provider-accounted context through the last assistant message, including overhead. If we
            # added the trailing tool results, a second registered hook could double count them.
            state.set(
                "context_tokens",
                self.token_counter.count(messages=compacted[: _last_assistant_index(messages=compacted) + 1])
                + estimated_overhead,
            )
        logger.debug(
            "Compacted the Agent's conversation at step {step} from {before} to {after} messages, targeting {target} "
            "tokens.",
            step=state.data.get("step_count", 0),
            before=before,
            after=len(compacted),
            target=target_tokens,
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
    def from_dict(cls, data: dict[str, Any]) -> "CompactionHook":
        """
        Deserialize the hook, reconstructing its compactor and token counter.

        :param data: A dictionary representation produced by `to_dict`.
        :returns: The deserialized `CompactionHook`.
        """
        init_params = data.get("init_parameters", {})
        for key in ("compactor", "token_counter"):
            if init_params.get(key) is not None:
                deserialize_component_inplace(data=init_params, key=key)
        return default_from_dict(cls, data=data)
