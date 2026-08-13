# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import logging
from haystack.components.generators.chat.types import ChatGenerator
from haystack.components.generators.chat.utils import _convert_haystack_generation_kwargs
from haystack.core.serialization import component_to_dict, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, FileContent, ImageContent
from haystack.dataclasses.chat_message import ChatMessageContentT
from haystack.hooks.compaction.types import Compactor
from haystack.hooks.compaction.utils import (
    _COMPACTION_META_KEY,
    _current_agent_step_groups,
    _historical_turn_groups,
    _is_compaction_message,
    _latest_user_index,
    _leading_system_end,
    _messages_at,
    _messages_except,
)
from haystack.token_counters import TokenCounter
from haystack.token_counters.utils import _rendered_conversation
from haystack.utils.async_utils import _execute_component_async
from haystack.utils.deserialization import deserialize_component_inplace
from haystack.utils.experimental import _experimental

logger = logging.getLogger(__name__)

# Recorded as the strategy on every summary this compactor produces, so a later run can recognize its own summaries.
_STRATEGY = "summarization"

# Recorded as the `source` on a summary, naming the stretch of conversation it stands in for. Compaction gives these up
# in order, so the Agent's current task is the last thing to go.
_HISTORICAL_TURNS = "historical_turns"
_HISTORICAL_SUMMARIES = "historical_summaries"
_CURRENT_TASK_SUMMARIES = "current_task_summaries"
_CURRENT_TASK_STEPS = "current_task_steps"

_DEFAULT_SUMMARY_INSTRUCTION = """You are compacting one portion of a conversation between a user and an AI agent so \
the agent can keep working with fewer tokens. You are shown only the portion being replaced. The rest of the \
conversation, including the user's current request, stays in place and is not shown to you. Summarize only what you \
are given, and never say or imply that something did not happen just because it is absent from this portion.

Use these sections, in this order. Keep every section, and write "(none)" when this portion says nothing about it.

## Objective
What the user was trying to accomplish, if this portion shows it.

## Decisions and constraints
Choices made and the reasoning behind them, and any requirements, preferences, or instructions the user gave. Note \
options that were rejected and why.

## Work completed
What was done, and what the tool results established.

## Identifiers
Exact file paths, URLs, IDs, names, commands, and error strings, copied character for character. Images and files \
appear only as <image: ...> and <file: ...> placeholders; their contents are not available to you and are lost once \
this portion is replaced, so copy the placeholder details here.

## Unresolved
Work still outstanding, and the immediate next step.

Rules:
- Record only what this portion shows. Do not infer, do not give advice, and do not add anything that is not here.
- Copy identifiers exactly rather than describing them. They cannot be recovered once this portion is gone.
- Fold any <conversation_summary> blocks you are given into your own: keep what is still true, drop what is now \
stale, and merge in the new facts.
- Use terse bullets. Do not address the user, and do not mention that you are summarizing."""


def _identifying_details(metadata: dict[str, Any]) -> list[str]:
    """Render an attachment's metadata as `key=value` pairs, such as the path a file was loaded from."""
    # For ease, we don't support nested keys, we are mostly interested in the top-level keys that identify the
    # attachment, such as a file path or URL.
    return [f"{key}={value}" for key, value in sorted(metadata.items()) if isinstance(value, (str, int, float, bool))]


def _attachment_placeholder(content: ChatMessageContentT) -> str:
    """
    Render a placeholder for an attachment that cannot survive summarization, so the summary can preserve its identity.
    """
    if isinstance(content, ImageContent):
        # Images have no filename, so whatever identifies one lives in its `meta`.
        return f"<image: {', '.join([content.mime_type or 'unknown type', *_identifying_details(content.meta)])}>"
    if isinstance(content, FileContent):
        details = [content.filename or "unnamed", content.mime_type or "unknown type"]
        return f"<file: {', '.join([*details, *_identifying_details(content.extra)])}>"
    return f"<{type(content).__name__}>"


def _is_summary(message: ChatMessage) -> bool:
    """Whether a message is a summary this strategy wrote."""
    return _is_compaction_message(message=message, strategy=_STRATEGY)


def _previous_summary_indices(messages: list[ChatMessage], start: int, end: int) -> list[int]:
    """Return the positions of the summaries an earlier compaction left in a bounded part of a conversation."""
    return [index for index in range(start, end) if _is_summary(message=messages[index])]


def _raw_historical_turn_groups(
    messages: list[ChatMessage], system_end: int, task_index: int | None
) -> list[list[int]]:
    """
    Return the historical turns that still hold raw, never-summarized conversation, oldest turn first.

    Summaries an earlier compaction wrote are excluded, so summarizing a turn leaves them in place for
    `_HISTORICAL_SUMMARIES` to fold later. The list is empty when there are no historical turns, or when every one of
    them is already nothing but summaries.
    """
    # Strip the previous summaries out of each turn, then drop the turns that strip away to nothing.
    groups = [
        [index for index in group if not _is_summary(message=messages[index])]
        for group in _historical_turn_groups(messages=messages, system_end=system_end, task_index=task_index)
    ]
    return [group for group in groups if group]


def _groups_to_summarize(
    messages: list[ChatMessage],
    groups: list[list[int]],
    target_tokens: int,
    summary_tokens: int,
    token_counter: TokenCounter,
) -> list[int]:
    """
    Return the fewest oldest groups whose removal makes room for a summary of the expected size.

    Groups are taken oldest first and counting stops as soon as what remains, plus the summary that replaces them,
    fits the target. When even taking all of them is not enough, all of them are returned.
    """
    selected: list[int] = []
    for group in groups:
        selected.extend(group)
        remaining = token_counter.count(messages=_messages_except(messages=messages, indices=selected))
        if remaining + summary_tokens <= target_tokens:
            break
    return selected


def _summary_message(text: str, summarized_messages: int, source: str) -> ChatMessage:
    """Build the marked user message that stands in for the messages the summary replaced."""
    body = f"<conversation_summary>\n{text.strip()}\n</conversation_summary>"
    meta = {_COMPACTION_META_KEY: {"strategy": _STRATEGY, "summarized_messages": summarized_messages, "source": source}}
    return ChatMessage.from_user(text=body, meta=meta)


def _replace_indices(messages: list[ChatMessage], indices: list[int], summary: ChatMessage) -> list[ChatMessage]:
    """Replace the selected messages, which need not be contiguous, with one summary at the oldest one's position."""
    selected = set(indices)
    # The summary stands in for everything it replaced, so it takes the position of the oldest message it covers.
    insertion_index = min(indices)
    compacted: list[ChatMessage] = []
    for index, message in enumerate(messages):
        # Emit the summary before the message it displaces, so the surrounding conversation keeps its order.
        if index == insertion_index:
            compacted.append(summary)
        if index not in selected:
            compacted.append(message)
    return compacted


@_experimental
class SummarizationCompactor(Compactor):
    """
    Condenses old historical turns first, then old steps from the Agent's current task.

    Leading system messages and the latest real user message are always kept. Historical turns are summarized in full,
    oldest first. Summaries normally accumulate so they are not repeatedly rewritten; if every historical turn has
    already been summarized and more space is needed, those historical summaries are folded into one before any
    current-task steps are summarized. An assistant message and all immediately following tool results form one step,
    so tool calls are never separated from their results.

    Each summary is requested within `max_summary_tokens`. A Chat Generator that supports an output-token limit is
    held to it at runtime, whatever its provider calls that setting. Any other generator receives the limit as prompt
    guidance, and the summary it returns is measured before it is accepted either way.

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIResponsesChatGenerator
    from haystack.hooks.compaction import CompactionHook, SummarizationCompactor

    summary_generator = OpenAIResponsesChatGenerator(model="gpt-5.4-nano")
    hook = CompactionHook(
        compactor=SummarizationCompactor(chat_generator=summary_generator),
        context_window=400_000,
        compact_at=0.7,
        compact_to=0.4,
    )
    agent = Agent(chat_generator=agent_generator, tools=[web_search], hooks={"before_llm": [hook]})
    ```
    """

    def __init__(
        self,
        chat_generator: ChatGenerator,
        *,
        min_keep_steps: int = 1,
        max_summary_tokens: int = 1024,
        summary_instruction: str = _DEFAULT_SUMMARY_INSTRUCTION,
        raise_on_failure: bool = False,
    ) -> None:
        """
        Initialize the compactor.

        :param chat_generator: The Chat Generator used to write summaries.
        :param min_keep_steps: The fewest complete recent Agent steps to keep, even when they exceed the target.
        :param max_summary_tokens: The output-token budget reserved for each summary. A Chat Generator that supports an
            output-token limit is sent this one at runtime, overriding any limit configured on the generator itself.
        :param summary_instruction: What the model is told to preserve when it writes a summary. The default asks for
            fixed sections covering the objective, decisions and constraints, completed work, exact identifiers, and
            unresolved work, each written as `(none)` when the summarized portion says nothing about it. It also states
            that only part of the conversation is shown, so the model does not conclude that something never happened
            just because it is absent. The token budget is appended to whatever is given here, so a replacement does
            not need to mention it.
        :param raise_on_failure: Whether a failed or non-shrinking summarization raises. By default the failure is
            logged and any successful partial compaction is returned.
        :raises ValueError: If `min_keep_steps` is negative or `max_summary_tokens` is not positive.
        """
        if min_keep_steps < 0:
            raise ValueError(f"`min_keep_steps` must be at least 0, got {min_keep_steps}.")
        if max_summary_tokens < 1:
            raise ValueError(f"`max_summary_tokens` must be a positive number of tokens, got {max_summary_tokens}.")
        self.chat_generator = chat_generator
        self.min_keep_steps = min_keep_steps
        self.max_summary_tokens = max_summary_tokens
        self.summary_instruction = summary_instruction
        self.raise_on_failure = raise_on_failure

    def compact(
        self, messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter
    ) -> list[ChatMessage] | None:
        """
        Return a progressively summarized conversation, or None when no useful reduction is possible.

        :param messages: The conversation to compact, ordered oldest to newest.
        :param target_tokens: The token budget the compacted messages should aim to fit.
        :param token_counter: The counter used both to plan compaction and verify generated summaries.
        :returns: A smaller replacement conversation, or None when nothing was reduced.
        """
        run_kwargs = self._generation_run_kwargs()

        # Rebound only when a summary is applied, and never mutated, so `messages` is left as the caller passed it.
        compacted = messages
        summarized = False
        while True:
            # Ask which stretch of the conversation to give up next. None means the target is met or nothing is left.
            plan = self._next_summary(messages=compacted, target_tokens=target_tokens, token_counter=token_counter)
            if plan is None:
                break
            indices, source = plan
            prompt = self._prompt(messages=compacted, indices=indices)
            try:
                # Summarize that stretch and swap it in, so the next round plans against the smaller conversation.
                # A generator error or a summary that does not shrink raises out of here.
                result = self.chat_generator.run(messages=prompt, **run_kwargs)
                compacted = self._apply_summary(
                    messages=compacted, indices=indices, source=source, result=result, token_counter=token_counter
                )
                summarized = True
            except Exception as error:
                # Stop at the last summary that worked, unless `raise_on_failure` says to propagate.
                self._report_failure(error=error)
                break
        # Every applied summary was measured as shrinking the conversation, so any summary at all is real progress,
        # whether or not the target was met. Without one there is nothing to hand back.
        return compacted if summarized else None

    async def compact_async(
        self, messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter
    ) -> list[ChatMessage] | None:
        """
        Asynchronously return a progressively summarized conversation.

        :param messages: The conversation to compact, ordered oldest to newest.
        :param target_tokens: The token budget the compacted messages should aim to fit.
        :param token_counter: The counter used both to plan compaction and verify generated summaries.
        :returns: A smaller replacement conversation, or None when nothing was reduced.
        """
        run_kwargs = self._generation_run_kwargs()

        # Rebound only when a summary is applied, and never mutated, so `messages` is left as the caller passed it.
        compacted = messages
        summarized = False
        while True:
            # Ask which stretch of the conversation to give up next. None means the target is met or nothing is left.
            plan = self._next_summary(messages=compacted, target_tokens=target_tokens, token_counter=token_counter)
            if plan is None:
                break
            indices, source = plan
            prompt = self._prompt(messages=compacted, indices=indices)
            try:
                # Summarize that stretch and swap it in, so the next round plans against the smaller conversation.
                # Only the generator call is awaited; planning and swapping are pure.
                result = await _execute_component_async(
                    component_instance=self.chat_generator, messages=prompt, **run_kwargs
                )
                compacted = self._apply_summary(
                    messages=compacted, indices=indices, source=source, result=result, token_counter=token_counter
                )
                summarized = True
            except Exception as error:
                # Stop at the last summary that worked, unless `raise_on_failure` says to propagate.
                self._report_failure(error=error)
                break
        # Every applied summary was measured as shrinking the conversation, so any summary at all is real progress,
        # whether or not the target was met. Without one there is nothing to hand back.
        return compacted if summarized else None

    def _generation_run_kwargs(self) -> dict[str, Any]:
        """
        Return the `run` keyword arguments that hold a summary to `max_summary_tokens`.

        `max_output_tokens` is Haystack's provider-neutral name for an output-token limit, so the Chat Generator
        translates it into whatever its own provider calls it. A generator that does not advertise the parameter gets
        no runtime setting at all and is held to the limit by the prompt alone, since the `ChatGenerator` protocol
        guarantees nothing beyond `run`.
        """
        generation_kwargs = _convert_haystack_generation_kwargs(
            chat_generator=self.chat_generator,
            haystack_generation_kwargs={"max_output_tokens": self.max_summary_tokens},
        )
        return {"generation_kwargs": generation_kwargs} if generation_kwargs else {}

    def _next_summary(
        self, messages: list[ChatMessage], target_tokens: int, token_counter: TokenCounter
    ) -> tuple[list[int], str] | None:
        """
        Choose the next stretch of conversation to replace with a summary.

        Four tiers are tried in order, so the oldest and least useful context goes first and the Agent's current task
        is given up last:

        1. `_HISTORICAL_TURNS`: the fewest oldest raw turns that make room for a summary.
        2. `_HISTORICAL_SUMMARIES`: nothing raw is left in history, so fold its summaries into one.
        3. `_CURRENT_TASK_SUMMARIES`: fold the summaries earlier steps left behind before giving up more steps.
        4. `_CURRENT_TASK_STEPS`: the fewest oldest steps of the current task, keeping `min_keep_steps` of the newest.

        :param messages: The conversation as it stands, ordered oldest to newest.
        :param target_tokens: The token budget the conversation should come in under.
        :param token_counter: The counter used to measure candidate selections.
        :returns: The message indices to summarize and the `source` to record on the resulting summary, or None when
            the conversation already fits or nothing is left that may be given up.
        """
        # Nothing to give up once the conversation fits.
        if token_counter.count(messages=messages) <= target_tokens:
            return None

        # The landmarks everything is measured against: the Agent's instructions, and the user message anchoring the
        # current task. History runs from the instructions up to that anchor, the current task from the anchor on.
        system_end = _leading_system_end(messages=messages)
        task_index = _latest_user_index(messages=messages)
        history_end = task_index if task_index is not None else system_end
        task_start = task_index + 1 if task_index is not None else system_end

        # Tier 1. Raw history is the cheapest context to lose, so take the oldest turns that still hold any.
        historical_turns = _raw_historical_turn_groups(messages=messages, system_end=system_end, task_index=task_index)
        if historical_turns:
            oldest_turns = _groups_to_summarize(
                messages=messages,
                groups=historical_turns,
                target_tokens=target_tokens,
                summary_tokens=self.max_summary_tokens,
                token_counter=token_counter,
            )
            return oldest_turns, _HISTORICAL_TURNS

        # Tier 2. History is nothing but summaries now, so the only room left there is in folding them into one. They
        # are left to accumulate until this point so that they are not rewritten on every compaction.
        history_summaries = _previous_summary_indices(messages=messages, start=system_end, end=history_end)
        if len(history_summaries) > 1:
            return history_summaries, _HISTORICAL_SUMMARIES

        # History is exhausted, so the current task has to pay. Its `min_keep_steps` newest steps are off limits.
        agent_steps = _current_agent_step_groups(messages=messages, system_end=system_end, task_index=task_index)
        eligible_steps = agent_steps[: max(len(agent_steps) - self.min_keep_steps, 0)]
        if not eligible_steps:
            return None

        # Tier 3. Fold the summaries earlier steps left behind before spending another raw step on the same space.
        task_summaries = _previous_summary_indices(messages=messages, start=task_start, end=len(messages))
        if len(task_summaries) > 1:
            return task_summaries, _CURRENT_TASK_SUMMARIES

        # Tier 4. Last resort: give up the oldest steps of the task the Agent is working on right now.
        oldest_steps = _groups_to_summarize(
            messages=messages,
            groups=eligible_steps,
            target_tokens=target_tokens,
            summary_tokens=self.max_summary_tokens,
            token_counter=token_counter,
        )
        return oldest_steps, _CURRENT_TASK_STEPS

    def _prompt(self, messages: list[ChatMessage], indices: list[int]) -> list[ChatMessage]:
        """Build the bounded summarization instruction and the rendered transcript of the selected messages."""
        transcript = _rendered_conversation(
            _messages_at(messages=messages, indices=indices), placeholder=_attachment_placeholder
        )
        instruction = (
            f"{self.summary_instruction}\n\nWrite a complete summary in no more than approximately "
            f"{self.max_summary_tokens} tokens. Prioritize completeness within that limit so the response is not "
            "cut off."
        )
        return [
            ChatMessage.from_system(text=instruction),
            ChatMessage.from_user(text=f"<conversation_to_summarize>\n{transcript}\n</conversation_to_summarize>"),
        ]

    @staticmethod
    def _apply_summary(
        messages: list[ChatMessage],
        indices: list[int],
        source: str,
        result: dict[str, Any],
        token_counter: TokenCounter,
    ) -> list[ChatMessage]:
        """
        Swap the selected messages for the generated summary.

        :raises RuntimeError: If the generator returned no usable text, or if the swap did not make the conversation
            smaller, in which case keeping the raw messages is the better outcome.
        """
        replies = result.get("replies") or []
        text = replies[-1].text if replies else None
        if not text or not text.strip():
            raise RuntimeError("The Chat Generator returned no text to use as a conversation summary.")
        summary = _summary_message(text=text, summarized_messages=len(indices), source=source)
        compacted = _replace_indices(messages=messages, indices=indices, summary=summary)
        before = token_counter.count(messages=messages)
        after = token_counter.count(messages=compacted)
        if after >= before:
            raise RuntimeError(
                f"The generated summary did not reduce the conversation size ({before} tokens before and {after} "
                "tokens after)."
            )
        return compacted

    def _report_failure(self, error: Exception) -> None:
        """Re-raise a failed summarization or log it, so whatever compacted successfully so far is still returned."""
        if self.raise_on_failure:
            raise error
        logger.warning(
            "Summarizing the conversation for context compaction failed; keeping the last successful result. "
            "Error: {error}",
            error=error,
        )

    def warm_up(self) -> None:
        """Warm up the Chat Generator that writes summaries."""
        if hasattr(self.chat_generator, "warm_up"):
            self.chat_generator.warm_up()

    async def warm_up_async(self) -> None:
        """Warm up the Chat Generator on the serving event loop."""
        warm_up_async = getattr(self.chat_generator, "warm_up_async", None)
        if warm_up_async is not None:
            await warm_up_async()
        elif hasattr(self.chat_generator, "warm_up"):
            self.chat_generator.warm_up()

    def close(self) -> None:
        """Release the Chat Generator's resources."""
        if hasattr(self.chat_generator, "close"):
            self.chat_generator.close()

    async def close_async(self) -> None:
        """Release the Chat Generator's resources."""
        close_async = getattr(self.chat_generator, "close_async", None)
        if close_async is not None:
            await close_async()
        elif hasattr(self.chat_generator, "close"):
            self.chat_generator.close()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the compactor and its Chat Generator."""
        return default_to_dict(
            self,
            chat_generator=component_to_dict(obj=self.chat_generator, name="chat_generator"),
            min_keep_steps=self.min_keep_steps,
            max_summary_tokens=self.max_summary_tokens,
            summary_instruction=self.summary_instruction,
            raise_on_failure=self.raise_on_failure,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SummarizationCompactor":
        """Deserialize the compactor and reconstruct its Chat Generator."""
        init_params = data.get("init_parameters", {})
        if init_params.get("chat_generator") is not None:
            deserialize_component_inplace(data=init_params, key="chat_generator")
        return default_from_dict(cls=cls, data=data)
