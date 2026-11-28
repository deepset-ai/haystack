# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import logging
from haystack.components.agents.state.state import State
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage
from haystack.hooks.loop_detection.detector import LoopDetectionResult, detect_tool_loop
from haystack.hooks.utils import _SYSTEM_REMINDER_META_KEY
from haystack.utils.experimental import _experimental

logger = logging.getLogger(__name__)

_EXIT_REASON_TOOL_LOOP = "tool_loop_detected"
_LOOP_WARNING_META_KEY = "tool_loop_warning"

_FINAL_MESSAGE_TEMPLATE = "The Agent stopped because a tool-calling loop was detected ({detail})."

_WARNING_TEMPLATE = (
    "Loop detection warning: {detail}. "
    "Do not repeat the same failing or redundant tool calls; try a different approach or provide your final answer "
    "with the information you already have."
)


def _warning_key(result: LoopDetectionResult) -> str:
    """Build a deterministic deduplication key for a loop warning reminder."""
    return f"{result.kind}:{result.repetitions}:{','.join(result.tool_names)}"


@_experimental
class ToolLoopDetectionHook:
    """
    Detect repetitive or stalled tool-calling loops and nudge or stop the Agent.

    This `before_llm` hook inspects the conversation history in `state.data["messages"]` before each LLM call without
    keeping mutable state on the hook instance, so a single instance can be safely shared across concurrent Agent runs.

    It detects three loop patterns within the active user turn:
    - **Exact repeats**: consecutive steps issuing the exact same tool calls and arguments.
    - **Ping-pong cycles**: alternating tool-call sequences of length `2..max_cycle_length` (e.g. `A -> B -> A -> B`).
    - **Consecutive errors**: the same tool returning errors across consecutive steps, even when arguments vary.

    Intervention happens in two stages:
    1. When repetitions reach `warn_threshold` (if set), a `<system-reminder>` user message marked with
       `meta={"system_reminder": True}` is appended so the model can change strategy and recover.
    2. When repetitions reach `stop_threshold`, the hook sets `state.set("stop_run", "tool_loop_detected")`, ending the
       run cleanly before spending another LLM call.

    <!-- test-ignore -->
    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIResponsesChatGenerator
    from haystack.hooks.loop_detection import ToolLoopDetectionHook

    agent = Agent(
        chat_generator=OpenAIResponsesChatGenerator(),
        tools=[web_search],
        hooks={"before_llm": [ToolLoopDetectionHook(warn_threshold=2, stop_threshold=3)]},
    )

    result = agent.run(messages=[...])
    ```
    """

    allowed_hook_points = ("before_llm",)

    def __init__(
        self,
        *,
        warn_threshold: int | None = 2,
        stop_threshold: int = 3,
        max_cycle_length: int = 3,
        window_size: int = 20,
        detect_consecutive_errors: bool = True,
        add_final_message: bool = False,
    ) -> None:
        """
        Create a tool-loop detection hook.

        :param warn_threshold: Number of consecutive repetitions, completed ping-pong cycles, or consecutive tool
            errors at which a `<system-reminder>` warning is injected. Set to `None` to disable reminders and only stop
            at `stop_threshold`. Must be at least 2 and strictly less than `stop_threshold` when provided.
        :param stop_threshold: Number of consecutive repetitions, completed ping-pong cycles, or consecutive tool
            errors at which the Agent is stopped with `exit_reason="tool_loop_detected"`. Must be at least 2.
        :param max_cycle_length: Maximum cycle length checked for alternating ping-pong loops. Set to `1` to disable
            ping-pong cycle detection. Must be at least 1.
        :param window_size: Maximum number of trailing tool-calling steps inspected per check. Must be at least
            `stop_threshold`.
        :param detect_consecutive_errors: Whether consecutive failing invocations of the same tool count as a loop.
        :param add_final_message: Whether to append an assistant message explaining why the Agent stopped.
        :raises ValueError: If any threshold or window parameter is outside its valid range.
        """
        if stop_threshold < 2:
            raise ValueError(f"`stop_threshold` must be at least 2, got {stop_threshold}.")
        if warn_threshold is not None:
            if warn_threshold < 2:
                raise ValueError(f"`warn_threshold` must be at least 2 or None, got {warn_threshold}.")
            if warn_threshold >= stop_threshold:
                raise ValueError(
                    f"`warn_threshold` ({warn_threshold}) must be less than `stop_threshold` ({stop_threshold})."
                )
        if max_cycle_length < 1:
            raise ValueError(f"`max_cycle_length` must be at least 1, got {max_cycle_length}.")
        if window_size < stop_threshold:
            raise ValueError(
                f"`window_size` ({window_size}) must be greater than or equal to `stop_threshold` ({stop_threshold})."
            )

        self.warn_threshold = warn_threshold
        self.stop_threshold = stop_threshold
        self.max_cycle_length = max_cycle_length
        self.window_size = window_size
        self.detect_consecutive_errors = detect_consecutive_errors
        self.add_final_message = add_final_message

    def run(self, state: State) -> None:
        """
        Inspect recent tool-calling steps and warn or stop the Agent if a loop is detected.

        :param state: The Agent's live `State` containing `messages`.
        """
        messages = state.data.get("messages") or []
        if not messages:
            return

        result, action = detect_tool_loop(
            messages=messages,
            warn_threshold=self.warn_threshold,
            stop_threshold=self.stop_threshold,
            max_cycle_length=self.max_cycle_length,
            window_size=self.window_size,
            detect_consecutive_errors=self.detect_consecutive_errors,
        )
        if result is None or action is None:
            return

        if action == "stop":
            logger.warning("Agent stopped due to tool loop ({kind}): {detail}.", kind=result.kind, detail=result.detail)
            state.set("stop_run", _EXIT_REASON_TOOL_LOOP)
            if self.add_final_message:
                state.set(
                    "messages", [ChatMessage.from_assistant(text=_FINAL_MESSAGE_TEMPLATE.format(detail=result.detail))]
                )
            return

        key = _warning_key(result)
        if messages[-1].meta.get(_LOOP_WARNING_META_KEY) == key:
            return

        logger.warning(
            "Tool loop warning triggered ({kind}): {detail}; injecting reminder.",
            kind=result.kind,
            detail=result.detail,
        )
        text = _WARNING_TEMPLATE.format(detail=result.detail)
        reminder = ChatMessage.from_user(
            text=f"<system-reminder>\n{text}\n</system-reminder>",
            meta={_SYSTEM_REMINDER_META_KEY: True, _LOOP_WARNING_META_KEY: key},
        )
        state.set("messages", [reminder])

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this hook to a dictionary.

        :returns: Serialized representation of the hook.
        """
        return default_to_dict(
            self,
            warn_threshold=self.warn_threshold,
            stop_threshold=self.stop_threshold,
            max_cycle_length=self.max_cycle_length,
            window_size=self.window_size,
            detect_consecutive_errors=self.detect_consecutive_errors,
            add_final_message=self.add_final_message,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolLoopDetectionHook":
        """
        Create a hook from its serialized representation.

        :param data: Serialized hook data.
        :returns: The deserialized hook.
        """
        return default_from_dict(cls, data=data)
