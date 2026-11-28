# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from haystack.dataclasses import ChatMessage, ChatRole
from haystack.hooks.utils import _is_system_reminder

LoopKind = Literal["exact_repeat", "ping_pong", "consecutive_error"]
InterventionAction = Literal["warn", "stop"]


@dataclass(frozen=True)
class LoopDetectionResult:
    """
    Structured description of a detected tool-calling loop in an Agent conversation.

    :param kind: Category of loop detected: `"exact_repeat"`, `"ping_pong"`, or `"consecutive_error"`.
    :param tool_names: Tool names involved in the detected loop.
    :param repetitions: Number of consecutive repetitions, completed cycles, or consecutive tool errors observed.
    :param cycle_length: Number of tool-call steps per cycle (`1` for exact repeats and consecutive errors, `>= 2` for
        ping-pong cycles).
    :param detail: Human-readable summary of the detected loop.
    """

    kind: LoopKind
    tool_names: tuple[str, ...]
    repetitions: int
    cycle_length: int
    detail: str


@dataclass(frozen=True)
class _ToolStep:
    """Normalized record of one assistant tool-calling turn and its immediately following tool errors."""

    call_signature: str
    tool_names: tuple[str, ...]
    errored_tools: tuple[str, ...]


def _canonical_json(value: Any) -> str:
    """Serialize arbitrary tool arguments deterministically, independent of dictionary key order."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _step_signature(message: ChatMessage) -> tuple[str, tuple[str, ...]]:
    """
    Compute a canonical SHA-256 signature and sorted distinct tool names for an assistant tool-call message.

    Sorting the `(tool_name, canonical_arguments)` pairs ensures that parallel tool calls issued in a different order
    produce the same signature.
    """
    normalized_calls = sorted((tc.tool_name, _canonical_json(tc.arguments)) for tc in message.tool_calls)
    digest = hashlib.sha256(_canonical_json(normalized_calls).encode("utf-8")).hexdigest()
    tool_names = tuple(sorted({tc.tool_name for tc in message.tool_calls}))
    return digest, tool_names


def _current_turn_start(messages: Sequence[ChatMessage]) -> int:
    """
    Return the index after the latest non-reminder user message.

    In multi-turn conversations, tool calls from earlier user turns belong to prior tasks and should not trigger loop
    detection for the current turn. Synthetic `<system-reminder>` user messages injected by hooks are skipped so they
    do not reset the active loop window.
    """
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if msg.is_from(role=ChatRole.USER) and not _is_system_reminder(msg):
            return idx + 1
    return 0


def _extract_tool_steps(messages: Sequence[ChatMessage], *, window_size: int) -> list[_ToolStep]:
    """Extract normalized tool-calling steps from the active conversation turn, capped to the trailing window."""
    start = _current_turn_start(messages)
    steps: list[_ToolStep] = []

    current_sig: str | None = None
    current_tools: tuple[str, ...] = ()
    current_errors: set[str] = set()

    for msg in messages[start:]:
        if msg.is_from(role=ChatRole.ASSISTANT) and msg.tool_calls:
            if current_sig is not None:
                steps.append(
                    _ToolStep(
                        call_signature=current_sig,
                        tool_names=current_tools,
                        errored_tools=tuple(sorted(current_errors)),
                    )
                )
            current_sig, current_tools = _step_signature(msg)
            current_errors = set()
        elif current_sig is not None and (result := msg.tool_call_result) is not None and result.error:
            current_errors.add(result.origin.tool_name)

    if current_sig is not None:
        steps.append(
            _ToolStep(call_signature=current_sig, tool_names=current_tools, errored_tools=tuple(sorted(current_errors)))
        )

    return steps[-window_size:]


def _detect_exact_repeat(steps: Sequence[_ToolStep]) -> LoopDetectionResult | None:
    """Detect consecutive trailing steps that issue identical tool calls."""
    if len(steps) < 2:
        return None

    target_sig = steps[-1].call_signature
    count = 1
    for idx in range(len(steps) - 2, -1, -1):
        if steps[idx].call_signature == target_sig:
            count += 1
        else:
            break

    if count < 2:
        return None

    tool_names = steps[-1].tool_names
    tools_label = ", ".join(tool_names)
    return LoopDetectionResult(
        kind="exact_repeat",
        tool_names=tool_names,
        repetitions=count,
        cycle_length=1,
        detail=f"identical tool call(s) [{tools_label}] repeated {count} times in a row",
    )


def _detect_ping_pong(steps: Sequence[_ToolStep], *, max_cycle_length: int) -> LoopDetectionResult | None:
    """Detect trailing alternating tool-call cycles of length `2..max_cycle_length`."""
    n_steps = len(steps)
    upper_k = min(max_cycle_length, n_steps // 2)
    if upper_k < 2:
        return None

    best: LoopDetectionResult | None = None
    for cycle_len in range(2, upper_k + 1):
        pattern = [step.call_signature for step in steps[-cycle_len:]]
        # A constant pattern is already captured by `_detect_exact_repeat`.
        if len(set(pattern)) == 1:
            continue

        cycles = 1
        pos = n_steps - 2 * cycle_len
        while pos >= 0:
            candidate = [step.call_signature for step in steps[pos : pos + cycle_len]]
            if candidate == pattern:
                cycles += 1
                pos -= cycle_len
            else:
                break

        if cycles >= 2 and (best is None or cycles > best.repetitions):
            seen: list[str] = []
            for step in steps[-cycle_len:]:
                for name in step.tool_names:
                    if name not in seen:
                        seen.append(name)
            tool_names = tuple(seen)
            tools_label = " -> ".join(tool_names)
            best = LoopDetectionResult(
                kind="ping_pong",
                tool_names=tool_names,
                repetitions=cycles,
                cycle_length=cycle_len,
                detail=f"alternating {cycle_len}-step tool cycle ({tools_label}) repeated {cycles} times",
            )

    return best


def _detect_consecutive_errors(steps: Sequence[_ToolStep]) -> LoopDetectionResult | None:
    """Detect a tool returning errors across consecutive trailing steps, even when arguments vary."""
    if len(steps) < 2 or not steps[-1].errored_tools:
        return None

    best_tool = ""
    best_count = 0
    for tool_name in steps[-1].errored_tools:
        count = 1
        for idx in range(len(steps) - 2, -1, -1):
            if tool_name in steps[idx].errored_tools:
                count += 1
            else:
                break
        if count > best_count:
            best_count = count
            best_tool = tool_name

    if best_count < 2:
        return None

    return LoopDetectionResult(
        kind="consecutive_error",
        tool_names=(best_tool,),
        repetitions=best_count,
        cycle_length=1,
        detail=f"tool '{best_tool}' failed {best_count} times in a row",
    )


def detect_tool_loop(
    messages: Sequence[ChatMessage],
    *,
    warn_threshold: int | None,
    stop_threshold: int,
    max_cycle_length: int,
    window_size: int,
    detect_consecutive_errors: bool,
) -> tuple[LoopDetectionResult | None, InterventionAction | None]:
    """
    Analyze the active Agent conversation for tool-calling loops and return the triggered intervention.

    Stop-level triggers take precedence over warn-level triggers. Within the same severity level, candidates are
    evaluated in deterministic order: exact repeat, ping-pong cycle, then consecutive tool errors.

    :param messages: Full Agent message history, oldest to newest.
    :param warn_threshold: Repetition threshold for emitting a warning reminder, or `None` to disable warnings.
    :param stop_threshold: Repetition threshold for stopping the Agent run.
    :param max_cycle_length: Maximum cycle length to inspect for alternating ping-pong loops (`1` disables it).
    :param window_size: Maximum number of trailing tool-calling steps to inspect.
    :param detect_consecutive_errors: Whether to detect consecutive errors from the same tool.
    :returns: A `(result, action)` tuple where `action` is `"stop"`, `"warn"`, or `None` when no threshold is met.
    """
    steps = _extract_tool_steps(messages=messages, window_size=window_size)
    if len(steps) < 2:
        return None, None

    candidates: list[LoopDetectionResult] = []
    exact = _detect_exact_repeat(steps)
    if exact is not None:
        candidates.append(exact)
    if max_cycle_length >= 2:
        ping_pong = _detect_ping_pong(steps, max_cycle_length=max_cycle_length)
        if ping_pong is not None:
            candidates.append(ping_pong)
    if detect_consecutive_errors:
        errors = _detect_consecutive_errors(steps)
        if errors is not None:
            candidates.append(errors)

    for candidate in candidates:
        if candidate.repetitions >= stop_threshold:
            return candidate, "stop"

    if warn_threshold is not None:
        for candidate in candidates:
            if candidate.repetitions >= warn_threshold:
                return candidate, "warn"

    return None, None
