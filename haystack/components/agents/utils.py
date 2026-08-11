# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from copy import deepcopy
from typing import Any

from haystack.components.agents.state.state import State
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.tools import Tool, Toolset, ToolsType

# Input/output token key conventions across chat generators: most report OpenAI-style
# `prompt_tokens`/`completion_tokens`; OpenAIResponsesChatGenerator reports `input_tokens`/`output_tokens`.
_INPUT_TOKEN_KEYS = ("prompt_tokens", "input_tokens")
_OUTPUT_TOKEN_KEYS = ("completion_tokens", "output_tokens")


# ---------------------------
# Run metadata helpers
# ---------------------------


def _accumulate_usage(current: Any, new: Any) -> Any:
    """
    Recursively sum numeric leaf values across two usage-like dicts.

    Used to aggregate `ChatMessage.meta["usage"]` payloads across LLM calls in a run. Nested dicts (e.g. OpenAI's
    `completion_tokens_details`) are merged recursively; numeric leaves are summed; other types fall back to the new
    value.

    :param current: The current accumulated usage data.
    :param new: The new usage data to merge in.
    """
    if isinstance(current, dict) and isinstance(new, dict):
        result = dict(current)
        for k, v in new.items():
            result[k] = _accumulate_usage(result[k], v) if k in result else deepcopy(v)
        return result
    if isinstance(current, (int, float)) and isinstance(new, (int, float)):
        return current + new
    return new


def _record_llm_usage(state: State, llm_messages: list[ChatMessage]) -> None:
    """
    Aggregate token usage from the latest LLM messages into the State.

    Only writes when at least one message reports `meta["usage"]`, so generators that don't surface usage data
    leave `token_usage` at its default empty dict rather than overwriting it.

    :param state: The Agent's State, used to read the running `token_usage` total and write back the new total.
    :param llm_messages: The ChatMessage objects returned from the latest LLM call. Token usage is read from each
        message's `meta["usage"]` field, if present.
    """
    current = state.data.get("token_usage")
    updated = False
    for msg in llm_messages:
        usage = msg.meta.get("usage")
        if isinstance(usage, dict):
            current = _accumulate_usage(current or {}, usage)
            updated = True
    if updated:
        state.set("token_usage", current)


def _record_tool_calls(state: State, tool_messages: list[ChatMessage]) -> None:
    """
    Increment per-tool call counts in the State for every successfully dispatched tool.

    :param state: The Agent's State, used to read the running `tool_call_counts` map and write back the new totals.
    :param tool_messages: The ChatMessage objects returned from the latest tool execution. Per-tool counts are
        incremented based on each message's `tool_call_result.origin.tool_name`.
    """
    counts = state.data.get("tool_call_counts") or {}
    updated = False
    for tm in tool_messages:
        if tm.tool_call_result is None:
            continue
        name = tm.tool_call_result.origin.tool_name
        counts[name] = counts.get(name, 0) + 1
        updated = True
    if updated:
        state.set("tool_call_counts", counts)


# ---------------------------
# Tool helpers
# ---------------------------


def _spawn_selection_copy(item: Tool | Toolset, selected_tool_names: set[str]) -> Toolset | None:
    """
    Return the per-run copy carrying the selection, or None if the item does not provide one.

    A Toolset with run-scoped state (e.g. SearchableToolset) overrides `spawn()` to return a copy that
    applies `selected_tool_names` itself. A plain Toolset returns itself from `spawn()` (it has nothing
    to isolate), and a standalone Tool has no `spawn()`: in both cases the caller applies the selection.

    :param item: A configured Tool or Toolset.
    :param selected_tool_names: The tool names selected for this run.
    :returns: The selection-carrying per-run copy, or None.
    """
    if not isinstance(item, Toolset):
        return None
    spawned = item.spawn(selected_tool_names=selected_tool_names)
    return spawned if spawned is not item else None


def _select_tools_by_name(configured_tools: ToolsType, names: list[str]) -> list[Tool | Toolset]:
    """
    Select configured tools by name for a single run.

    Standalone Tools are kept when their name is requested. A Toolset with run-scoped state (one overriding
    `spawn()`, such as SearchableToolset) is replaced by a per-run copy carrying the requested names, so its
    dynamic behavior (search/lazy-loading) is preserved without mutating the shared, configured Toolset. Any
    other Toolset is warmed up and reduced to the matching Tools.

    :param configured_tools: The tools configured on the Agent.
    :param names: The requested tool names.
    :returns: The selected Tools and/or selection-scoped Toolset copies.
    :raises ValueError: If no tools were configured, or if any requested name is not a valid tool name.
    """
    if configured_tools is None:
        raise ValueError("No tools were configured for the Agent at initialization.")

    requested_names = set(names)
    items: list[Tool | Toolset] = (
        [configured_tools] if isinstance(configured_tools, Toolset) else list(configured_tools)
    )

    # Resolve the tools each item offers for selection
    selectable_per_item: list[tuple[Tool | Toolset, list[Tool]]] = []
    for item in items:
        selectable = item.get_selectable_tools() if isinstance(item, Toolset) else [item]
        selectable_per_item.append((item, selectable))

    valid_tool_names = {tool.name for _, selectable in selectable_per_item for tool in selectable}
    # A dynamic Toolset may look empty before its catalog is resolved, so emptiness is checked here.
    if not valid_tool_names:
        raise ValueError("No tools were configured for the Agent at initialization.")

    invalid_tool_names = requested_names - valid_tool_names
    if invalid_tool_names:
        raise ValueError(
            f"The following tool names are not valid: {invalid_tool_names}. Valid tool names are: {valid_tool_names}."
        )

    selected: list[Tool | Toolset] = []
    for item, selectable in selectable_per_item:
        matched = requested_names & {tool.name for tool in selectable}
        if not matched:
            continue
        run_copy = _spawn_selection_copy(item, matched)
        if run_copy is not None:
            selected.append(run_copy)
        else:
            # Select from `selectable`, the list the names were validated against: iterating a dynamic
            # Toolset could silently miss tools.
            selected.extend(tool for tool in selectable if tool.name in matched)
    return selected


def _spawn_tools(tools: ToolsType) -> ToolsType:
    """
    Return per-run copies of `tools`, replacing each Toolset with its `spawn()` (Tools are passed through).

    This isolates run-scoped Toolset state (e.g. a SearchableToolset's discovered tools and any active name
    selection) so that concurrent runs sharing the same configured Toolset — such as parallel sub-agent tool calls
    or concurrent requests against one Agent — don't corrupt each other. A plain Toolset has no run-scoped state
    and its `spawn()` returns itself unchanged.
    """
    if isinstance(tools, Toolset):
        return tools.spawn()
    return [item.spawn() if isinstance(item, Toolset) else item for item in tools]


# ---------------------------
# Context token helpers
# ---------------------------


def _first_numeric(usage: dict[str, Any], keys: tuple[str, ...]) -> int:
    """
    Return the first numeric value found under `keys` in `usage`, or 0 if none is present.

    :param usage: A ChatMessage `meta["usage"]` payload.
    :param keys: Candidate keys to check, in priority order.
    :returns: The first `int`/`float` value (as an `int`), or 0. bool values are skipped (not token counts).
    """
    for key in keys:
        value = usage.get(key)
        # bool is an int subclass, so exclude it explicitly: True/False is not a token count.
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def _context_tokens_from_usage(usage: dict[str, Any]) -> int:
    """
    Sum the input and output tokens reported in a single `meta["usage"]` dict.

    :param usage: A ChatMessage `meta["usage"]` payload.
    :returns: Input plus output tokens, or 0 if neither key convention is present.
    """
    return _first_numeric(usage, _INPUT_TOKEN_KEYS) + _first_numeric(usage, _OUTPUT_TOKEN_KEYS)


def _record_context_tokens(state: State, llm_messages: list[ChatMessage]) -> None:
    """
    Store the approximate current context-window token count from the latest LLM call.

    A chat-generator call returns a single reply, so only the last message is inspected. Unlike
    `token_usage`, which accumulates across the run, this value is replaced each call with that reply's
    prompt-plus-completion tokens. Only writes when usage is reported, so generators that don't surface
    usage leave the previous value untouched.

    :param state: The Agent's State, used to write the latest `context_tokens` count.
    :param llm_messages: The ChatMessage objects returned from the latest LLM call.
    """
    if not llm_messages:
        return
    usage = llm_messages[-1].meta.get("usage")
    if isinstance(usage, dict):
        tokens = _context_tokens_from_usage(usage)
        if tokens:
            state.set("context_tokens", tokens)


# ---------------------------
# Prompt helpers
# ---------------------------

# Regex to detect the Jinja2 chat template syntax
_JINJA2_CHAT_TEMPLATE_RE = re.compile(r"\{%\s*message\s")
# Regex to extract the role from a Jinja2 message block, e.g. {% message role="user" %}
_JINJA2_MESSAGE_ROLE_RE = re.compile(r'\{%\s*message\s+role\s*=\s*["\'](\w+)["\']')


def _validate_prompt_message_blocks(user_prompt: str | None, system_prompt: str | None) -> None:
    """
    Validate explicit Jinja2 message blocks in Agent prompts.

    :param user_prompt: Optional user prompt template.
    :param system_prompt: Optional system prompt template.
    :raises ValueError: If a prompt contains multiple message blocks or a literal block role is invalid.
    """
    if user_prompt is not None:
        message_blocks = _JINJA2_CHAT_TEMPLATE_RE.findall(user_prompt)
        roles = _JINJA2_MESSAGE_ROLE_RE.findall(user_prompt)
        if len(message_blocks) > 1:
            raise ValueError(f"user_prompt must define exactly one message block, found {len(message_blocks)}.")
        if roles and roles[0] != "user":
            raise ValueError(f"user_prompt message block must have role 'user', found role '{roles[0]}'.")

    if system_prompt is not None and _JINJA2_CHAT_TEMPLATE_RE.search(system_prompt):
        message_blocks = _JINJA2_CHAT_TEMPLATE_RE.findall(system_prompt)
        roles = _JINJA2_MESSAGE_ROLE_RE.findall(system_prompt)
        if len(message_blocks) > 1:
            raise ValueError(f"system_prompt must define exactly one message block, found {len(message_blocks)}.")
        if roles and roles[0] != "system":
            raise ValueError(f"system_prompt message block must have role 'system', found role '{roles[0]}'.")


def _template_for_role(prompt: str, role: str) -> str:
    """
    Convert a prompt into a ChatPromptBuilder string template for the expected role.

    :param prompt: Prompt template, with or without an explicit Jinja2 message block.
    :param role: Role to use when wrapping a plain string prompt.
    :returns: The original message-block template, or a plain string prompt wrapped in one message block.
    """
    if _JINJA2_CHAT_TEMPLATE_RE.search(prompt):
        return prompt
    return f'{{% message role="{role}" %}}{prompt}{{% endmessage %}}'


def _render_prompt_messages(
    *, prompt_builder: ChatPromptBuilder, expected_role: ChatRole, prompt_label: str, kwargs: dict[str, Any]
) -> list[ChatMessage]:
    """
    Render one Agent prompt and validate the rendered message.

    :param prompt_builder: Builder configured with the prompt template.
    :param expected_role: Role the rendered message must have.
    :param prompt_label: Prompt name used in error messages.
    :param kwargs: Runtime values available to the prompt template.
    :returns: A single rendered prompt message.
    :raises ValueError: If the prompt renders to zero, multiple, or wrong-role messages.
    """
    prompt_kwargs = {var: kwargs[var] for var in prompt_builder.variables if var in kwargs}
    prompt_messages = prompt_builder.run(**prompt_kwargs)["prompt"]
    if len(prompt_messages) != 1:
        raise ValueError(
            f"{prompt_label} must render to exactly one {expected_role.value} message. "
            f"Got {len(prompt_messages)} messages."
        )
    if not prompt_messages[0].is_from(expected_role):
        raise ValueError(
            f"{prompt_label} must render to a {expected_role.value} message. "
            f"Got a message with role {prompt_messages[0].role}."
        )
    return prompt_messages
