# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.components.agents.state.state import State
from haystack.dataclasses import ChatMessage

# Input/output token key conventions across chat generators: most report OpenAI-style
# `prompt_tokens`/`completion_tokens`; OpenAIResponsesChatGenerator reports `input_tokens`/`output_tokens`.
_INPUT_TOKEN_KEYS = ("prompt_tokens", "input_tokens")
_OUTPUT_TOKEN_KEYS = ("completion_tokens", "output_tokens")


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
