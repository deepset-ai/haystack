# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any


def _openai_tool_definition_key(item: dict[str, Any]) -> str | None:
    """Return a stable dedupe key for an OpenAI-style tool definition, or None if it cannot be keyed."""
    tool_type = item.get("type")
    nested = item.get("function")
    if tool_type == "function" and isinstance(nested, dict):
        name = nested.get("name")
        if isinstance(name, str) and name:
            return f"function:{name}"
    name = item.get("name")
    if isinstance(name, str) and name:
        type_label = tool_type if isinstance(tool_type, str) else "tool"
        return f"{type_label}:{name}"
    return None


def _merge_openai_tool_definitions(primary: list[dict[str, Any]], extra: Any) -> list[dict[str, Any]]:
    """
    Merge OpenAI API tool definition lists.

    Entries from ``primary`` (typically from the ``tools`` run argument) win over ``extra`` (typically from
    ``generation_kwargs["tools"]``) when they share the same dedupe key.
    """
    merged = list(primary)
    seen = {key for item in primary if (key := _openai_tool_definition_key(item)) is not None}
    if not isinstance(extra, list):
        return merged
    for item in extra:
        if not isinstance(item, dict):
            continue
        key = _openai_tool_definition_key(item)
        if key is None:
            merged.append(item)
            continue
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _check_duplicate_openai_tool_definition_keys(tool_definitions: list[dict[str, Any]]) -> None:
    """Raise ``ValueError`` when two keyed tool definitions share the same dedupe key."""
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item in tool_definitions:
        key = _openai_tool_definition_key(item)
        if key is None:
            continue
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        raise ValueError(f"Duplicate tool names found: {duplicates}")
