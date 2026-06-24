# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import component_to_dict
from haystack.hooks.protocol import Hook, HookEvent
from haystack.utils.deserialization import deserialize_component_inplace


def _serialize_hooks(hooks: dict[HookEvent, list[Hook]]) -> dict[str, list[dict[str, Any]]]:
    """Serialize an event-keyed dict of hooks to plain dictionaries."""
    return {event: [component_to_dict(obj=h, name="hook") for h in hook_list] for event, hook_list in hooks.items()}


def _deserialize_hooks(data: dict[str, list[dict[str, Any]]]) -> dict[str, list[Hook]]:
    """Deserialize an event-keyed dict of hooks from its serialized form."""
    deserialized: dict[str, list[Hook]] = {}
    for event, serialized_hooks in data.items():
        hooks: list[Hook] = []
        for serialized_hook in serialized_hooks:
            wrapper: dict[str, Any] = {"hook": serialized_hook}
            deserialize_component_inplace(wrapper, key="hook")
            hooks.append(wrapper["hook"])
        deserialized[event] = hooks
    return deserialized


def _unique_hooks(hooks: dict[HookEvent, list[Hook]]) -> list[Hook]:
    """Collect each distinct hook object once, preserving order, since a hook may be registered under many events."""
    unique: list[Hook] = []
    seen: set[int] = set()
    for hook_list in hooks.values():
        for h in hook_list:
            if id(h) not in seen:
                seen.add(id(h))
                unique.append(h)
    return unique


def warm_up_hooks(hooks: dict[HookEvent, list[Hook]]) -> None:
    """Warm up every hook that defines a `warm_up` method (e.g. to open clients or load credentials)."""
    for h in _unique_hooks(hooks):
        if hasattr(h, "warm_up"):
            h.warm_up()


async def warm_up_hooks_async(hooks: dict[HookEvent, list[Hook]]) -> None:
    """Warm up every hook, awaiting `warm_up_async` when defined and falling back to `warm_up` otherwise."""
    for h in _unique_hooks(hooks):
        warm_up_async = getattr(h, "warm_up_async", None)
        if warm_up_async is not None:
            await warm_up_async()
        elif hasattr(h, "warm_up"):
            h.warm_up()


def close_hooks(hooks: dict[HookEvent, list[Hook]]) -> None:
    """Release the resources of every hook that defines a `close` method."""
    for h in _unique_hooks(hooks):
        if hasattr(h, "close"):
            h.close()


async def close_hooks_async(hooks: dict[HookEvent, list[Hook]]) -> None:
    """Release hook resources, awaiting `close_async` when defined and falling back to `close` otherwise."""
    for h in _unique_hooks(hooks):
        close_async = getattr(h, "close_async", None)
        if close_async is not None:
            await close_async()
        elif hasattr(h, "close"):
            h.close()
