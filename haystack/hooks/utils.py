# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.serialization import component_to_dict
from haystack.hooks.protocol import Hook, HookPoint
from haystack.utils.deserialization import deserialize_component_inplace


# Hooks are (de)serialized with `component_to_dict` / `deserialize_component_inplace` even though they aren't
# Components. Despite the name, those helpers aren't component-specific: they just produce/consume the standard
# `{"type", "init_parameters"}` dict, and deserialization enforces the import allowlist (so only trusted modules are
# loaded).
def _serialize_hooks_dictionary(hooks: dict[HookPoint, list[Hook]]) -> dict[str, list[dict[str, Any]]]:
    """
    Serialize a hook-point-keyed dict of hooks to plain dictionaries.

    :param hooks: Hooks keyed by hook point; each hook must implement `to_dict`.
    :returns: The same mapping with each hook replaced by its serialized dictionary.
    """
    return {
        hook_point: [component_to_dict(obj=h, name="hook") for h in hook_list]
        for hook_point, hook_list in hooks.items()
    }


def _deserialize_hooks_dictionary(data: dict[str, list[dict[str, Any]]]) -> dict[str, list[Hook]]:
    """
    Deserialize a hook-point-keyed dict of hooks from its serialized form.

    :param data: Hook-point-keyed lists of serialized hook dictionaries (each with a `type` field).
    :returns: The same mapping with each entry rebuilt into a `Hook` instance.
    """
    deserialized: dict[str, list[Hook]] = {}
    for hook_point, serialized_hooks in data.items():
        hooks: list[Hook] = []
        for serialized_hook in serialized_hooks:
            wrapper: dict[str, Any] = {"hook": serialized_hook}
            deserialize_component_inplace(wrapper, key="hook")
            hooks.append(wrapper["hook"])
        deserialized[hook_point] = hooks
    return deserialized


def _unique_hooks(hooks: dict[HookPoint, list[Hook]]) -> list[Hook]:
    """
    Collect each distinct hook once, preserving first-seen order.

    A hook may be registered under several hook points; deduplicating by identity ensures lifecycle methods
    (warm up / close) run once per hook object.

    :param hooks: Hooks keyed by hook point.
    :returns: The distinct hook objects, in the order first encountered.
    """
    unique: list[Hook] = []
    seen: set[int] = set()
    for hook_list in hooks.values():
        for h in hook_list:
            if id(h) not in seen:
                seen.add(id(h))
                unique.append(h)
    return unique


def warm_up_hooks(hooks: dict[HookPoint, list[Hook]]) -> None:
    """
    Warm up every hook that defines a `warm_up` method (e.g. to open clients or load credentials).

    :param hooks: Hooks keyed by hook point. Each distinct hook is warmed up at most once.
    """
    for h in _unique_hooks(hooks):
        if hasattr(h, "warm_up"):
            h.warm_up()


async def warm_up_hooks_async(hooks: dict[HookPoint, list[Hook]]) -> None:
    """
    Warm up every hook, awaiting `warm_up_async` when defined and falling back to `warm_up` otherwise.

    :param hooks: Hooks keyed by hook point. Each distinct hook is warmed up at most once.
    """
    for h in _unique_hooks(hooks):
        warm_up_async = getattr(h, "warm_up_async", None)
        if warm_up_async is not None:
            await warm_up_async()
        elif hasattr(h, "warm_up"):
            h.warm_up()


def close_hooks(hooks: dict[HookPoint, list[Hook]]) -> None:
    """
    Release the resources of every hook that defines a `close` method.

    :param hooks: Hooks keyed by hook point. Each distinct hook is closed at most once.
    """
    for h in _unique_hooks(hooks):
        if hasattr(h, "close"):
            h.close()


async def close_hooks_async(hooks: dict[HookPoint, list[Hook]]) -> None:
    """
    Release hook resources, awaiting `close_async` when defined and falling back to `close` otherwise.

    :param hooks: Hooks keyed by hook point. Each distinct hook is closed at most once.
    """
    for h in _unique_hooks(hooks):
        close_async = getattr(h, "close_async", None)
        if close_async is not None:
            await close_async()
        elif hasattr(h, "close"):
            h.close()
