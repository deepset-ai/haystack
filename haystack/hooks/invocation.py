# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack import tracing
from haystack.components.agents.state.state import State
from haystack.core.errors import SerializationError
from haystack.core.serialization import generate_qualified_class_name
from haystack.hooks.from_function import FunctionHook
from haystack.hooks.protocol import Hook, HookPoint
from haystack.tracing import Span
from haystack.utils.async_utils import _execute_component_async
from haystack.utils.callable_serialization import serialize_callable


def _hook_name(hook: Hook) -> str:
    """Return a human-readable identifier for a hook."""
    if isinstance(hook, FunctionHook):
        # FunctionHook is a generic wrapper, so identify it by its wrapped callable instead.
        function = hook.function or hook.async_function
        if function is not None:
            try:
                try:
                    return serialize_callable(callable_handle=function)
                except SerializationError:
                    # Nested functions and lambdas are unsupported by serialization, but their qualified names are still
                    # useful.
                    return f"{function.__module__}.{function.__qualname__}"
            # functools.partial and callable-object instances may not expose __module__, __qualname__, or __name__.
            except Exception:
                pass
    return type(hook).__name__


def _create_hook_span(hook: Hook, hook_point: HookPoint, parent_span: Span | None) -> Any:
    """Create a content-free tracing span for one hook invocation."""
    return tracing.tracer.trace(
        "haystack.agent.hook",
        tags={
            "haystack.agent.hook.point": hook_point,
            "haystack.agent.hook.name": _hook_name(hook=hook),
            "haystack.agent.hook.type": generate_qualified_class_name(cls=type(hook)),
        },
        parent_span=parent_span,
    )


def _run_hooks(hooks: dict[HookPoint, list[Hook]], hook_point: HookPoint, state: State) -> None:
    """
    Run every hook registered for the given hook point, in list order.

    :param hooks: Hooks keyed by hook point.
    :param hook_point: The hook point whose hooks to run; hooks registered under other hook points are skipped.
    :param state: The Agent's live `State`, passed to each hook and mutated in place.
    """
    hooks_to_run = hooks.get(hook_point, [])
    if not hooks_to_run:
        return

    parent_span = tracing.tracer.current_span()
    for h in hooks_to_run:
        with _create_hook_span(hook=h, hook_point=hook_point, parent_span=parent_span):
            h.run(state)


async def _run_hooks_async(hooks: dict[HookPoint, list[Hook]], hook_point: HookPoint, state: State) -> None:
    """
    Run every hook for the given hook point, preferring `run_async` and offloading sync-only `run` hooks.

    :param hooks: Hooks keyed by hook point.
    :param hook_point: The hook point whose hooks to run; hooks registered under other hook points are skipped.
    :param state: The Agent's live `State`, passed to each hook and mutated in place.
    """
    hooks_to_run = hooks.get(hook_point, [])
    if not hooks_to_run:
        return

    parent_span = tracing.tracer.current_span()
    for h in hooks_to_run:
        with _create_hook_span(hook=h, hook_point=hook_point, parent_span=parent_span):
            await _execute_component_async(h, state=state)
