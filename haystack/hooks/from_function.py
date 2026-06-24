# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, cast

from haystack.components.agents.state.state import State
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable


class FunctionHook:
    """
    Wraps a function (or a sync/async pair) into a serializable `Hook`.

    Produced by the `@hook` decorator for the single-function case. To give a hook both an optimized sync and async
    path, construct it directly with both `function` and `async_function` set.
    """

    def __init__(
        self,
        function: Callable[[State], None] | None = None,
        async_function: Callable[[State], Awaitable[None]] | None = None,
    ) -> None:
        """
        Initialize the hook with a synchronous function, an async function, or both.

        :param function: The synchronous function invoked by `run`. Must be a regular function — coroutine functions
            should be passed to `async_function` instead. Either `function` or `async_function` (or both) must be set.
        :param async_function: Optional coroutine function awaited by `run_async`. When only `async_function` is set,
            `run` raises a `RuntimeError`. When only `function` is set, `run_async` calls `function`.
        :raises ValueError: If neither is set, if `function` is a coroutine function, or if `async_function` is not.
        """
        if function is None and async_function is None:
            raise ValueError("A FunctionHook requires at least one of `function` or `async_function` to be set.")
        if function is not None and inspect.iscoroutinefunction(function):
            raise ValueError(
                f"`function` must be a synchronous function. '{function.__name__}' is a coroutine function. "
                "Pass it as `async_function` instead."
            )
        if async_function is not None and not inspect.iscoroutinefunction(async_function):
            raise ValueError(
                f"`async_function` must be a coroutine function defined with `async def`. "
                f"Got '{getattr(async_function, '__name__', repr(async_function))}'."
            )
        self.function = function
        self.async_function = async_function

    def run(self, state: State) -> None:
        """Run the synchronous function against the live `State`."""
        if self.function is None:
            raise RuntimeError(
                "This FunctionHook only has an `async_function` and cannot run in a synchronous Agent run. "
                "Use the Agent's async run methods, or provide a synchronous `function`."
            )
        self.function(state)

    async def run_async(self, state: State) -> None:
        """Await the async function if set, otherwise call the synchronous function."""
        if self.async_function is not None:
            await self.async_function(state)
        else:
            self.function(state)  # type: ignore[misc]  # guaranteed non-None: at least one is always set

    def to_dict(self) -> dict[str, Any]:
        """Serialize the hook, storing each function as an importable reference."""
        return default_to_dict(
            self,
            function=serialize_callable(self.function) if self.function is not None else None,
            async_function=serialize_callable(self.async_function) if self.async_function is not None else None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FunctionHook":
        """Deserialize the hook, resolving each function from its importable reference."""
        init_params = data.get("init_parameters", {})
        if init_params.get("function") is not None:
            init_params["function"] = deserialize_callable(init_params["function"])
        if init_params.get("async_function") is not None:
            init_params["async_function"] = deserialize_callable(init_params["async_function"])
        return default_from_dict(cls, data)


def hook(function: Callable[[State], None | Awaitable[None]]) -> FunctionHook:
    """
    Wrap a function into a `Hook` the Agent can invoke during its run loop.

    The decorated function receives the Agent's `State` and influences the run by mutating it in place. A coroutine
    function is wrapped as the hook's async path; a regular function as its sync path. To give a single hook both
    paths, construct a `FunctionHook` directly with both `function` and `async_function`.

    ### Usage example

    ```python
    from haystack.components.agents import Agent
    from haystack.hooks import hook
    from haystack.components.agents.state import State
    from haystack.dataclasses import ChatMessage

    @hook
    def require_save(state: State) -> None:
        if state.get("tool_call_counts", {}).get("save", 0) == 0:
            state.set("messages", [ChatMessage.from_system("You must call `save` before finishing.")])

    agent = Agent(chat_generator=..., tools=[...], hooks={"on_exit": [require_save]})
    ```

    :param function: A callable taking the Agent's `State` and returning `None` (sync or async).
    :returns: A `FunctionHook` wrapping the function.
    """
    # `iscoroutinefunction` narrows which slot the callable belongs in; cast to satisfy the typed slots.
    if inspect.iscoroutinefunction(function):
        return FunctionHook(async_function=cast("Callable[[State], Awaitable[None]]", function))
    return FunctionHook(function=cast("Callable[[State], None]", function))
