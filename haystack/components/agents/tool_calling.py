# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextvars
import inspect
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from haystack import logging
from haystack.components.agents.state.state import State
from haystack.core.component.sockets import Sockets
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses.streaming_chunk import StreamingCallbackT, StreamingChunk, _invoke_streaming_callback
from haystack.tools import ComponentTool, Tool, ToolsType, _check_duplicate_tool_names, flatten_tools_or_toolsets
from haystack.tools.errors import ToolInvocationError
from haystack.tools.parameters_schema_utils import _unwrap_optional
from haystack.tracing.utils import _serializable_value

logger = logging.getLogger(__name__)


class ToolNotFoundException(Exception):
    """Exception raised when a tool is not found in the list of available tools."""

    def __init__(self, tool_name: str, available_tools: list[str]) -> None:
        message = f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}"
        super().__init__(message)


def _validate_and_prepare_tools(tools: ToolsType) -> dict[str, Tool]:
    """
    Flatten, deduplicate-check, and index tools by name.

    :raises ValueError: If no tools are provided or if duplicate tool names are found.
    """
    if not tools:
        raise ValueError("Tool execution requires at least one tool.")

    available_tools = flatten_tools_or_toolsets(tools)
    _check_duplicate_tool_names(available_tools)
    tool_names = [tool.name for tool in available_tools]

    return dict(zip(tool_names, available_tools, strict=True))


def _merge_tool_outputs_into_state(tool: Tool, result: Any, state: State) -> None:
    """
    Write tool outputs into State according to the tool's `outputs_to_state` mapping.

    :raises RuntimeError: If writing an output value into the state fails.
    """
    if not isinstance(result, dict):
        return
    if not hasattr(tool, "outputs_to_state") or not isinstance(tool.outputs_to_state, dict):
        return

    for state_key, config in tool.outputs_to_state.items():
        source_key = config.get("source", None)
        if source_key and source_key not in result:
            continue
        output_value = result.get(source_key) if source_key else result
        try:
            state.set(state_key, output_value, handler_override=config.get("handler"))
        except Exception as e:
            raise RuntimeError(f"Tool '{tool.name}': failed to merge outputs into state. {e}") from e


def _result_to_string(result: Any) -> str:
    """
    Convert a tool result to a string.

    Strings are returned as-is; all other types are passed through a JSON serialization step to produce more readable
    output, with a fallback to plain str() conversion if serialization fails.

    :param result: The tool result to convert.
    :returns: A string representation of the tool result.
    """
    if isinstance(result, str):
        return result
    serializable = _serializable_value(value=result, use_placeholders=False)
    try:
        return json.dumps(serializable, ensure_ascii=False)
    except Exception as error:
        logger.warning(
            "Tool result is not JSON serializable. Falling back to str conversion. Result: {result}\nError: {err}",
            result=result,
            err=error,
        )
        return str(result)


def _process_tool_output(config: dict[str, Any], result: Any, tool_call: ToolCall, *, raise_on_failure: bool) -> Any:
    """
    Extract and convert a single tool output according to `config`.

    `config` may contain `source` (key to extract from result dict), `handler` (conversion callable), and
    `raw_result` (return the value without string conversion).

    If a configured `handler` raises, the exception is re-raised when `raise_on_failure` is True; otherwise
    a warning is logged and the value is converted via `_result_to_string`.
    """
    source_key = config.get("source")
    value = result.get(source_key) if source_key is not None and isinstance(result, dict) else result

    handler = config.get("handler")
    raw_result = config.get("raw_result", False)

    if handler is None:
        # raw result is mostly used to allow ImageContent or TextContent blocks to be directly returned and consumed
        # by ChatMessage.from_tool without string conversion.
        if raw_result:
            return value
        return _result_to_string(value)

    try:
        return handler(value)
    except Exception as e:
        if raise_on_failure:
            raise
        logger.warning(
            "Output handler '{handler}' for tool '{tool}' failed, falling back to string conversion. Error: {err}",
            handler=handler.__name__,
            tool=tool_call.tool_name,
            err=e,
        )
        return _result_to_string(value)


def _build_tool_result_message(result: Any, tool_call: ToolCall, tool: Tool, *, raise_on_failure: bool) -> ChatMessage:
    """Convert a raw tool result into a ChatMessage, applying `outputs_to_string` config if present."""
    outputs_config = tool.outputs_to_string or {}

    # Single-output config (or no config): keys are at the root level
    if not outputs_config or any(k in outputs_config for k in ("source", "handler", "raw_result")):
        tool_result = _process_tool_output(outputs_config, result, tool_call, raise_on_failure=raise_on_failure)
        return ChatMessage.from_tool(tool_result=tool_result, origin=tool_call)

    # Multi-output config: each key maps to its own sub-config — stringify each value, then stringify the whole dict
    tool_result_dict = {
        output_key: _process_tool_output(
            {**cfg, "raw_result": False}, result, tool_call, raise_on_failure=raise_on_failure
        )
        for output_key, cfg in outputs_config.items()
    }
    return ChatMessage.from_tool(tool_result=_result_to_string(tool_result_dict), origin=tool_call)


def _create_tool_result_streaming_chunk(tool_messages: list[ChatMessage], tool_call: ToolCall) -> StreamingChunk:
    """
    Create a streaming chunk that carries the latest tool result.

    :param tool_messages: The list of tool result messages so far, with the latest result at the end.
    :param tool_call: The ToolCall object that triggered the tool invocation.
    :returns: A StreamingChunk containing the latest tool result and metadata about the tool call.
    """
    return StreamingChunk(
        content="",
        index=len(tool_messages) - 1,
        tool_call_result=tool_messages[-1].tool_call_results[0],
        start=True,
        meta={"tool_result": tool_messages[-1].tool_call_results[0].result, "tool_call": tool_call},
    )


def _make_context_bound_invoke(tool: Tool, args: dict[str, Any]) -> Callable[[], Any]:
    """
    Return a zero-arg callable that runs `tool.invoke(**args)` under the current contextvars snapshot.

    This preserves tracing spans and other context-local state across thread-pool boundaries.
    The callable returns a ToolInvocationError instead of raising so that parallel executions can
    collect failures without aborting the whole batch.
    """
    ctx = contextvars.copy_context()

    def _runner() -> Any:
        try:
            return ctx.run(partial(tool.invoke, **args))
        except ToolInvocationError as e:
            return e

    return _runner


def _make_bounded_invoke_async(tool: Tool, args: dict[str, Any], semaphore: asyncio.Semaphore) -> Callable[[], Any]:
    """
    Return a zero-arg async callable that awaits `tool.invoke_async(**args)` while holding `semaphore`.

    Concurrency is bounded uniformly across native-async tools and sync-fallback tools (which dispatch
    to a worker thread inside `Tool.invoke_async`). ContextVars naturally inherit into child tasks for
    the native-async branch, and `asyncio.to_thread` propagates them for the fallback branch.

    Returns a `ToolInvocationError` instead of raising so that gathered executions can collect failures
    without aborting the whole batch.
    """

    async def _runner() -> Any:
        async with semaphore:
            try:
                return await tool.invoke_async(**args)
            except ToolInvocationError as e:
                return e

    return _runner


def _get_func_params(tool: Tool) -> dict[str, Any]:
    """
    Return parameter names → annotations for a tool's invocation function.

    - For ComponentTool, this is the annotated input schema defined on the underlying component.
    - For regular Tools, this is the function signature of the `function` callable, falling back to `async_function`
      for async-only tools.

    :param tool: The tool to inspect.
    :returns: A dict mapping parameter names to their type annotations.
    """
    if isinstance(tool, ComponentTool):
        assert hasattr(tool._component, "__haystack_input__") and isinstance(
            tool._component.__haystack_input__, Sockets
        )
        return {name: socket.type for name, socket in tool._component.__haystack_input__._sockets_dict.items()}
    # Tool.__post_init__ guarantees that at least one of `function` / `async_function` is set.
    target = tool.function if tool.function is not None else tool.async_function
    return {name: param.annotation for name, param in inspect.signature(target).parameters.items()}  # type: ignore[arg-type]


def _inject_state_args(tool: Tool, llm_args: dict[str, Any], state: State) -> dict[str, Any]:
    """
    Merge LLM-provided arguments with state-sourced arguments.

    LLM args take precedence. State values are pulled in via `inputs_from_state` mappings or parameter-name matching,
    then the live State object is injected for any param annotated as State.

    :param tool: The tool being invoked, used to determine parameter mappings and State injection.
    :param llm_args: The arguments provided by the LLM, which take precedence over state values.
    :param state: The current runtime state, used to source additional arguments as needed.
    :returns: A dict of arguments to invoke the tool with, combining LLM and state values according to the rules
        described above.
    """
    final_args = dict(llm_args)
    func_params = _get_func_params(tool)

    param_mappings: dict[str, str]
    if hasattr(tool, "inputs_from_state") and isinstance(tool.inputs_from_state, dict):
        param_mappings = tool.inputs_from_state
    else:
        param_mappings = {name: name for name in func_params}

    for state_key, param_name in param_mappings.items():
        if param_name not in final_args and state.has(state_key):
            final_args[param_name] = state.get(state_key)

    for param_name, param_type in func_params.items():
        if _unwrap_optional(param_type) is State:
            final_args[param_name] = state

    return final_args


def _prepare_tool_args(
    *,
    tool: Tool,
    tool_call_arguments: dict[str, Any],
    state: State,
    streaming_callback: StreamingCallbackT | None = None,
    enable_streaming_passthrough: bool = False,
) -> dict[str, Any]:
    """
    Prepare the final arguments for a tool by injecting state inputs and optionally a streaming callback.

    :param tool:
        The tool instance to prepare arguments for.
    :param tool_call_arguments:
        The initial arguments provided for the tool call.
    :param state:
        The current state containing inputs to be injected into the tool arguments.
    :param streaming_callback:
        Optional streaming callback to be injected if enabled and applicable.
    :param enable_streaming_passthrough:
        Flag indicating whether to inject the streaming callback into the tool arguments.

    :returns:
        A dictionary of final arguments ready for tool invocation.
    """
    # Combine user + state inputs
    final_args = _inject_state_args(tool, tool_call_arguments.copy(), state)
    # Check whether to inject streaming_callback
    if (
        enable_streaming_passthrough
        and streaming_callback is not None
        and "streaming_callback" not in final_args
        and "streaming_callback" in _get_func_params(tool)
    ):
        final_args["streaming_callback"] = streaming_callback
    return final_args


def _collect_tool_call_params(
    messages_with_tool_calls: list[ChatMessage],
    state: State,
    tools_with_names: dict[str, Tool],
    streaming_callback: StreamingCallbackT | None,
    *,
    raise_on_failure: bool,
    enable_streaming_callback_passthrough: bool,
) -> tuple[list[ToolCall], list[dict[str, Any]], list[ChatMessage]]:
    """
    Walk all tool calls in `messages_with_tool_calls` and prepare their execution parameters.

    :returns: (tool_calls, tool_call_params, error_messages)
        - tool_calls: ToolCall objects for each valid call
        - tool_call_params: {"tool": ..., "args": ...} dicts ready for _make_context_bound_invoke
        - error_messages: ChatMessages for tool-not-found errors (when raise_on_failure is False)
    """
    tool_calls: list[ToolCall] = []
    tool_call_params: list[dict[str, Any]] = []
    error_messages: list[ChatMessage] = []

    for message in messages_with_tool_calls:
        for tool_call in message.tool_calls:
            tool_name = tool_call.tool_name

            if tool_name not in tools_with_names:
                error = ToolNotFoundException(tool_name, list(tools_with_names.keys()))
                if raise_on_failure:
                    raise error
                logger.error("{error_exception}", error_exception=error)
                error_messages.append(ChatMessage.from_tool(tool_result=str(error), origin=tool_call, error=True))
                continue

            tool = tools_with_names[tool_name]

            final_args = _prepare_tool_args(
                tool=tool,
                tool_call_arguments=tool_call.arguments,
                state=state,
                streaming_callback=streaming_callback,
                enable_streaming_passthrough=enable_streaming_callback_passthrough,
            )

            tool_calls.append(tool_call)
            tool_call_params.append({"tool": tool, "args": final_args})

    return tool_calls, tool_call_params, error_messages


def _run_tool(
    *,
    messages: list[ChatMessage],
    state: State,
    tools: ToolsType,
    streaming_callback: StreamingCallbackT | None = None,
    raise_on_failure: bool = True,
    enable_streaming_callback_passthrough: bool = False,
    max_workers: int = 4,
) -> tuple[list[ChatMessage], State]:
    """
    Invoke all tools referenced by tool calls in `messages`.

    :param messages: ChatMessage objects that may contain tool calls.
    :param state: Runtime state passed to and updated by tools.
    :param tools: The tools available for invocation.
    :param streaming_callback: Called once per tool result as it becomes available.
    :param raise_on_failure: If True, raise on tool invocation failure; otherwise return an error message.
    :param enable_streaming_callback_passthrough: If True, pass the streaming callback to tools that accept it.
    :param max_workers: Maximum number of parallel tool invocations.
    :returns: (tool_messages, updated_state)
    """
    tools_with_names = _validate_and_prepare_tools(tools)

    messages_with_tool_calls = [m for m in messages if m.tool_calls]
    if not messages_with_tool_calls:
        return [], state

    tool_calls, tool_call_params, tool_messages = _collect_tool_call_params(
        messages_with_tool_calls=messages_with_tool_calls,
        state=state,
        tools_with_names=tools_with_names,
        streaming_callback=streaming_callback,
        raise_on_failure=raise_on_failure,
        enable_streaming_callback_passthrough=enable_streaming_callback_passthrough,
    )

    if not tool_call_params:
        return tool_messages, state

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_make_context_bound_invoke(p["tool"], p["args"])) for p in tool_call_params]

        for future, tool_call in zip(futures, tool_calls, strict=True):
            result = future.result()

            if isinstance(result, ToolInvocationError):
                if raise_on_failure:
                    raise result
                logger.error("{error_exception}", error_exception=result)
                tool_messages.append(ChatMessage.from_tool(tool_result=str(result), origin=tool_call, error=True))
            else:
                tool = tools_with_names[tool_call.tool_name]
                _merge_tool_outputs_into_state(tool, result, state)
                tool_messages.append(
                    _build_tool_result_message(result, tool_call, tool, raise_on_failure=raise_on_failure)
                )

            if streaming_callback is not None:
                streaming_callback(_create_tool_result_streaming_chunk(tool_messages, tool_call))

    # We emit a final empty chunk with finish_reason "tool_call_results" to signal the end of the tool results stream.
    if tool_messages and streaming_callback is not None:
        streaming_callback(
            StreamingChunk(content="", finish_reason="tool_call_results", meta={"finish_reason": "tool_call_results"})
        )

    return tool_messages, state


async def _run_tool_async(
    *,
    messages: list[ChatMessage],
    state: State,
    tools: ToolsType,
    streaming_callback: StreamingCallbackT | None = None,
    raise_on_failure: bool = True,
    enable_streaming_callback_passthrough: bool = False,
    max_workers: int = 4,
) -> tuple[list[ChatMessage], State]:
    """
    Asynchronous variant of `run_tool`. Tool calls execute concurrently via a thread pool.

    :param messages: ChatMessage objects that may contain tool calls.
    :param state: Runtime state passed to and updated by tools.
    :param tools: The tools available for invocation.
    :param streaming_callback: Async callback called once per tool result.
    :param raise_on_failure: If True, raise on tool invocation failure; otherwise return an error message.
    :param enable_streaming_callback_passthrough: If True, pass the streaming callback to tools that accept it.
    :param max_workers: Maximum number of parallel tool invocations.
    :returns: (tool_messages, updated_state)
    """
    tools_with_names = _validate_and_prepare_tools(tools)

    messages_with_tool_calls = [m for m in messages if m.tool_calls]
    if not messages_with_tool_calls:
        return [], state

    tool_calls, tool_call_params, tool_messages = _collect_tool_call_params(
        messages_with_tool_calls=messages_with_tool_calls,
        state=state,
        tools_with_names=tools_with_names,
        streaming_callback=streaming_callback,
        raise_on_failure=raise_on_failure,
        enable_streaming_callback_passthrough=enable_streaming_callback_passthrough,
    )

    if not tool_call_params:
        return tool_messages, state

    # `max_workers` + Semaphore bounds concurrency for both sync and async tool calls async tools are awaited directly,
    # and sync tools are dispatched to a worker thread inside `Tool.invoke_async`.
    semaphore = asyncio.Semaphore(max_workers)
    tasks = [_make_bounded_invoke_async(p["tool"], p["args"], semaphore)() for p in tool_call_params]
    results = await asyncio.gather(*tasks)

    for result, tool_call in zip(results, tool_calls, strict=True):
        if isinstance(result, ToolInvocationError):
            if raise_on_failure:
                raise result
            logger.error("{error_exception}", error_exception=result)
            tool_messages.append(ChatMessage.from_tool(tool_result=str(result), origin=tool_call, error=True))
        else:
            tool = tools_with_names[tool_call.tool_name]
            _merge_tool_outputs_into_state(tool, result, state)
            tool_messages.append(_build_tool_result_message(result, tool_call, tool, raise_on_failure=raise_on_failure))

        if streaming_callback is not None:
            await _invoke_streaming_callback(
                streaming_callback, _create_tool_result_streaming_chunk(tool_messages, tool_call)
            )

    # We emit a final empty chunk with finish_reason "tool_call_results" to signal the end of the tool results stream.
    if tool_messages and streaming_callback is not None:
        await _invoke_streaming_callback(
            streaming_callback,
            StreamingChunk(content="", finish_reason="tool_call_results", meta={"finish_reason": "tool_call_results"}),
        )

    return tool_messages, state
