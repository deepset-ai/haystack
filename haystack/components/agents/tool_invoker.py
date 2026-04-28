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

from haystack.components.agents.state.state import State
from haystack.core.component.sockets import Sockets
from haystack.core.serialization import logging
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.dataclasses.streaming_chunk import StreamingCallbackT, StreamingChunk
from haystack.tools import ComponentTool, Tool, ToolsType, _check_duplicate_tool_names, flatten_tools_or_toolsets
from haystack.tools.errors import ToolInvocationError
from haystack.tools.parameters_schema_utils import _unwrap_optional
from haystack.tracing.utils import _serializable_value

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ToolInvokerError(Exception):
    """Base exception class for ToolInvoker errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ToolNotFoundException(ToolInvokerError):
    """Exception raised when a tool is not found in the list of available tools."""

    def __init__(self, tool_name: str, available_tools: list[str]) -> None:
        message = f"Tool '{tool_name}' not found. Available tools: {', '.join(available_tools)}"
        super().__init__(message)


class StringConversionError(ToolInvokerError):
    """Exception raised when the conversion of a tool result to a string fails."""

    def __init__(self, tool_name: str, conversion_function: str, error: Exception) -> None:
        message = f"Failed to convert tool result from tool {tool_name} using '{conversion_function}'. Error: {error}"
        super().__init__(message)


class ResultConversionError(ToolInvokerError):
    """Exception raised when the conversion of a tool output to a result fails."""

    def __init__(self, tool_name: str, conversion_function: str, error: Exception) -> None:
        message = f"Failed to convert tool output from tool {tool_name} using '{conversion_function}'. Error: {error}"
        super().__init__(message)


class ToolOutputMergeError(ToolInvokerError):
    """Exception raised when merging tool outputs into state fails."""

    @classmethod
    def from_exception(cls, tool_name: str, error: Exception) -> "ToolOutputMergeError":
        """
        Create a ToolOutputMergeError with a message based on the original exception.

        :param tool_name: Name of the tool whose output merge failed.
        :param error: The original exception that was raised during the merge process.
        :returns: An instance of ToolOutputMergeError with a detailed error message.
        """
        message = f"Failed to merge tool outputs from tool {tool_name} into State: {error}"
        return cls(message)


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------


def _validate_and_prepare_tools(tools: ToolsType) -> dict[str, Tool]:
    """
    Flatten, deduplicate-check, and index tools by name.

    :raises ValueError: If no tools are provided or if duplicate tool names are found.
    """
    if not tools:
        raise ValueError("ToolInvoker requires at least one tool.")

    converted_tools = flatten_tools_or_toolsets(tools)
    _check_duplicate_tool_names(converted_tools)
    tool_names = [tool.name for tool in converted_tools]
    duplicates = {name for name in tool_names if tool_names.count(name) > 1}
    if duplicates:
        raise ValueError(f"Duplicate tool names found: {duplicates}")

    return dict(zip(tool_names, converted_tools, strict=True))


def _get_func_params(tool: Tool) -> dict[str, Any]:
    """Return parameter names → annotations for a tool's invocation function."""
    if isinstance(tool, ComponentTool):
        assert hasattr(tool._component, "__haystack_input__") and isinstance(
            tool._component.__haystack_input__, Sockets
        )
        return {name: socket.type for name, socket in tool._component.__haystack_input__._sockets_dict.items()}
    return {name: param.annotation for name, param in inspect.signature(tool.function).parameters.items()}


def _inject_state_args(tool: Tool, llm_args: dict[str, Any], state: State) -> dict[str, Any]:
    """
    Merge LLM-provided arguments with state-sourced arguments.

    LLM args take precedence. State values are pulled in via `inputs_from_state` mappings or
    parameter-name matching, then the live State object is injected for any param annotated as State.
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


def _merge_tool_outputs(tool: Tool, result: Any, state: State) -> None:
    """Write tool outputs into State according to the tool's `outputs_to_state` mapping."""
    if not isinstance(result, dict):
        return
    if not hasattr(tool, "outputs_to_state") or not isinstance(tool.outputs_to_state, dict):
        return

    for state_key, config in tool.outputs_to_state.items():
        source_key = config.get("source", None)
        if source_key and source_key not in result:
            continue
        output_value = result.get(source_key) if source_key else result
        state.set(state_key, output_value, handler_override=config.get("handler"))


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


def _process_tool_output(config: dict[str, Any], result: Any, tool_call: ToolCall) -> Any:
    """
    Extract and convert a single tool output according to `config`.

    `config` may contain `source` (key to extract from result dict), `handler` (conversion
    callable), and `raw_result` (return the value without string conversion).
    """
    source_key = config.get("source")
    value = result.get(source_key) if source_key is not None and isinstance(result, dict) else result

    handler = config.get("handler")
    raw_result = config.get("raw_result", False)

    if handler is None:
        if raw_result:
            return value
        return _result_to_string(value)

    try:
        return handler(value)
    except Exception as e:
        if raw_result:
            raise ResultConversionError(tool_call.tool_name, handler.__name__, e) from e
        raise StringConversionError(tool_call.tool_name, handler.__name__, e) from e


def _build_tool_result_message(result: Any, tool_call: ToolCall, tool: Tool, *, raise_on_failure: bool) -> ChatMessage:
    """Convert a raw tool result into a ChatMessage, applying `outputs_to_string` config if present."""
    outputs_config = tool.outputs_to_string or {}
    try:
        # Single-output config (or no config): keys are at the root level
        if not outputs_config or any(k in outputs_config for k in ("source", "handler", "raw_result")):
            tool_result = _process_tool_output(outputs_config, result, tool_call)
            return ChatMessage.from_tool(tool_result=tool_result, origin=tool_call)

        # Multi-output config: each key maps to its own sub-config — collect raw values then stringify once
        tool_result_dict = {
            output_key: _process_tool_output({**cfg, "raw_result": True}, result, tool_call)
            for output_key, cfg in outputs_config.items()
        }
        return ChatMessage.from_tool(tool_result=_result_to_string(tool_result_dict), origin=tool_call)
    except (StringConversionError, ResultConversionError) as e:
        if raise_on_failure:
            raise
        logger.exception("{error_exception}", error_exception=e)
        return ChatMessage.from_tool(tool_result=str(e), origin=tool_call, error=True)


def _create_tool_result_streaming_chunk(tool_messages: list[ChatMessage], tool_call: ToolCall) -> StreamingChunk:
    """Create a streaming chunk that carries the latest tool result."""
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
            args = _inject_state_args(tool, tool_call.arguments.copy(), state)

            if (
                enable_streaming_callback_passthrough
                and streaming_callback is not None
                and "streaming_callback" not in args
                and "streaming_callback" in _get_func_params(tool)
            ):
                args["streaming_callback"] = streaming_callback

            tool_calls.append(tool_call)
            tool_call_params.append({"tool": tool, "args": args})

    return tool_calls, tool_call_params, error_messages


# ---------------------------------------------------------------------------
# ToolInvoker — thin config shell + orchestrator
# ---------------------------------------------------------------------------


class ToolInvoker:
    """
    Orchestrates parallel tool invocation for the Agent.

    Holds configuration (failure mode, output conversion, worker count) and delegates
    all logic to module-level functions.  This class is an internal implementation
    detail of the `Agent` component and is not intended for standalone use.
    """

    def __init__(
        self,
        tools: ToolsType,
        raise_on_failure: bool = True,
        enable_streaming_callback_passthrough: bool = False,
        max_workers: int = 4,
    ) -> None:
        self.tools = tools
        self.raise_on_failure = raise_on_failure
        self.enable_streaming_callback_passthrough = enable_streaming_callback_passthrough
        self.max_workers = max_workers
        self._tools_with_names = _validate_and_prepare_tools(tools)

    def run(
        self,
        messages: list[ChatMessage],
        state: State,
        streaming_callback: StreamingCallbackT | None = None,
        tools: ToolsType | None = None,
    ) -> tuple[list[ChatMessage], State]:
        """
        Invoke all tools referenced by tool calls in `messages`.

        :param messages: ChatMessage objects that may contain tool calls.
        :param state: Runtime state passed to and updated by tools.
        :param streaming_callback: Called once per tool result as it becomes available.
        :param tools: Override the tool set for this invocation.
        :returns: (tool_messages, updated_state)
        """
        tools_with_names = _validate_and_prepare_tools(tools) if tools is not None else self._tools_with_names
        if tools is not None:
            logger.debug("Overriding tools for this invocation: {t}", t=", ".join(tools_with_names))

        messages_with_tool_calls = [m for m in messages if m.tool_calls]
        if not messages_with_tool_calls:
            return [], state

        tool_calls, tool_call_params, tool_messages = _collect_tool_call_params(
            messages_with_tool_calls,
            state,
            tools_with_names,
            streaming_callback,
            raise_on_failure=self.raise_on_failure,
            enable_streaming_callback_passthrough=self.enable_streaming_callback_passthrough,
        )

        if not tool_call_params:
            return tool_messages, state

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(_make_context_bound_invoke(p["tool"], p["args"])) for p in tool_call_params]

            for future, tool_call in zip(futures, tool_calls, strict=True):
                result = future.result()

                if isinstance(result, ToolInvocationError):
                    if self.raise_on_failure:
                        raise result
                    logger.error("{error_exception}", error_exception=result)
                    tool_messages.append(ChatMessage.from_tool(tool_result=str(result), origin=tool_call, error=True))
                else:
                    try:
                        tool = tools_with_names[tool_call.tool_name]
                        _merge_tool_outputs(tool, result, state)
                        tool_messages.append(
                            _build_tool_result_message(result, tool_call, tool, raise_on_failure=self.raise_on_failure)
                        )
                    except Exception as e:
                        error = ToolOutputMergeError.from_exception(tool_name=tool_call.tool_name, error=e)
                        if self.raise_on_failure:
                            raise error from e
                        logger.exception("{error_exception}", error_exception=error)
                        tool_messages.append(
                            ChatMessage.from_tool(tool_result=str(error), origin=tool_call, error=True)
                        )

                if streaming_callback is not None:
                    streaming_callback(_create_tool_result_streaming_chunk(tool_messages, tool_call))

        if tool_messages and streaming_callback is not None:
            streaming_callback(
                StreamingChunk(
                    content="", finish_reason="tool_call_results", meta={"finish_reason": "tool_call_results"}
                )
            )

        return tool_messages, state

    async def run_async(
        self,
        messages: list[ChatMessage],
        state: State,
        streaming_callback: StreamingCallbackT | None = None,
        tools: ToolsType | None = None,
    ) -> tuple[list[ChatMessage], State]:
        """
        Asynchronous variant of `run`. Tool calls execute concurrently via a thread pool.

        :param messages: ChatMessage objects that may contain tool calls.
        :param state: Runtime state passed to and updated by tools.
        :param streaming_callback: Async callback called once per tool result.
        :param tools: Override the tool set for this invocation.
        :returns: (tool_messages, updated_state)
        """
        tools_with_names = _validate_and_prepare_tools(tools) if tools is not None else self._tools_with_names
        if tools is not None:
            logger.debug("Overriding tools for this invocation: {t}", t=", ".join(tools_with_names))

        messages_with_tool_calls = [m for m in messages if m.tool_calls]
        if not messages_with_tool_calls:
            return [], state

        tool_calls, tool_call_params, tool_messages = _collect_tool_call_params(
            messages_with_tool_calls,
            state,
            tools_with_names,
            streaming_callback,
            raise_on_failure=self.raise_on_failure,
            enable_streaming_callback_passthrough=self.enable_streaming_callback_passthrough,
        )

        if not tool_call_params:
            return tool_messages, state

        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            tasks = [
                loop.run_in_executor(executor, _make_context_bound_invoke(p["tool"], p["args"]))
                for p in tool_call_params
            ]
            results = await asyncio.gather(*tasks)

            for result, tool_call in zip(results, tool_calls, strict=True):
                if isinstance(result, ToolInvocationError):
                    if self.raise_on_failure:
                        raise result
                    logger.error("{error_exception}", error_exception=result)
                    tool_messages.append(ChatMessage.from_tool(tool_result=str(result), origin=tool_call, error=True))
                else:
                    try:
                        tool = tools_with_names[tool_call.tool_name]
                        _merge_tool_outputs(tool, result, state)
                        tool_messages.append(
                            _build_tool_result_message(result, tool_call, tool, raise_on_failure=self.raise_on_failure)
                        )
                    except Exception as e:
                        error = ToolOutputMergeError.from_exception(tool_name=tool_call.tool_name, error=e)
                        if self.raise_on_failure:
                            raise error from e
                        logger.exception("{error_exception}", error_exception=error)
                        tool_messages.append(
                            ChatMessage.from_tool(tool_result=str(error), origin=tool_call, error=True)
                        )

                if streaming_callback is not None:
                    await streaming_callback(  # type: ignore[misc]
                        _create_tool_result_streaming_chunk(tool_messages, tool_call)
                    )

        if tool_messages and streaming_callback is not None:
            await streaming_callback(  # type: ignore[misc]
                StreamingChunk(
                    content="", finish_reason="tool_call_results", meta={"finish_reason": "tool_call_results"}
                )
            )

        return tool_messages, state
