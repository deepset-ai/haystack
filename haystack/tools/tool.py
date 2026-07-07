# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from haystack.core.serialization import generate_qualified_class_name
from haystack.tools.errors import ToolInvocationError
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable


@dataclass
class Tool:
    """
    Data class representing a Tool that Language Models can prepare a call for.

    Accurate definitions of the textual attributes such as `name` and `description`
    are important for the Language Model to correctly prepare the call.

    For resource-intensive operations like establishing connections to remote services or
    loading models, override the `warm_up()` method. This method is called before the Tool
    is used and should be idempotent, as it may be called multiple times during
    pipeline/agent setup.

    :param name:
        Name of the Tool.
    :param description:
        Description of the Tool.
    :param parameters:
        A JSON schema defining the parameters expected by the Tool.
    :param function:
        The synchronous function invoked by `Tool.invoke`. Must be a regular function — coroutine functions should
        be passed to `async_function` instead. Either `function` or `async_function` (or both) must be set.
    :param async_function:
        Optional coroutine function awaited by `Tool.invoke_async`. When only `async_function` is set, `invoke` raises
        a `ToolInvocationError`. When only `function` is set, `invoke_async` falls back to running `function` in a
        worker thread via `asyncio.to_thread`.
    :param outputs_to_string:
        Optional dictionary defining how tool outputs should be converted into string(s) or results.
        If not provided, the tool result is converted to a string using a default handler.

        `outputs_to_string` supports two formats:

        1. Single output format - use "source", "handler", and/or "raw_result" at the root level:
           ```python
           {
               "source": "docs", "handler": format_documents, "raw_result": False
           }
           ```
           - `source`: If provided, only the specified output key is sent to the handler. If not provided, the whole
              tool result is sent to the handler.
           - `handler`: A function that takes the tool output (or the extracted source value) and returns the
             final result.
           - `raw_result`: If `True`, the result is returned raw without string conversion, but applying the `handler`
             if provided. This is intended for tools that return images. In this mode, the Tool function or the
             `handler` must return a list of `TextContent`/`ImageContent` objects to ensure compatibility with Chat
             Generators.

        2. Multiple output format - map keys to individual configurations:
           ```python
           {
               "formatted_docs": {"source": "docs", "handler": format_documents},
               "summary": {"source": "summary_text", "handler": str.upper}
           }
           ```
           Each key maps to a dictionary that can contain "source" and/or "handler".
           Note that `raw_result` is not supported in the multiple output format.
    :param inputs_from_state:
        Optional dictionary mapping state keys to tool parameter names.
        Example: `{"repository": "repo"}` maps state's "repository" to tool's "repo" parameter.
    :param outputs_to_state:
        Optional dictionary defining how tool outputs map to keys within state as well as optional handlers.
        If the source is provided only the specified output key is sent to the handler.
        Example:
        ```python
        {
            "documents": {"source": "docs", "handler": custom_handler}
        }
        ```
        If the source is omitted the whole tool result is sent to the handler.
        Example:
        ```python
        {
            "documents": {"handler": custom_handler}
        }
        ```
    :raises ValueError: If neither `function` nor `async_function` is provided, if `function` is a
        coroutine function, if `async_function` is not a coroutine function, if `parameters` is not a
        valid JSON schema, or if the `outputs_to_state`, `outputs_to_string`, or `inputs_from_state`
        configurations are invalid.
    :raises TypeError: If any configuration value in `outputs_to_state`, `outputs_to_string`, or
        `inputs_from_state` has the wrong type.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable | None = None
    outputs_to_string: dict[str, Any] | None = None
    inputs_from_state: dict[str, str] | None = None
    outputs_to_state: dict[str, dict[str, Any]] | None = None
    async_function: Callable | None = None

    def __post_init__(self) -> None:  # noqa: C901, PLR0912
        # At least one of function / async_function must be set.
        if self.function is None and self.async_function is None:
            raise ValueError(f"Tool '{self.name}' requires at least one of `function` or `async_function` to be set.")

        # `function` must be a regular (sync) function. Coroutine functions belong on `async_function`.
        if self.function is not None and inspect.iscoroutinefunction(self.function):
            raise ValueError(
                f"`function` must be a synchronous function. "
                f"The function '{self.function.__name__}' is a coroutine function. "
                f"Pass it as `async_function` instead."
            )

        # `async_function` must be a coroutine function defined with `async def`.
        if self.async_function is not None and not inspect.iscoroutinefunction(self.async_function):
            raise ValueError(
                f"`async_function` must be a coroutine function defined with `async def`. "
                f"Got '{getattr(self.async_function, '__name__', repr(self.async_function))}'."
            )

        # Check that the parameters define a valid JSON schema
        try:
            Draft202012Validator.check_schema(self.parameters)
        except SchemaError as e:
            raise ValueError("The provided parameters do not define a valid JSON schema") from e

        # Validate outputs structure if provided
        if self.outputs_to_state is not None:
            for key, config in self.outputs_to_state.items():
                if not isinstance(config, dict):
                    raise TypeError(f"outputs_to_state configuration for key '{key}' must be a dictionary")
                if "source" in config and not isinstance(config["source"], str):
                    raise ValueError(f"outputs_to_state source for key '{key}' must be a string.")
                if "handler" in config and not callable(config["handler"]):
                    raise ValueError(f"outputs_to_state handler for key '{key}' must be callable")

            # Validate that outputs_to_state source keys exist as valid tool outputs
            valid_outputs: set[str] | None = self._get_valid_outputs()
            if valid_outputs is not None:
                for state_key, config in self.outputs_to_state.items():
                    source = config.get("source")
                    if source is not None and source not in valid_outputs:
                        raise ValueError(
                            f"outputs_to_state: '{self.name}' maps state key '{state_key}' to unknown output '{source}'"
                            f"Valid outputs are: {valid_outputs}."
                        )

        if self.outputs_to_string is not None:
            if "source" in self.outputs_to_string and not isinstance(self.outputs_to_string["source"], str):
                raise ValueError("outputs_to_string source must be a string.")
            if "handler" in self.outputs_to_string and not callable(self.outputs_to_string["handler"]):
                raise ValueError("outputs_to_string handler must be callable")
            if "raw_result" in self.outputs_to_string and not isinstance(self.outputs_to_string["raw_result"], bool):
                raise ValueError("outputs_to_string raw_result must be a boolean.")

            if (
                "source" in self.outputs_to_string
                or "handler" in self.outputs_to_string
                or "raw_result" in self.outputs_to_string
            ):
                # Single output configuration
                for key in self.outputs_to_string:
                    if key not in {"source", "handler", "raw_result"}:
                        raise ValueError(
                            "Invalid outputs_to_string config. "
                            "When using 'source', 'handler' or 'raw_result' at the root level, no other keys are "
                            " allowed. Use individual output configs instead."
                        )
            else:
                # Multiple outputs configuration
                for key, config in self.outputs_to_string.items():
                    if not isinstance(config, dict):
                        raise TypeError(f"outputs_to_string configuration for key '{key}' must be a dictionary")
                    if "raw_result" in config:
                        raise ValueError(
                            f"Invalid outputs_to_string configuration for key '{key}': "
                            f"'raw_result' is not supported in the multiple output format."
                        )
                    if "source" not in config:
                        raise ValueError(
                            f"Invalid outputs_to_string configuration for key '{key}': "
                            f"each output must have a 'source' defined."
                        )
                    if "source" in config and not isinstance(config["source"], str):
                        raise ValueError(f"outputs_to_string source for key '{key}' must be a string.")
                    if "handler" in config and not callable(config["handler"]):
                        raise ValueError(f"outputs_to_string handler for key '{key}' must be callable")

        # Validate that inputs_from_state parameter names exist as valid tool parameters
        if self.inputs_from_state is not None:
            valid_inputs = self._get_valid_inputs()
            for state_key, param_name in self.inputs_from_state.items():
                if not isinstance(param_name, str):
                    raise TypeError(
                        f"inputs_from_state values must be str, not {type(param_name).__name__}. "
                        f"Got {param_name!r} for key '{state_key}'."
                    )
                if valid_inputs and param_name not in valid_inputs:
                    raise ValueError(
                        f"inputs_from_state maps '{state_key}' to unknown parameter '{param_name}'. "
                        f"Valid parameters are: {valid_inputs}."
                    )

    def _get_valid_inputs(self) -> set[str]:
        """
        Return the set of valid input parameter names that this tool accepts.

        Used to validate that `inputs_from_state` only references parameters that actually exist.
        This prevents typos and catches configuration errors at tool construction time.

        By default, introspects the function signature to get ALL parameters, including those
        that may be excluded from the JSON schema (e.g., parameters mapped from state).
        Falls back to schema properties if introspection fails.

        Subclasses like ComponentTool override this to return component input socket names.

        :returns: Set of valid input parameter names for validation.
        """
        # Combine parameters from both function signature and schema for robustness
        # Function signature includes all parameters (even those excluded from schema)
        # Schema properties provide the validated parameter set
        valid_params: set[str] = set()

        # Try to get parameters from function introspection.
        # Prefer `function`; fall back to `async_function` for async-only tools.
        introspection_target = self.function if self.function is not None else self.async_function
        if introspection_target is not None:
            try:
                sig = inspect.signature(introspection_target)
                valid_params.update(sig.parameters.keys())
            except (ValueError, TypeError):
                pass  # Introspection failed, will rely on schema

        # Add parameters from schema (union with function params)
        valid_params.update(self.parameters.get("properties", {}).keys())

        return valid_params

    def _get_valid_outputs(self) -> set[str] | None:
        """
        Return the set of valid output names that this tool produces.

        Used to validate that `outputs_to_state` only references outputs that actually exist.
        This prevents typos and catches configuration errors at tool construction time.

        By default, returns None because regular function-based tools don't have a formal
        output schema. When None is returned, output validation is skipped.

        Subclasses like ComponentTool override this to return component output socket names,
        enabling validation for tools where outputs are known.

        :returns: Set of valid output names for validation, or None to skip validation.
        """
        return None

    @property
    def tool_spec(self) -> dict[str, Any]:
        """
        Return the Tool specification to be used by the Language Model.
        """
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def warm_up(self) -> None:
        """
        Prepare the Tool for use.

        Override this method to establish connections to remote services, load models,
        or perform other resource-intensive initialization. This method should be idempotent,
        as it may be called multiple times.
        """
        pass

    def invoke(self, **kwargs: Any) -> Any:
        """
        Invoke the Tool synchronously with the provided keyword arguments.

        :raises ToolInvocationError: If the Tool has no sync `function`, or if the underlying call
            raises an exception.
        """
        if self.function is None:
            raise ToolInvocationError(
                f"Tool `{self.name}` has no sync `function` and can only be invoked via `invoke_async` "
                f"(use `Agent.run_async`).",
                tool_name=self.name,
            )

        try:
            result = self.function(**kwargs)
        except Exception as e:
            raise ToolInvocationError(
                f"Failed to invoke Tool `{self.name}` with parameters {kwargs}. Error: {e}", tool_name=self.name
            ) from e
        return result

    async def invoke_async(self, **kwargs: Any) -> Any:
        """
        Invoke the Tool asynchronously with the provided keyword arguments.

        If `async_function` is set, it is awaited directly. Otherwise the sync `function` is dispatched to a worker
        thread via `asyncio.to_thread`, which propagates the current context to the worker.

        :raises ToolInvocationError: If the underlying call raises an exception.
        """
        try:
            if self.async_function is not None:
                return await self.async_function(**kwargs)
            # `function` is guaranteed to be set: __post_init__ enforces at least one of the two.
            return await asyncio.to_thread(self.function, **kwargs)  # type: ignore[arg-type]
        except Exception as e:
            raise ToolInvocationError(
                f"Failed to invoke Tool `{self.name}` with parameters {kwargs}. Error: {e}", tool_name=self.name
            ) from e

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the Tool to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        data = asdict(self)
        data["function"] = serialize_callable(self.function) if self.function is not None else None
        data["async_function"] = serialize_callable(self.async_function) if self.async_function is not None else None

        if self.outputs_to_state is not None:
            data["outputs_to_state"] = _serialize_outputs_to_state(self.outputs_to_state)

        if self.outputs_to_string is not None:
            data["outputs_to_string"] = _serialize_outputs_to_string(self.outputs_to_string)

        return {"type": generate_qualified_class_name(type(self)), "data": data}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Tool":
        """
        Deserializes the Tool from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized Tool.
        """
        init_parameters = data["data"]
        init_parameters["function"] = (
            deserialize_callable(init_parameters["function"]) if init_parameters.get("function") is not None else None
        )
        if init_parameters.get("async_function") is not None:
            init_parameters["async_function"] = deserialize_callable(init_parameters["async_function"])
        if "outputs_to_state" in init_parameters and init_parameters["outputs_to_state"]:
            init_parameters["outputs_to_state"] = _deserialize_outputs_to_state(init_parameters["outputs_to_state"])

        if init_parameters.get("outputs_to_string") is not None:
            init_parameters["outputs_to_string"] = _deserialize_outputs_to_string(init_parameters["outputs_to_string"])

        return cls(**init_parameters)


def _check_duplicate_tool_names(tools: list[Tool] | None) -> None:
    """
    Checks for duplicate tool names and raises a ValueError if they are found.

    :param tools: The list of tools to check.
    :raises ValueError: If duplicate tool names are found.
    """
    if tools is None:
        return
    tool_names = [tool.name for tool in tools]
    duplicate_tool_names = {name for name in tool_names if tool_names.count(name) > 1}
    if duplicate_tool_names:
        raise ValueError(f"Duplicate tool names found: {duplicate_tool_names}")


def _serialize_outputs_to_state(outputs_to_state: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Serializes the outputs_to_state dictionary, converting any callable handlers to their string representation.

    :param outputs_to_state: The outputs_to_state dictionary to serialize.
    :returns: The serialized outputs_to_state dictionary.
    """
    serialized_outputs = {}
    for key, config in outputs_to_state.items():
        serialized_config = config.copy()
        if "handler" in config:
            serialized_config["handler"] = serialize_callable(config["handler"])
        serialized_outputs[key] = serialized_config
    return serialized_outputs


def _deserialize_outputs_to_state(outputs_to_state: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """
    Deserializes the outputs_to_state dictionary, converting any string handlers back to callables.

    :param outputs_to_state: The outputs_to_state dictionary to deserialize.
    :returns: The deserialized outputs_to_state dictionary.
    """
    deserialized_outputs = {}
    for key, config in outputs_to_state.items():
        deserialized_config = config.copy()
        if "handler" in config:
            deserialized_config["handler"] = deserialize_callable(config["handler"])
        deserialized_outputs[key] = deserialized_config
    return deserialized_outputs


def _serialize_outputs_to_string(outputs_to_string: dict[str, Any]) -> dict[str, Any]:
    """
    Serializes the outputs_to_string dictionary, converting any callable handlers to their string representation.

    :param outputs_to_string: The outputs_to_string dictionary to serialize.
    :returns: The serialized outputs_to_string dictionary.
    """
    if "source" in outputs_to_string or "handler" in outputs_to_string or "raw_result" in outputs_to_string:
        # Single output configuration
        serialized_outputs = outputs_to_string.copy()
        if "handler" in outputs_to_string:
            serialized_outputs["handler"] = serialize_callable(outputs_to_string["handler"])
        return serialized_outputs

    # Multiple outputs configuration
    serialized_outputs = {}
    for key, config in outputs_to_string.items():
        serialized_config = config.copy()
        if "handler" in config:
            serialized_config["handler"] = serialize_callable(config["handler"])
        serialized_outputs[key] = serialized_config
    return serialized_outputs


def _deserialize_outputs_to_string(outputs_to_string: dict[str, Any]) -> dict[str, Any]:
    """
    Deserializes the outputs_to_string dictionary, converting any string handlers back to callables.

    :param outputs_to_string: The outputs_to_string dictionary to deserialize.
    :returns: The deserialized outputs_to_string dictionary.
    """
    if "source" in outputs_to_string or "handler" in outputs_to_string or "raw_result" in outputs_to_string:
        # Single output configuration
        deserialized_outputs = outputs_to_string.copy()
        if "handler" in outputs_to_string:
            deserialized_outputs["handler"] = deserialize_callable(outputs_to_string["handler"])
        return deserialized_outputs

    # Multiple outputs configuration
    deserialized_outputs = {}
    for key, config in outputs_to_string.items():
        deserialized_config = config.copy()
        if "handler" in config:
            deserialized_config["handler"] = deserialize_callable(config["handler"])
        deserialized_outputs[key] = deserialized_config
    return deserialized_outputs
