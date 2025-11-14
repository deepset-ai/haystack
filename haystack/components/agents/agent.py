# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from dataclasses import dataclass
from typing import Any, Optional, Union, cast

from haystack import logging, tracing
from haystack.components.generators.chat.types import ChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.core.component.component import component
from haystack.core.errors import PipelineRuntimeError
from haystack.core.pipeline.async_pipeline import AsyncPipeline
from haystack.core.pipeline.breakpoint import (
    _create_pipeline_snapshot_from_chat_generator,
    _create_pipeline_snapshot_from_tool_invoker,
    _trigger_chat_generator_breakpoint,
    _trigger_tool_invoker_breakpoint,
    _validate_tool_breakpoint_is_valid,
)
from haystack.core.pipeline.pipeline import Pipeline
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.core.serialization import component_to_dict, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.dataclasses.breakpoints import AgentBreakpoint, AgentSnapshot, PipelineSnapshot, ToolBreakpoint
from haystack.dataclasses.streaming_chunk import StreamingCallbackT, select_streaming_callback
from haystack.tools import (
    Tool,
    Toolset,
    ToolsType,
    deserialize_tools_or_toolset_inplace,
    flatten_tools_or_toolsets,
    serialize_tools_or_toolset,
)
from haystack.utils import _deserialize_value_with_schema
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable
from haystack.utils.deserialization import deserialize_chatgenerator_inplace

from .state.state import State, _schema_from_dict, _schema_to_dict, _validate_schema
from .state.state_utils import merge_lists

logger = logging.getLogger(__name__)


@dataclass
class _ExecutionContext:
    """
    Context for executing the agent.

    :param state: The current state of the agent, including messages and any additional data.
    :param component_visits: A dictionary tracking how many times each component has been visited.
    :param chat_generator_inputs: Runtime inputs to be passed to the chat generator.
    :param tool_invoker_inputs: Runtime inputs to be passed to the tool invoker.
    :param counter: A counter to track the number of steps taken in the agent's run.
    :param skip_chat_generator: A flag to indicate whether to skip the chat generator in the next iteration.
        This is useful when resuming from a ToolBreakpoint where the ToolInvoker needs to be called first.
    """

    state: State
    component_visits: dict
    chat_generator_inputs: dict
    tool_invoker_inputs: dict
    counter: int = 0
    skip_chat_generator: bool = False


@component
class Agent:
    """
    A Haystack component that implements a tool-using agent with provider-agnostic chat model support.

    The component processes messages and executes tools until an exit condition is met.
    The exit condition can be triggered either by a direct text response or by invoking a specific designated tool.
    Multiple exit conditions can be specified.

    When you call an Agent without tools, it acts as a ChatGenerator, produces one response, then exits.

    ### Usage example
    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools import Tool

    # Tool functions - in practice, these would have real implementations
    def search(query: str) -> str:
        '''Search for information on the web.'''
        # Placeholder: would call actual search API
        return "In France, a 15% service charge is typically included, but leaving 5-10% extra is appreciated."

    def calculator(operation: str, a: float, b: float) -> float:
        '''Perform mathematical calculations.'''
        if operation == "multiply":
            return a * b
        elif operation == "percentage":
            return (a / 100) * b
        return 0

    # Define tools with JSON Schema
    tools = [
        Tool(
            name="search",
            description="Searches for information on the web",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"}
                },
                "required": ["query"]
            },
            function=search
        ),
        Tool(
            name="calculator",
            description="Performs mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "description": "Operation: multiply, percentage"},
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["operation", "a", "b"]
            },
            function=calculator
        )
    ]

    # Create and run the agent
    agent = Agent(
        chat_generator=OpenAIChatGenerator(),
        tools=tools
    )

    result = agent.run(
        messages=[ChatMessage.from_user("Calculate the appropriate tip for an €85 meal in France")]
    )

    # The agent will:
    # 1. Search for tipping customs in France
    # 2. Use calculator to compute tip based on findings
    # 3. Return the final answer with context
    print(result["messages"][-1].text)
    ```

    """

    def __init__(
        self,
        *,
        chat_generator: ChatGenerator,
        tools: Optional[ToolsType] = None,
        system_prompt: Optional[str] = None,
        exit_conditions: Optional[list[str]] = None,
        state_schema: Optional[dict[str, Any]] = None,
        max_agent_steps: int = 100,
        streaming_callback: Optional[StreamingCallbackT] = None,
        raise_on_tool_invocation_failure: bool = False,
        tool_invoker_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the agent component.

        :param chat_generator: An instance of the chat generator that your agent should use. It must support tools.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset that the agent can use.
        :param system_prompt: System prompt for the agent.
        :param exit_conditions: List of conditions that will cause the agent to return.
            Can include "text" if the agent should return when it generates a message without tool calls,
            or tool names that will cause the agent to return once the tool was executed. Defaults to ["text"].
        :param state_schema: The schema for the runtime state used by the tools.
        :param max_agent_steps: Maximum number of steps the agent will run before stopping. Defaults to 100.
            If the agent exceeds this number of steps, it will stop and return the current state.
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
            The same callback can be configured to emit tool results when a tool is called.
        :param raise_on_tool_invocation_failure: Should the agent raise an exception when a tool invocation fails?
            If set to False, the exception will be turned into a chat message and passed to the LLM.
        :param tool_invoker_kwargs: Additional keyword arguments to pass to the ToolInvoker.
        :raises TypeError: If the chat_generator does not support tools parameter in its run method.
        :raises ValueError: If the exit_conditions are not valid.
        """
        # Check if chat_generator supports tools parameter
        chat_generator_run_method = inspect.signature(chat_generator.run)
        if "tools" not in chat_generator_run_method.parameters:
            raise TypeError(
                f"{type(chat_generator).__name__} does not accept tools parameter in its run method. "
                "The Agent component requires a chat generator that supports tools."
            )

        valid_exits = ["text"] + [tool.name for tool in flatten_tools_or_toolsets(tools)]
        if exit_conditions is None:
            exit_conditions = ["text"]
        if not all(condition in valid_exits for condition in exit_conditions):
            raise ValueError(
                f"Invalid exit conditions provided: {exit_conditions}. "
                f"Valid exit conditions must be a subset of {valid_exits}. "
                "Ensure that each exit condition corresponds to either 'text' or a valid tool name."
            )

        # Validate state schema if provided
        if state_schema is not None:
            _validate_schema(state_schema)
        self._state_schema = state_schema or {}

        # Initialize state schema
        resolved_state_schema = _deepcopy_with_exceptions(self._state_schema)
        if resolved_state_schema.get("messages") is None:
            resolved_state_schema["messages"] = {"type": list[ChatMessage], "handler": merge_lists}
        self.state_schema = resolved_state_schema

        self.chat_generator = chat_generator
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.exit_conditions = exit_conditions
        self.max_agent_steps = max_agent_steps
        self.raise_on_tool_invocation_failure = raise_on_tool_invocation_failure
        self.streaming_callback = streaming_callback

        output_types = {"last_message": ChatMessage}
        for param, config in self.state_schema.items():
            output_types[param] = config["type"]
            # Skip setting input types for parameters that are already in the run method
            if param in ["messages", "streaming_callback"]:
                continue
            component.set_input_type(self, name=param, type=config["type"], default=None)
        component.set_output_types(self, **output_types)

        self.tool_invoker_kwargs = tool_invoker_kwargs
        self._tool_invoker = None
        if self.tools:
            resolved_tool_invoker_kwargs = {
                "tools": self.tools,
                "raise_on_failure": self.raise_on_tool_invocation_failure,
                **(tool_invoker_kwargs or {}),
            }
            self._tool_invoker = ToolInvoker(**resolved_tool_invoker_kwargs)
        else:
            logger.warning(
                "No tools provided to the Agent. The Agent will behave like a ChatGenerator and only return text "
                "responses. To enable tool usage, pass tools directly to the Agent, not to the chat_generator."
            )

        self._is_warmed_up = False

    def warm_up(self) -> None:
        """
        Warm up the Agent.
        """
        if not self._is_warmed_up:
            if hasattr(self.chat_generator, "warm_up"):
                self.chat_generator.warm_up()
            if hasattr(self._tool_invoker, "warm_up") and self._tool_invoker is not None:
                self._tool_invoker.warm_up()
            self._is_warmed_up = True

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :return: Dictionary with serialized data
        """
        return default_to_dict(
            self,
            chat_generator=component_to_dict(obj=self.chat_generator, name="chat_generator"),
            tools=serialize_tools_or_toolset(self.tools),
            system_prompt=self.system_prompt,
            exit_conditions=self.exit_conditions,
            # We serialize the original state schema, not the resolved one to reflect the original user input
            state_schema=_schema_to_dict(self._state_schema),
            max_agent_steps=self.max_agent_steps,
            streaming_callback=serialize_callable(self.streaming_callback) if self.streaming_callback else None,
            raise_on_tool_invocation_failure=self.raise_on_tool_invocation_failure,
            tool_invoker_kwargs=self.tool_invoker_kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Agent":
        """
        Deserialize the agent from a dictionary.

        :param data: Dictionary to deserialize from
        :return: Deserialized agent
        """
        init_params = data.get("init_parameters", {})

        deserialize_chatgenerator_inplace(init_params, key="chat_generator")

        if init_params.get("state_schema") is not None:
            init_params["state_schema"] = _schema_from_dict(init_params["state_schema"])

        if init_params.get("streaming_callback") is not None:
            init_params["streaming_callback"] = deserialize_callable(init_params["streaming_callback"])

        deserialize_tools_or_toolset_inplace(init_params, key="tools")

        return default_from_dict(cls, data)

    def _create_agent_span(self) -> Any:
        """Create a span for the agent run."""
        return tracing.tracer.trace(
            "haystack.agent.run",
            tags={
                "haystack.agent.max_steps": self.max_agent_steps,
                "haystack.agent.tools": self.tools,
                "haystack.agent.exit_conditions": self.exit_conditions,
                "haystack.agent.state_schema": _schema_to_dict(self.state_schema),
            },
        )

    def _initialize_fresh_execution(
        self,
        messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT],
        requires_async: bool,
        *,
        system_prompt: Optional[str] = None,
        generation_kwargs: Optional[dict[str, Any]] = None,
        tools: Optional[Union[ToolsType, list[str]]] = None,
        **kwargs,
    ) -> _ExecutionContext:
        """
        Initialize execution context for a fresh run of the agent.

        :param messages: List of ChatMessage objects to start the agent with.
        :param streaming_callback: Optional callback for streaming responses.
        :param requires_async: Whether the agent run requires asynchronous execution.
        :param system_prompt: System prompt for the agent. If provided, it overrides the default system prompt.
        :param generation_kwargs: Additional keyword arguments for chat generator. These parameters will
            override the parameters passed during component initialization.
        :param tools: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
            When passing tool names, tools are selected from the Agent's originally configured tools.
        :param kwargs: Additional data to pass to the State used by the Agent.
        """
        system_prompt = system_prompt or self.system_prompt
        if system_prompt is not None:
            messages = [ChatMessage.from_system(system_prompt)] + messages

        if all(m.is_from(ChatRole.SYSTEM) for m in messages):
            logger.warning("All messages provided to the Agent component are system messages. This is not recommended.")

        state = State(schema=self.state_schema, data=kwargs)
        state.set("messages", messages)

        streaming_callback = select_streaming_callback(  # type: ignore[call-overload]
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=requires_async
        )

        selected_tools = self._select_tools(tools)
        tool_invoker_inputs: dict[str, Any] = {"tools": selected_tools}
        generator_inputs: dict[str, Any] = {"tools": selected_tools}
        if streaming_callback is not None:
            tool_invoker_inputs["streaming_callback"] = streaming_callback
            generator_inputs["streaming_callback"] = streaming_callback
        if generation_kwargs is not None:
            generator_inputs["generation_kwargs"] = generation_kwargs

        return _ExecutionContext(
            state=state,
            component_visits=dict.fromkeys(["chat_generator", "tool_invoker"], 0),
            chat_generator_inputs=generator_inputs,
            tool_invoker_inputs=tool_invoker_inputs,
        )

    def _select_tools(self, tools: Optional[Union[ToolsType, list[str]]] = None) -> ToolsType:
        """
        Select tools for the current run based on the provided tools parameter.

        :param tools: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
            When passing tool names, tools are selected from the Agent's originally configured tools.
        :returns: Selected tools for the current run.
        :raises ValueError: If tool names are provided but no tools were configured at initialization,
            or if any provided tool name is not valid.
        :raises TypeError: If tools is not a list of Tool objects, a Toolset, or a list of tool names (strings).
        """
        if tools is None:
            return self.tools

        if isinstance(tools, list) and all(isinstance(t, str) for t in tools):
            if not self.tools:
                raise ValueError("No tools were configured for the Agent at initialization.")
            available_tools = flatten_tools_or_toolsets(self.tools)
            selected_tool_names = cast(list[str], tools)  # mypy thinks this could still be list[Tool] or Toolset
            valid_tool_names = {tool.name for tool in available_tools}
            invalid_tool_names = {name for name in selected_tool_names if name not in valid_tool_names}
            if invalid_tool_names:
                raise ValueError(
                    f"The following tool names are not valid: {invalid_tool_names}. "
                    f"Valid tool names are: {valid_tool_names}."
                )
            return [tool for tool in available_tools if tool.name in selected_tool_names]

        if isinstance(tools, Toolset):
            return tools

        if isinstance(tools, list):
            return cast(list[Union[Tool, Toolset]], tools)  # mypy can't narrow the Union type from isinstance check

        raise TypeError(
            "tools must be a list of Tool and/or Toolset objects, a Toolset, or a list of tool names (strings)."
        )

    def _initialize_from_snapshot(
        self,
        snapshot: AgentSnapshot,
        streaming_callback: Optional[StreamingCallbackT],
        requires_async: bool,
        *,
        generation_kwargs: Optional[dict[str, Any]] = None,
        tools: Optional[Union[ToolsType, list[str]]] = None,
    ) -> _ExecutionContext:
        """
        Initialize execution context from an AgentSnapshot.

        :param snapshot: An AgentSnapshot containing the state of a previously saved agent execution.
        :param streaming_callback: Optional callback for streaming responses.
        :param requires_async: Whether the agent run requires asynchronous execution.
        :param generation_kwargs: Additional keyword arguments for chat generator. These parameters will
            override the parameters passed during component initialization.
        :param tools: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
            When passing tool names, tools are selected from the Agent's originally configured tools.
        """
        component_visits = snapshot.component_visits
        current_inputs = {
            "chat_generator": _deserialize_value_with_schema(snapshot.component_inputs["chat_generator"]),
            "tool_invoker": _deserialize_value_with_schema(snapshot.component_inputs["tool_invoker"]),
        }

        state_data = current_inputs["tool_invoker"]["state"].data
        state = State(schema=self.state_schema, data=state_data)

        skip_chat_generator = isinstance(snapshot.break_point.break_point, ToolBreakpoint)
        streaming_callback = current_inputs["chat_generator"].get("streaming_callback", streaming_callback)
        streaming_callback = select_streaming_callback(  # type: ignore[call-overload]
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=requires_async
        )

        selected_tools = self._select_tools(tools)
        tool_invoker_inputs: dict[str, Any] = {"tools": selected_tools}
        generator_inputs: dict[str, Any] = {"tools": selected_tools}
        if streaming_callback is not None:
            tool_invoker_inputs["streaming_callback"] = streaming_callback
            generator_inputs["streaming_callback"] = streaming_callback
        if generation_kwargs is not None:
            generator_inputs["generation_kwargs"] = generation_kwargs

        return _ExecutionContext(
            state=state,
            component_visits=component_visits,
            chat_generator_inputs=generator_inputs,
            tool_invoker_inputs=tool_invoker_inputs,
            counter=snapshot.break_point.break_point.visit_count,
            skip_chat_generator=skip_chat_generator,
        )

    def _runtime_checks(self, break_point: Optional[AgentBreakpoint]) -> None:
        """
        Perform runtime checks before running the agent.

        :param break_point: An AgentBreakpoint, can be a Breakpoint for the "chat_generator" or a ToolBreakpoint
            for "tool_invoker".
        :raises ValueError: If the break_point is invalid.
        """
        if not self._is_warmed_up:
            self.warm_up()

        if break_point and isinstance(break_point.break_point, ToolBreakpoint):
            _validate_tool_breakpoint_is_valid(agent_breakpoint=break_point, tools=self.tools)

    @staticmethod
    def _check_chat_generator_breakpoint(
        execution_context: _ExecutionContext,
        break_point: Optional[AgentBreakpoint],
        parent_snapshot: Optional[PipelineSnapshot],
    ) -> None:
        """
        Check if the chat generator breakpoint should be triggered.

        If the breakpoint should be triggered, create an agent snapshot and trigger the chat generator breakpoint.

        :param execution_context: The current execution context of the agent.
        :param break_point: An AgentBreakpoint, can be a Breakpoint for the "chat_generator" or a ToolBreakpoint
            for "tool_invoker".
        :param parent_snapshot: An optional parent snapshot for the agent execution.
        """
        if (
            break_point
            and break_point.break_point.component_name == "chat_generator"
            and execution_context.component_visits["chat_generator"] == break_point.break_point.visit_count
        ):
            pipeline_snapshot = _create_pipeline_snapshot_from_chat_generator(
                execution_context=execution_context, break_point=break_point, parent_snapshot=parent_snapshot
            )
            _trigger_chat_generator_breakpoint(pipeline_snapshot=pipeline_snapshot)

    @staticmethod
    def _check_tool_invoker_breakpoint(
        execution_context: _ExecutionContext,
        break_point: Optional[AgentBreakpoint],
        parent_snapshot: Optional[PipelineSnapshot],
    ) -> None:
        """
        Check if the tool invoker breakpoint should be triggered.

        If the breakpoint should be triggered, create an agent snapshot and trigger the tool invoker breakpoint.

        :param execution_context: The current execution context of the agent.
        :param break_point: An AgentBreakpoint, can be a Breakpoint for the "chat_generator" or a ToolBreakpoint
            for "tool_invoker".
        :param parent_snapshot: An optional parent snapshot for the agent execution.
        """
        if (
            break_point
            and break_point.break_point.component_name == "tool_invoker"
            and break_point.break_point.visit_count == execution_context.component_visits["tool_invoker"]
        ):
            pipeline_snapshot = _create_pipeline_snapshot_from_tool_invoker(
                execution_context=execution_context, break_point=break_point, parent_snapshot=parent_snapshot
            )
            _trigger_tool_invoker_breakpoint(
                llm_messages=execution_context.state.data["messages"][-1:], pipeline_snapshot=pipeline_snapshot
            )

    def run(  # noqa: PLR0915
        self,
        messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        *,
        generation_kwargs: Optional[dict[str, Any]] = None,
        break_point: Optional[AgentBreakpoint] = None,
        snapshot: Optional[AgentSnapshot] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Union[ToolsType, list[str]]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Process messages and execute tools until an exit condition is met.

        :param messages: List of Haystack ChatMessage objects to process.
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
            The same callback can be configured to emit tool results when a tool is called.
        :param generation_kwargs: Additional keyword arguments for LLM. These parameters will
            override the parameters passed during component initialization.
        :param break_point: An AgentBreakpoint, can be a Breakpoint for the "chat_generator" or a ToolBreakpoint
            for "tool_invoker".
        :param snapshot: A dictionary containing a snapshot of a previously saved agent execution. The snapshot contains
            the relevant information to restart the Agent execution from where it left off.
        :param system_prompt: System prompt for the agent. If provided, it overrides the default system prompt.
        :param tools: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
            When passing tool names, tools are selected from the Agent's originally configured tools.
        :param kwargs: Additional data to pass to the State schema used by the Agent.
            The keys must match the schema defined in the Agent's `state_schema`.
        :returns:
            A dictionary with the following keys:
            - "messages": List of all messages exchanged during the agent's run.
            - "last_message": The last message exchanged during the agent's run.
            - Any additional keys defined in the `state_schema`.
        :raises RuntimeError: If the Agent component wasn't warmed up before calling `run()`.
        :raises BreakpointException: If an agent breakpoint is triggered.
        """
        # We pop parent_snapshot from kwargs to avoid passing it into State.
        parent_snapshot = kwargs.pop("parent_snapshot", None)
        agent_inputs = {
            "messages": messages,
            "streaming_callback": streaming_callback,
            "break_point": break_point,
            "snapshot": snapshot,
            **kwargs,
        }
        self._runtime_checks(break_point=break_point)

        if snapshot:
            exe_context = self._initialize_from_snapshot(
                snapshot=snapshot,
                streaming_callback=streaming_callback,
                requires_async=False,
                tools=tools,
                generation_kwargs=generation_kwargs,
            )
        else:
            exe_context = self._initialize_fresh_execution(
                messages=messages,
                streaming_callback=streaming_callback,
                requires_async=False,
                system_prompt=system_prompt,
                tools=tools,
                generation_kwargs=generation_kwargs,
                **kwargs,
            )

        with self._create_agent_span() as span:
            span.set_content_tag("haystack.agent.input", _deepcopy_with_exceptions(agent_inputs))

            while exe_context.counter < self.max_agent_steps:
                # Handle breakpoint and ChatGenerator call
                Agent._check_chat_generator_breakpoint(
                    execution_context=exe_context, break_point=break_point, parent_snapshot=parent_snapshot
                )
                # We skip the chat generator when restarting from a snapshot from a ToolBreakpoint
                if exe_context.skip_chat_generator:
                    llm_messages = exe_context.state.get("messages", [])[-1:]
                    # Set to False so the next iteration will call the chat generator
                    exe_context.skip_chat_generator = False
                else:
                    try:
                        result = Pipeline._run_component(
                            component_name="chat_generator",
                            component={"instance": self.chat_generator},
                            inputs={
                                "messages": exe_context.state.data["messages"],
                                **exe_context.chat_generator_inputs,
                            },
                            component_visits=exe_context.component_visits,
                            parent_span=span,
                        )
                    except PipelineRuntimeError as e:
                        pipeline_snapshot = _create_pipeline_snapshot_from_chat_generator(
                            agent_name=getattr(self, "__component_name__", None),
                            execution_context=exe_context,
                            parent_snapshot=parent_snapshot,
                        )
                        e.pipeline_snapshot = pipeline_snapshot
                        raise e

                    llm_messages = result["replies"]
                    exe_context.state.set("messages", llm_messages)

                # Check if any of the LLM responses contain a tool call or if the LLM is not using tools
                if not any(msg.tool_call for msg in llm_messages) or self._tool_invoker is None:
                    exe_context.counter += 1
                    break

                # Handle breakpoint and ToolInvoker call
                Agent._check_tool_invoker_breakpoint(
                    execution_context=exe_context, break_point=break_point, parent_snapshot=parent_snapshot
                )
                try:
                    # We only send the messages from the LLM to the tool invoker
                    tool_invoker_result = Pipeline._run_component(
                        component_name="tool_invoker",
                        component={"instance": self._tool_invoker},
                        inputs={
                            "messages": llm_messages,
                            "state": exe_context.state,
                            **exe_context.tool_invoker_inputs,
                        },
                        component_visits=exe_context.component_visits,
                        parent_span=span,
                    )
                except PipelineRuntimeError as e:
                    # Access the original Tool Invoker exception
                    original_error = e.__cause__
                    tool_name = getattr(original_error, "tool_name", None)

                    pipeline_snapshot = _create_pipeline_snapshot_from_tool_invoker(
                        tool_name=tool_name,
                        agent_name=getattr(self, "__component_name__", None),
                        execution_context=exe_context,
                        parent_snapshot=parent_snapshot,
                    )
                    e.pipeline_snapshot = pipeline_snapshot
                    raise e

                tool_messages = tool_invoker_result["tool_messages"]
                exe_context.state = tool_invoker_result["state"]
                exe_context.state.set("messages", tool_messages)

                # Check if any LLM message's tool call name matches an exit condition
                if self.exit_conditions != ["text"] and self._check_exit_conditions(llm_messages, tool_messages):
                    exe_context.counter += 1
                    break

                # Increment the step counter
                exe_context.counter += 1

            if exe_context.counter >= self.max_agent_steps:
                logger.warning(
                    "Agent reached maximum agent steps of {max_agent_steps}, stopping.",
                    max_agent_steps=self.max_agent_steps,
                )
            span.set_content_tag("haystack.agent.output", exe_context.state.data)
            span.set_tag("haystack.agent.steps_taken", exe_context.counter)

        result = {**exe_context.state.data}
        if msgs := result.get("messages"):
            result["last_message"] = msgs[-1]
        return result

    async def run_async(
        self,
        messages: list[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        *,
        generation_kwargs: Optional[dict[str, Any]] = None,
        break_point: Optional[AgentBreakpoint] = None,
        snapshot: Optional[AgentSnapshot] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[Union[ToolsType, list[str]]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Asynchronously process messages and execute tools until the exit condition is met.

        This is the asynchronous version of the `run` method. It follows the same logic but uses
        asynchronous operations where possible, such as calling the `run_async` method of the ChatGenerator
        if available.

        :param messages: List of Haystack ChatMessage objects to process.
        :param streaming_callback: An asynchronous callback that will be invoked when a response is streamed from the
            LLM. The same callback can be configured to emit tool results when a tool is called.
        :param generation_kwargs: Additional keyword arguments for LLM. These parameters will
            override the parameters passed during component initialization.
        :param break_point: An AgentBreakpoint, can be a Breakpoint for the "chat_generator" or a ToolBreakpoint
            for "tool_invoker".
        :param snapshot: A dictionary containing a snapshot of a previously saved agent execution. The snapshot contains
            the relevant information to restart the Agent execution from where it left off.
        :param system_prompt: System prompt for the agent. If provided, it overrides the default system prompt.
        :param tools: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
        :param kwargs: Additional data to pass to the State schema used by the Agent.
            The keys must match the schema defined in the Agent's `state_schema`.
        :returns:
            A dictionary with the following keys:
            - "messages": List of all messages exchanged during the agent's run.
            - "last_message": The last message exchanged during the agent's run.
            - Any additional keys defined in the `state_schema`.
        :raises RuntimeError: If the Agent component wasn't warmed up before calling `run_async()`.
        :raises BreakpointException: If an agent breakpoint is triggered.
        """
        # We pop parent_snapshot from kwargs to avoid passing it into State.
        parent_snapshot = kwargs.pop("parent_snapshot", None)
        agent_inputs = {
            "messages": messages,
            "streaming_callback": streaming_callback,
            "break_point": break_point,
            "snapshot": snapshot,
            **kwargs,
        }
        self._runtime_checks(break_point=break_point)

        if snapshot:
            exe_context = self._initialize_from_snapshot(
                snapshot=snapshot,
                streaming_callback=streaming_callback,
                requires_async=True,
                tools=tools,
                generation_kwargs=generation_kwargs,
            )
        else:
            exe_context = self._initialize_fresh_execution(
                messages=messages,
                streaming_callback=streaming_callback,
                requires_async=True,
                system_prompt=system_prompt,
                tools=tools,
                generation_kwargs=generation_kwargs,
                **kwargs,
            )

        with self._create_agent_span() as span:
            span.set_content_tag("haystack.agent.input", _deepcopy_with_exceptions(agent_inputs))

            while exe_context.counter < self.max_agent_steps:
                # Handle breakpoint and ChatGenerator call
                self._check_chat_generator_breakpoint(
                    execution_context=exe_context, break_point=break_point, parent_snapshot=parent_snapshot
                )
                # We skip the chat generator when restarting from a snapshot from a ToolBreakpoint
                if exe_context.skip_chat_generator:
                    llm_messages = exe_context.state.get("messages", [])[-1:]
                    # Set to False so the next iteration will call the chat generator
                    exe_context.skip_chat_generator = False
                else:
                    result = await AsyncPipeline._run_component_async(
                        component_name="chat_generator",
                        component={"instance": self.chat_generator},
                        component_inputs={
                            "messages": exe_context.state.data["messages"],
                            **exe_context.chat_generator_inputs,
                        },
                        component_visits=exe_context.component_visits,
                        parent_span=span,
                    )
                    llm_messages = result["replies"]
                    exe_context.state.set("messages", llm_messages)

                # Check if any of the LLM responses contain a tool call or if the LLM is not using tools
                if not any(msg.tool_call for msg in llm_messages) or self._tool_invoker is None:
                    exe_context.counter += 1
                    break

                # Handle breakpoint and ToolInvoker call
                self._check_tool_invoker_breakpoint(
                    execution_context=exe_context, break_point=break_point, parent_snapshot=parent_snapshot
                )
                # We only send the messages from the LLM to the tool invoker
                tool_invoker_result = await AsyncPipeline._run_component_async(
                    component_name="tool_invoker",
                    component={"instance": self._tool_invoker},
                    component_inputs={
                        "messages": llm_messages,
                        "state": exe_context.state,
                        **exe_context.tool_invoker_inputs,
                    },
                    component_visits=exe_context.component_visits,
                    parent_span=span,
                )
                tool_messages = tool_invoker_result["tool_messages"]
                exe_context.state = tool_invoker_result["state"]
                exe_context.state.set("messages", tool_messages)

                # Check if any LLM message's tool call name matches an exit condition
                if self.exit_conditions != ["text"] and self._check_exit_conditions(llm_messages, tool_messages):
                    exe_context.counter += 1
                    break

                # Increment the step counter
                exe_context.counter += 1

            if exe_context.counter >= self.max_agent_steps:
                logger.warning(
                    "Agent reached maximum agent steps of {max_agent_steps}, stopping.",
                    max_agent_steps=self.max_agent_steps,
                )
            span.set_content_tag("haystack.agent.output", exe_context.state.data)
            span.set_tag("haystack.agent.steps_taken", exe_context.counter)

        result = {**exe_context.state.data}
        if msgs := result.get("messages"):
            result["last_message"] = msgs[-1]
        return result

    def _check_exit_conditions(self, llm_messages: list[ChatMessage], tool_messages: list[ChatMessage]) -> bool:
        """
        Check if any of the LLM messages' tool calls match an exit condition and if there are no errors.

        :param llm_messages: List of messages from the LLM
        :param tool_messages: List of messages from the tool invoker
        :return: True if an exit condition is met and there are no errors, False otherwise
        """
        matched_exit_conditions = set()
        has_errors = False

        for msg in llm_messages:
            if msg.tool_call and msg.tool_call.tool_name in self.exit_conditions:
                matched_exit_conditions.add(msg.tool_call.tool_name)

                # Check if any error is specifically from the tool matching the exit condition
                tool_errors = [
                    tool_msg.tool_call_result.error
                    for tool_msg in tool_messages
                    if tool_msg.tool_call_result is not None
                    and tool_msg.tool_call_result.origin.tool_name == msg.tool_call.tool_name
                ]
                if any(tool_errors):
                    has_errors = True
                    # No need to check further if we found an error
                    break

        # Only return True if at least one exit condition was matched AND none had errors
        return bool(matched_exit_conditions) and not has_errors
