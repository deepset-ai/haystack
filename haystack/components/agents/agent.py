# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from dataclasses import dataclass
from typing import Any, Literal, cast

from haystack import Pipeline, component, logging, tracing
from haystack.components.agents.state.state import (
    State,
    _schema_from_dict,
    _schema_to_dict,
    _validate_schema,
    replace_values,
)
from haystack.components.agents.state.state_utils import merge_lists
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat.types import ChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.core.errors import BreakpointException, PipelineRuntimeError
from haystack.core.pipeline.async_pipeline import AsyncPipeline
from haystack.core.pipeline.breakpoint import (
    SnapshotCallback,
    _create_pipeline_snapshot_from_chat_generator,
    _create_pipeline_snapshot_from_tool_invoker,
    _save_pipeline_snapshot,
    _should_trigger_tool_invoker_breakpoint,
    _validate_tool_breakpoint_is_valid,
)
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.core.serialization import component_to_dict, default_from_dict, default_to_dict
from haystack.dataclasses import (
    AgentBreakpoint,
    AgentSnapshot,
    ChatMessage,
    ChatRole,
    StreamingCallbackT,
    ToolBreakpoint,
    select_streaming_callback,
)
from haystack.human_in_the_loop import ToolExecutionDecision
from haystack.human_in_the_loop.strategies import (
    _process_confirmation_strategies,
    _process_confirmation_strategies_async,
)
from haystack.human_in_the_loop.types import ConfirmationStrategy
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
from haystack.utils.deserialization import deserialize_component_inplace

logger = logging.getLogger(__name__)


def _get_run_method_params(instance: "Agent") -> set[str]:
    """Derive the parameter names of the Agent.run method via introspection."""
    sig = inspect.signature(instance.run)
    return {name for name, p in sig.parameters.items() if p.kind != inspect.Parameter.VAR_KEYWORD}


@dataclass(kw_only=True)
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
    :param confirmation_strategy_context: Optional dictionary for passing request-scoped resources
        to confirmation strategies. In web/server environments, this enables passing per-request
        objects (e.g., WebSocket connections, async queues, or pub/sub clients) that strategies can use for
        non-blocking user interaction. This is passed directly to strategies via the `confirmation_strategy_context`
        parameter in their `run()` and `run_async()` methods.
    :param tool_execution_decisions: Optional list of ToolExecutionDecision objects to use instead of prompting
        the user. This is useful when restarting from a snapshot where tool execution decisions were already made.
    """

    state: State
    component_visits: dict
    chat_generator_inputs: dict
    tool_invoker_inputs: dict
    counter: int = 0
    skip_chat_generator: bool = False
    confirmation_strategy_context: dict[str, Any] | None = None
    tool_execution_decisions: list[ToolExecutionDecision] | None = None


@component
class Agent:
    """
    A tool-using Agent powered by a large language model.

    The Agent processes messages and calls tools until it meets an exit condition.
    You can set one or more exit conditions to control when it stops.
    For example, it can stop after generating a response or after calling a tool.

    Without tools, the Agent works like a standard LLM that generates text. It produces one response and then stops.

    ### Usage examples

    This is an example agent that:
    1. Searches for tipping customs in France.
    2. Uses a calculator to compute tips based on its findings.
    3. Returns the final answer with its context.

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

    print(result["messages"][-1].text)
    ```

    #### Using a `user_prompt` template with variables

    You can define a reusable `user_prompt` with Jinja2 template variables so the Agent can be invoked
    with different inputs without manually constructing `ChatMessage` objects each time.
    This is especially useful when embedding the Agent in a pipeline or calling it in a loop.

    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.tools import tool
    from typing import Annotated


    @tool
    def translate(
        text: Annotated[str, "The text to translate"],
        target_language: Annotated[str, "The language to translate to"],
    ) -> str:
        \"\"\"Translate text to a target language.\"\"\"
        # Placeholder: would call an actual translation API
        return f"[Translated '{text}' to {target_language}]"

    agent = Agent(
        chat_generator=OpenAIChatGenerator(),
        tools=[translate],
        system_prompt="You are a helpful translation assistant.",
        user_prompt=\"\"\"{% message role="user"%}
    Translate the following document to {{ language }}: {{ document }}
    {% endmessage %}\"\"\",
        required_variables=["language", "document"],
    )

    # The template variables 'language' and 'document' become inputs to the run method
    result = agent.run(
        language="French",
        document="The weather is lovely today and the sun is shining.",
    )

    print(result["last_message"].text)
    ```

    """

    def __init__(
        self,
        *,
        chat_generator: ChatGenerator,
        tools: ToolsType | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        required_variables: list[str] | Literal["*"] | None = None,
        exit_conditions: list[str] | None = None,
        state_schema: dict[str, Any] | None = None,
        max_agent_steps: int = 100,
        streaming_callback: StreamingCallbackT | None = None,
        raise_on_tool_invocation_failure: bool = False,
        tool_invoker_kwargs: dict[str, Any] | None = None,
        confirmation_strategies: dict[str | tuple[str, ...], ConfirmationStrategy] | None = None,
    ) -> None:
        """
        Initialize the agent component.

        :param chat_generator: An instance of the chat generator that your agent should use. It must support tools.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset that the agent can use.
        :param system_prompt: System prompt for the agent.
        :param user_prompt: User prompt for the agent. If provided this is appended to the messages provided at runtime.
        :param required_variables:
            List variables that must be provided as input to user_prompt.
            If a variable listed as required is not provided, an exception is raised.
            If set to `"*"`, all variables found in the prompt are required. Optional.
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
        :param confirmation_strategies: A dictionary mapping tool names to ConfirmationStrategy instances.
        :raises TypeError: If the chat_generator does not support tools parameter in its run method.
        :raises ValueError: If the exit_conditions are not valid.
        :raises ValueError: If any `user_prompt` variable overlaps with `state` schema or `run` parameters.
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
        self.user_prompt = user_prompt
        self.required_variables = required_variables
        self.exit_conditions = exit_conditions
        self.max_agent_steps = max_agent_steps
        self.raise_on_tool_invocation_failure = raise_on_tool_invocation_failure
        self.streaming_callback = streaming_callback

        # Set input and output types for the component based on the State schema
        self._run_method_params = _get_run_method_params(self)
        output_types = {"last_message": ChatMessage}
        for param, config in self.state_schema.items():
            output_types[param] = config["type"]
            # Skip setting input types for parameters that are already in the run method
            if param in self._run_method_params:
                continue
            component.set_input_type(self, name=param, type=config["type"], default=None)
        component.set_output_types(self, **output_types)

        self._chat_prompt_builder: ChatPromptBuilder | None = self._initialize_chat_prompt_builder(
            user_prompt, required_variables
        )

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

        self._confirmation_strategies = confirmation_strategies or {}

        self._is_warmed_up = False

    def _initialize_chat_prompt_builder(
        self, user_prompt: str | None, required_variables: list[str] | Literal["*"] | None
    ) -> ChatPromptBuilder | None:
        """
        Initialize the ChatPromptBuilder if a user prompt is provided.
        """
        if user_prompt is None:
            if required_variables is not None:
                logger.warning(
                    "The parameter required_variables is provided but user_prompt is not. "
                    "Either provide a user_prompt or remove required_variables, it has otherwise no effect."
                )
            return None

        chat_prompt_builder = ChatPromptBuilder(template=user_prompt, required_variables=required_variables)
        prompt_variables = chat_prompt_builder.variables
        for var_name in prompt_variables:
            if var_name in self.state_schema:
                raise ValueError(
                    f"Variable '{var_name}' from the user prompt is already defined in the state schema. "
                    "Please rename the variable or remove it from the user prompt to avoid conflicts."
                )
            if var_name in self._run_method_params:
                raise ValueError(
                    f"Variable '{var_name}' from the user prompt conflicts with input names in the run method. "
                    "Please rename the variable or remove it from the user prompt to avoid conflicts."
                )
            if required_variables == "*" or var_name in (required_variables or []):
                component.set_input_type(self, name=var_name, type=Any)
            else:
                component.set_input_type(self, name=var_name, type=Any, default=None)
        return chat_prompt_builder

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
            user_prompt=self.user_prompt,
            required_variables=self.required_variables,
            exit_conditions=self.exit_conditions,
            # We serialize the original state schema, not the resolved one to reflect the original user input
            state_schema=_schema_to_dict(self._state_schema),
            max_agent_steps=self.max_agent_steps,
            streaming_callback=serialize_callable(self.streaming_callback) if self.streaming_callback else None,
            raise_on_tool_invocation_failure=self.raise_on_tool_invocation_failure,
            tool_invoker_kwargs=self.tool_invoker_kwargs,
            confirmation_strategies={
                (list(key) if isinstance(key, tuple) else key): component_to_dict(
                    obj=strategy, name="confirmation_strategy"
                )
                for key, strategy in self._confirmation_strategies.items()
            }
            if self._confirmation_strategies
            else None,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Agent":
        """
        Deserialize the agent from a dictionary.

        :param data: Dictionary to deserialize from
        :return: Deserialized agent
        """
        init_params = data.get("init_parameters", {})

        deserialize_component_inplace(init_params, key="chat_generator")

        if init_params.get("state_schema") is not None:
            init_params["state_schema"] = _schema_from_dict(init_params["state_schema"])

        if init_params.get("streaming_callback") is not None:
            init_params["streaming_callback"] = deserialize_callable(init_params["streaming_callback"])

        deserialize_tools_or_toolset_inplace(init_params, key="tools")

        if init_params.get("confirmation_strategies") is not None:
            restored: dict[str | tuple[str, ...], Any] = {}
            for raw_key in init_params["confirmation_strategies"].keys():
                deserialize_component_inplace(init_params["confirmation_strategies"], key=raw_key)
                strategy = init_params["confirmation_strategies"][raw_key]
                if isinstance(raw_key, list):
                    key = tuple(raw_key)
                else:
                    key = raw_key
                restored[key] = strategy
            init_params["confirmation_strategies"] = restored

        return default_from_dict(cls, data)

    def _create_agent_span(self) -> Any:
        """
        Create a span for the agent run.

        If the agent is running as part of a pipeline, this span will be nested
        under the current active span (the pipeline's component span).
        """
        parent_span = tracing.tracer.current_span()
        return tracing.tracer.trace(
            "haystack.agent.run",
            tags={
                "haystack.agent.max_steps": self.max_agent_steps,
                "haystack.agent.tools": self.tools,
                "haystack.agent.exit_conditions": self.exit_conditions,
                "haystack.agent.state_schema": _schema_to_dict(self.state_schema),
            },
            parent_span=parent_span,
        )

    def _initialize_fresh_execution(
        self,
        messages: list[ChatMessage] | None,
        streaming_callback: StreamingCallbackT | None,
        requires_async: bool,
        *,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | list[str] | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
        **kwargs,
    ) -> _ExecutionContext:
        """
        Initialize execution context for a fresh run of the agent.

        :param messages: List of ChatMessage objects to start the agent with.
        :param streaming_callback: Optional callback for streaming responses.
        :param requires_async: Whether the agent run requires asynchronous execution.
        :param system_prompt: System prompt for the agent. If provided, it overrides the default system prompt.
        :param user_prompt: User prompt for the agent. If provided, it overrides the default user prompt and is
            appended to the messages provided at runtime.
        :param generation_kwargs: Additional keyword arguments for chat generator. These parameters will
            override the parameters passed during component initialization.
        :param tools: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
            When passing tool names, tools are selected from the Agent's originally configured tools.
        :param confirmation_strategy_context: Optional dictionary for passing request-scoped resources
            to confirmation strategies.
        :param kwargs: Additional data to pass to the State used by the Agent.
        """
        user_prompt = user_prompt or self.user_prompt
        system_prompt = system_prompt or self.system_prompt
        if messages is None and user_prompt is None and system_prompt is None:
            raise ValueError(
                "No messages provided to the Agent and neither user_prompt nor system_prompt is set. "
                "Please provide at least one of these inputs."
            )

        messages = messages or []

        if user_prompt is not None:
            if self._chat_prompt_builder is None:
                raise ValueError(
                    "user_prompt is provided but the ChatPromptBuilder is not initialized. "
                    "Please make sure a user_prompt is provided at initialization time."
                )

            # Only forward the prompt kwargs to the prompt builder
            prompt_kwargs = {var: kwargs[var] for var in self._chat_prompt_builder.variables if var in kwargs}
            user_messages = self._chat_prompt_builder.run(template=user_prompt, **prompt_kwargs)["prompt"]
            messages = messages + user_messages

        system_prompt = system_prompt or self.system_prompt
        if system_prompt is not None:
            messages = [ChatMessage.from_system(system_prompt)] + messages

        if all(m.is_from(ChatRole.SYSTEM) for m in messages):
            logger.warning("All messages provided to the Agent component are system messages. This is not recommended.")

        state_kwargs: dict[str, Any] = {key: kwargs[key] for key in self.state_schema.keys() if key in kwargs}
        state = State(schema=self.state_schema, data=state_kwargs)
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

        # We add enable_streaming_callback_passthrough to the tool invoker inputs
        if self._tool_invoker:
            tool_invoker_inputs["enable_streaming_callback_passthrough"] = (
                self._tool_invoker.enable_streaming_callback_passthrough
            )

        return _ExecutionContext(
            state=state,
            component_visits=dict.fromkeys(["chat_generator", "tool_invoker"], 0),
            chat_generator_inputs=generator_inputs,
            tool_invoker_inputs=tool_invoker_inputs,
            confirmation_strategy_context=confirmation_strategy_context,
        )

    def _select_tools(self, tools: ToolsType | list[str] | None = None) -> ToolsType:
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
            return cast(list[Tool | Toolset], tools)  # mypy can't narrow the Union type from isinstance check

        raise TypeError(
            "tools must be a list of Tool and/or Toolset objects, a Toolset, or a list of tool names (strings)."
        )

    def _initialize_from_snapshot(
        self,
        snapshot: AgentSnapshot,
        streaming_callback: StreamingCallbackT | None,
        requires_async: bool,
        *,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | list[str] | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
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
        :param confirmation_strategy_context: Optional dictionary for passing request-scoped resources
            to confirmation strategies.
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

        # We add enable_streaming_callback_passthrough to the tool invoker inputs
        if self._tool_invoker:
            tool_invoker_inputs["enable_streaming_callback_passthrough"] = (
                self._tool_invoker.enable_streaming_callback_passthrough
            )

        return _ExecutionContext(
            state=state,
            component_visits=component_visits,
            chat_generator_inputs=generator_inputs,
            tool_invoker_inputs=tool_invoker_inputs,
            counter=snapshot.break_point.break_point.visit_count,
            skip_chat_generator=skip_chat_generator,
            confirmation_strategy_context=confirmation_strategy_context,
        )

    def _runtime_checks(self, break_point: AgentBreakpoint | None) -> None:
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

    def run(  # noqa: PLR0915
        self,
        messages: list[ChatMessage] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        *,
        generation_kwargs: dict[str, Any] | None = None,
        break_point: AgentBreakpoint | None = None,
        snapshot: AgentSnapshot | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: ToolsType | list[str] | None = None,
        snapshot_callback: SnapshotCallback | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
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
        :param user_prompt: User prompt for the agent. If provided, it overrides the default user prompt and is
            appended to the messages provided at runtime.
        :param tools: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
            When passing tool names, tools are selected from the Agent's originally configured tools.
        :param snapshot_callback: Optional callback function that is invoked when a pipeline snapshot is created.
            The callback receives a `PipelineSnapshot` object and can return an optional string.
            If provided, the callback is used instead of the default file-saving behavior.
        :param confirmation_strategy_context: Optional dictionary for passing request-scoped resources
            to confirmation strategies. Useful in web/server environments to provide per-request
            objects (e.g., WebSocket connections, async queues, Redis pub/sub clients) that strategies
            can use for non-blocking user interaction.
        :param kwargs: Additional data to pass to the State schema used by the Agent.
            The keys must match the schema defined in the Agent's `state_schema`.
        :returns:
            A dictionary with the following keys:
            - "messages": List of all messages exchanged during the agent's run.
            - "last_message": The last message exchanged during the agent's run.
            - Any additional keys defined in the `state_schema`.
        :raises BreakpointException: If an agent breakpoint is triggered.
        """
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
                generation_kwargs=generation_kwargs,
                tools=tools,
                confirmation_strategy_context=confirmation_strategy_context,
            )
        else:
            exe_context = self._initialize_fresh_execution(
                messages=messages,
                streaming_callback=streaming_callback,
                requires_async=False,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                generation_kwargs=generation_kwargs,
                tools=tools,
                confirmation_strategy_context=confirmation_strategy_context,
                **kwargs,
            )

        with self._create_agent_span() as span:
            span.set_content_tag("haystack.agent.input", _deepcopy_with_exceptions(agent_inputs))

            while exe_context.counter < self.max_agent_steps:
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
                            break_point=break_point.break_point if isinstance(break_point, AgentBreakpoint) else None,
                        )
                    except PipelineRuntimeError as e:
                        agent_name = getattr(self, "__component_name__", None)
                        pipe_snapshot = _create_pipeline_snapshot_from_chat_generator(
                            agent_name=agent_name, execution_context=exe_context, break_point=None
                        )
                        new_error = PipelineRuntimeError.from_exception(agent_name or "Agent", Agent, e)
                        new_error.pipeline_snapshot = pipe_snapshot
                        # If Agent is not in a pipeline, we save the snapshot to a file or invoke a custom callback.
                        # Checked by __component_name__ not being set.
                        if agent_name is None:
                            new_error.pipeline_snapshot_file_path = _save_pipeline_snapshot(
                                pipeline_snapshot=pipe_snapshot, snapshot_callback=snapshot_callback
                            )
                        raise new_error from e
                    except BreakpointException as e:
                        agent_name = break_point.agent_name if break_point else None
                        e.pipeline_snapshot = _create_pipeline_snapshot_from_chat_generator(
                            agent_name=agent_name, execution_context=exe_context, break_point=break_point
                        )
                        e._break_point = e.pipeline_snapshot.break_point
                        # If Agent is not in a pipeline, we save the snapshot to a file or invoke a custom callback.
                        # Checked by __component_name__ not being set.
                        if getattr(self, "__component_name__", None) is None:
                            full_file_path = _save_pipeline_snapshot(
                                pipeline_snapshot=e.pipeline_snapshot, snapshot_callback=snapshot_callback
                            )
                            e.pipeline_snapshot_file_path = full_file_path
                        raise e

                    llm_messages = result["replies"]
                    exe_context.state.set("messages", llm_messages)

                # Check if any of the LLM responses contain a tool call or if the LLM is not using tools
                if not any(msg.tool_call for msg in llm_messages) or self._tool_invoker is None:
                    exe_context.counter += 1
                    break

                # We only pass down the breakpoint if the tool name matches the tool call in the LLM messages
                break_point_to_pass = None
                if (
                    break_point
                    and isinstance(break_point.break_point, ToolBreakpoint)
                    and _should_trigger_tool_invoker_breakpoint(
                        break_point=break_point.break_point, llm_messages=llm_messages
                    )
                ):
                    break_point_to_pass = break_point.break_point

                # Apply confirmation strategies and update State and messages sent to ToolInvoker
                # Run confirmation strategies to get updated tool call messages and modified chat history
                modified_tool_call_messages, new_chat_history = _process_confirmation_strategies(
                    confirmation_strategies=self._confirmation_strategies,
                    messages_with_tool_calls=llm_messages,
                    execution_context=exe_context,
                )
                # Replace the chat history in state with the modified one
                exe_context.state.set(key="messages", value=new_chat_history, handler_override=replace_values)

                # Run ToolInvoker
                try:
                    # We only send the messages from the LLM to the tool invoker
                    tool_invoker_result = Pipeline._run_component(
                        component_name="tool_invoker",
                        component={"instance": self._tool_invoker},
                        inputs={
                            "messages": modified_tool_call_messages,
                            "state": exe_context.state,
                            **exe_context.tool_invoker_inputs,
                        },
                        component_visits=exe_context.component_visits,
                        parent_span=span,
                        break_point=break_point_to_pass,
                    )
                except PipelineRuntimeError as e:
                    agent_name = getattr(self, "__component_name__", None)
                    tool_name = getattr(e.__cause__, "tool_name", None)
                    pipe_snapshot = _create_pipeline_snapshot_from_tool_invoker(
                        tool_name=tool_name, agent_name=agent_name, execution_context=exe_context, break_point=None
                    )
                    new_error = PipelineRuntimeError.from_exception(agent_name or "Agent", Agent, e)
                    new_error.pipeline_snapshot = pipe_snapshot
                    # If Agent is not in a pipeline, we save the snapshot to a file or invoke a custom callback.
                    # Checked by __component_name__ not being set.
                    if agent_name is None:
                        new_error.pipeline_snapshot_file_path = _save_pipeline_snapshot(
                            pipeline_snapshot=pipe_snapshot, snapshot_callback=snapshot_callback
                        )
                    raise new_error from e
                except BreakpointException as e:
                    e.pipeline_snapshot = _create_pipeline_snapshot_from_tool_invoker(
                        tool_name=e.break_point.tool_name if isinstance(e.break_point, ToolBreakpoint) else None,
                        agent_name=break_point.agent_name if break_point else None,
                        execution_context=exe_context,
                        break_point=break_point,
                    )
                    e._break_point = e.pipeline_snapshot.break_point
                    # If Agent is not in a pipeline, we save the snapshot to a file or invoke a custom callback.
                    # Checked by __component_name__ not being set.
                    if getattr(self, "__component_name__", None) is None:
                        e.pipeline_snapshot_file_path = _save_pipeline_snapshot(
                            pipeline_snapshot=e.pipeline_snapshot, snapshot_callback=snapshot_callback
                        )
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

    async def run_async(  # noqa: PLR0915
        self,
        messages: list[ChatMessage] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        *,
        generation_kwargs: dict[str, Any] | None = None,
        break_point: AgentBreakpoint | None = None,
        snapshot: AgentSnapshot | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        tools: ToolsType | list[str] | None = None,
        snapshot_callback: SnapshotCallback | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
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
        :param user_prompt: User prompt for the agent. If provided, it overrides the default user prompt and is
            appended to the messages provided at runtime.
        :param tools: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
        :param snapshot_callback: Optional callback function that is invoked when a pipeline snapshot is created.
            The callback receives a `PipelineSnapshot` object and can return an optional string.
            If provided, the callback is used instead of the default file-saving behavior.
        :param kwargs: Additional data to pass to the State schema used by the Agent.
            The keys must match the schema defined in the Agent's `state_schema`.
        :param confirmation_strategy_context: Optional dictionary for passing request-scoped resources
            to confirmation strategies. Useful in web/server environments to provide per-request
            objects (e.g., WebSocket connections, async queues, Redis pub/sub clients) that strategies
            can use for non-blocking user interaction.
        :returns:
            A dictionary with the following keys:
            - "messages": List of all messages exchanged during the agent's run.
            - "last_message": The last message exchanged during the agent's run.
            - Any additional keys defined in the `state_schema`.
        :raises BreakpointException: If an agent breakpoint is triggered.
        """
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
                confirmation_strategy_context=confirmation_strategy_context,
            )
        else:
            exe_context = self._initialize_fresh_execution(
                messages=messages,
                streaming_callback=streaming_callback,
                requires_async=True,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tools,
                generation_kwargs=generation_kwargs,
                confirmation_strategy_context=confirmation_strategy_context,
                **kwargs,
            )

        with self._create_agent_span() as span:
            span.set_content_tag("haystack.agent.input", _deepcopy_with_exceptions(agent_inputs))

            while exe_context.counter < self.max_agent_steps:
                # We skip the chat generator when restarting from a snapshot from a ToolBreakpoint
                if exe_context.skip_chat_generator:
                    llm_messages = exe_context.state.get("messages", [])[-1:]
                    # Set to False so the next iteration will call the chat generator
                    exe_context.skip_chat_generator = False
                else:
                    try:
                        result = await AsyncPipeline._run_component_async(
                            component_name="chat_generator",
                            component={"instance": self.chat_generator},
                            component_inputs={
                                "messages": exe_context.state.data["messages"],
                                **exe_context.chat_generator_inputs,
                            },
                            component_visits=exe_context.component_visits,
                            parent_span=span,
                            break_point=break_point.break_point if isinstance(break_point, AgentBreakpoint) else None,
                        )
                    except BreakpointException as e:
                        e.pipeline_snapshot = _create_pipeline_snapshot_from_chat_generator(
                            agent_name=break_point.agent_name if break_point else None,
                            execution_context=exe_context,
                            break_point=break_point,
                        )
                        e._break_point = e.pipeline_snapshot.break_point
                        # We check if the agent is part of a pipeline by checking for __component_name__
                        # If it is not in a pipeline, we save the snapshot to a file or invoke a custom callback.
                        in_pipeline = getattr(self, "__component_name__", None) is not None
                        if not in_pipeline:
                            e.pipeline_snapshot_file_path = _save_pipeline_snapshot(
                                pipeline_snapshot=e.pipeline_snapshot, snapshot_callback=snapshot_callback
                            )
                        raise e

                    llm_messages = result["replies"]
                    exe_context.state.set("messages", llm_messages)

                # Check if any of the LLM responses contain a tool call or if the LLM is not using tools
                if not any(msg.tool_call for msg in llm_messages) or self._tool_invoker is None:
                    exe_context.counter += 1
                    break

                # We only pass down the breakpoint if the tool name matches the tool call in the LLM messages
                break_point_to_pass = None
                if (
                    break_point
                    and isinstance(break_point.break_point, ToolBreakpoint)
                    and _should_trigger_tool_invoker_breakpoint(
                        break_point=break_point.break_point, llm_messages=llm_messages
                    )
                ):
                    break_point_to_pass = break_point.break_point

                # Apply confirmation strategies and update State and messages sent to ToolInvoker
                # Run confirmation strategies to get updated tool call messages and modified chat history
                modified_tool_call_messages, new_chat_history = await _process_confirmation_strategies_async(
                    confirmation_strategies=self._confirmation_strategies,
                    messages_with_tool_calls=llm_messages,
                    execution_context=exe_context,
                )
                # Replace the chat history in state with the modified one
                exe_context.state.set(key="messages", value=new_chat_history, handler_override=replace_values)

                try:
                    # We only send the messages from the LLM to the tool invoker
                    tool_invoker_result = await AsyncPipeline._run_component_async(
                        component_name="tool_invoker",
                        component={"instance": self._tool_invoker},
                        component_inputs={
                            "messages": modified_tool_call_messages,
                            "state": exe_context.state,
                            **exe_context.tool_invoker_inputs,
                        },
                        component_visits=exe_context.component_visits,
                        parent_span=span,
                        break_point=break_point_to_pass,
                    )
                except BreakpointException as e:
                    e.pipeline_snapshot = _create_pipeline_snapshot_from_tool_invoker(
                        tool_name=e.break_point.tool_name if isinstance(e.break_point, ToolBreakpoint) else None,
                        agent_name=break_point.agent_name if break_point else None,
                        execution_context=exe_context,
                        break_point=break_point,
                    )
                    e._break_point = e.pipeline_snapshot.break_point
                    # If Agent is not in a pipeline, we save the snapshot to a file or invoke a custom callback.
                    # Checked by __component_name__ not being set.
                    if getattr(self, "__component_name__", None) is None:
                        e.pipeline_snapshot_file_path = _save_pipeline_snapshot(
                            pipeline_snapshot=e.pipeline_snapshot, snapshot_callback=snapshot_callback
                        )
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
