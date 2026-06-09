# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal, cast

from haystack import component, logging, tracing
from haystack.components.agents.state.state import (
    State,
    _schema_from_dict,
    _schema_to_dict,
    _validate_schema,
    replace_values,
)
from haystack.components.agents.state.state_utils import merge_lists
from haystack.components.agents.tool_calling import _run_tool, _run_tool_async
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.serialization import component_to_dict, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole, StreamingCallbackT, select_streaming_callback
from haystack.human_in_the_loop.strategies import (
    _deserialize_confirmation_strategies,
    _process_confirmation_strategies,
    _process_confirmation_strategies_async,
)
from haystack.human_in_the_loop.types import ConfirmationStrategy
from haystack.tools import (
    Tool,
    Toolset,
    ToolsType,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    flatten_tools_or_toolsets,
    serialize_tools_or_toolset,
    warm_up_tools,
)
from haystack.utils.async_utils import _run_component_async
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable
from haystack.utils.deserialization import deserialize_component_inplace

logger = logging.getLogger(__name__)

# Regex to detect the Jinja2 chat template syntax
_JINJA2_CHAT_TEMPLATE_RE = re.compile(r"\{%\s*message\s")
# Regex to extract the role from a Jinja2 message block, e.g. {% message role="user" %}
_JINJA2_MESSAGE_ROLE_RE = re.compile(r'\{%\s*message\s+role\s*=\s*["\'](\w+)["\']')

# State keys that the Agent populates automatically during a run.
# Users may not define them in their own `state_schema`, and they are exposed only as Agent outputs.
_INTERNAL_STATE_KEYS: dict[str, dict[str, Any]] = {
    "step_count": {"type": int, "handler": replace_values},
    "token_usage": {"type": dict[str, Any], "handler": replace_values},
    "tool_call_counts": {"type": dict[str, int], "handler": replace_values},
}


def _accumulate_usage(current: Any, new: Any) -> Any:
    """
    Recursively sum numeric leaf values across two usage-like dicts.

    Used to aggregate `ChatMessage.meta["usage"]` payloads across LLM calls in a run. Nested dicts (e.g. OpenAI's
    `completion_tokens_details`) are merged recursively; numeric leaves are summed; other types fall back to the new
    value.

    :param current: The current accumulated usage data.
    :param new: The new usage data to merge in.
    """
    if isinstance(current, dict) and isinstance(new, dict):
        result = dict(current)
        for k, v in new.items():
            result[k] = _accumulate_usage(result[k], v) if k in result else deepcopy(v)
        return result
    if isinstance(current, (int, float)) and isinstance(new, (int, float)):
        return current + new
    return new


def _record_llm_usage(state: State, llm_messages: list[ChatMessage]) -> None:
    """
    Aggregate token usage from the latest LLM messages into the State.

    Only writes when at least one message reports `meta["usage"]`, so generators that don't surface usage data
    leave `token_usage` at its default empty dict rather than overwriting it.

    :param state: The Agent's State, used to read the running `token_usage` total and write back the new total.
    :param llm_messages: The ChatMessage objects returned from the latest LLM call. Token usage is read from each
        message's `meta["usage"]` field, if present.
    """
    current = state.get("token_usage")
    updated = False
    for msg in llm_messages:
        usage = msg.meta.get("usage")
        if isinstance(usage, dict):
            current = _accumulate_usage(current or {}, usage)
            updated = True
    if updated:
        state.set("token_usage", current)


def _record_tool_calls(state: State, tool_messages: list[ChatMessage]) -> None:
    """
    Increment per-tool call counts in the State for every successfully dispatched tool.

    :param state: The Agent's State, used to read the running `tool_call_counts` map and write back the new totals.
    :param tool_messages: The ChatMessage objects returned from the latest tool execution. Per-tool counts are
        incremented based on each message's `tool_call_result.origin.tool_name`.
    """
    counts = state.get("tool_call_counts") or {}
    updated = False
    for tm in tool_messages:
        if tm.tool_call_result is None:
            continue
        name = tm.tool_call_result.origin.tool_name
        counts[name] = counts.get(name, 0) + 1
        updated = True
    if updated:
        state.set("tool_call_counts", counts)


def _get_run_method_params(instance: "Agent") -> set[str]:
    """Derive the parameter names of the Agent.run method via introspection."""
    sig = inspect.signature(instance.run)
    return {name for name, p in sig.parameters.items() if p.kind != inspect.Parameter.VAR_KEYWORD}


def _validate_prompt_message_blocks(user_prompt: str | None, system_prompt: str | None) -> None:
    """
    Validate explicit Jinja2 message blocks in Agent prompts.

    :param user_prompt: Optional user prompt template.
    :param system_prompt: Optional system prompt template.
    :raises ValueError: If a prompt contains multiple message blocks or a literal block role is invalid.
    """
    if user_prompt is not None:
        message_blocks = _JINJA2_CHAT_TEMPLATE_RE.findall(user_prompt)
        roles = _JINJA2_MESSAGE_ROLE_RE.findall(user_prompt)
        if len(message_blocks) > 1:
            raise ValueError(f"user_prompt must define exactly one message block, found {len(message_blocks)}.")
        if roles and roles[0] != "user":
            raise ValueError(f"user_prompt message block must have role 'user', found role '{roles[0]}'.")

    if system_prompt is not None and _JINJA2_CHAT_TEMPLATE_RE.search(system_prompt):
        message_blocks = _JINJA2_CHAT_TEMPLATE_RE.findall(system_prompt)
        roles = _JINJA2_MESSAGE_ROLE_RE.findall(system_prompt)
        if len(message_blocks) > 1:
            raise ValueError(f"system_prompt must define exactly one message block, found {len(message_blocks)}.")
        if roles and roles[0] != "system":
            raise ValueError(f"system_prompt message block must have role 'system', found role '{roles[0]}'.")


def _template_for_role(prompt: str, role: str) -> str:
    """
    Convert a prompt into a ChatPromptBuilder string template for the expected role.

    :param prompt: Prompt template, with or without an explicit Jinja2 message block.
    :param role: Role to use when wrapping a plain string prompt.
    :returns: The original message-block template, or a plain string prompt wrapped in one message block.
    """
    if _JINJA2_CHAT_TEMPLATE_RE.search(prompt):
        return prompt
    return f'{{% message role="{role}" %}}{prompt}{{% endmessage %}}'


def _render_prompt_messages(
    *, prompt_builder: ChatPromptBuilder, expected_role: ChatRole, prompt_label: str, kwargs: dict[str, Any]
) -> list[ChatMessage]:
    """
    Render one Agent prompt and validate the rendered message.

    :param prompt_builder: Builder configured with the prompt template.
    :param expected_role: Role the rendered message must have.
    :param prompt_label: Prompt name used in error messages.
    :param kwargs: Runtime values available to the prompt template.
    :returns: A single rendered prompt message.
    :raises ValueError: If the prompt renders to zero, multiple, or wrong-role messages.
    """
    prompt_kwargs = {var: kwargs[var] for var in prompt_builder.variables if var in kwargs}
    prompt_messages = prompt_builder.run(**prompt_kwargs)["prompt"]
    if len(prompt_messages) != 1:
        raise ValueError(
            f"{prompt_label} must render to exactly one {expected_role.value} message. "
            f"Got {len(prompt_messages)} messages."
        )
    if not prompt_messages[0].is_from(expected_role):
        raise ValueError(
            f"{prompt_label} must render to a {expected_role.value} message. "
            f"Got a message with role {prompt_messages[0].role}."
        )
    return prompt_messages


@dataclass(kw_only=True)
class _ExecutionContext:
    """
    Context for executing the agent.

    :param state: The current state of the agent, including messages and any additional data.
    :param tools: The tools selected for this run, kept unflattened (the original Toolset or list of
        Tools/Toolsets). Storing the unflattened form lets each step re-flatten it and pick up tools a dynamic
        toolset (e.g. SearchableToolset) discovers over time; flattening would freeze a snapshot. The chat
        generator and tool execution receive a freshly flattened snapshot per step.
    :param chat_generator_inputs: Runtime inputs to be passed to the chat generator (tools are injected per step).
    :param tool_execution_inputs: Runtime inputs to be passed to tool execution (tools are injected per step).
    :param counter: A counter to track the number of steps taken in the agent's run.
    :param confirmation_strategy_context: Optional dictionary for passing request-scoped resources
        to confirmation strategies. In web/server environments, this enables passing per-request
        objects (e.g., WebSocket connections, async queues, or pub/sub clients) that strategies can use for
        non-blocking user interaction. This is passed directly to strategies via the `confirmation_strategy_context`
        parameter in their `run()` and `run_async()` methods.
    """

    state: State
    tools: ToolsType
    chat_generator_inputs: dict
    tool_execution_inputs: dict
    counter: int = 0
    confirmation_strategy_context: dict[str, Any] | None = None


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
    from haystack.components.generators.utils import print_streaming_chunk
    from haystack.dataclasses import ChatMessage
    from haystack.tools import tool
    from typing import Annotated, Literal

    # Tool functions - in practice, these would have real implementations
    @tool
    def search(query: Annotated[str, "The search query"]) -> str:
        '''Search for information on the web.'''
        # Placeholder: would call actual search API
        return "In France, a 15% service charge is typically included, but leaving 5-10% extra is appreciated."

    @tool
    def calculator(
        operation: Annotated[Literal["multiply", "percentage"], "The mathematical operation to perform"],
        a: Annotated[float, "First number"],
        b: Annotated[float, "Second number"],
    ) -> float:
        '''Perform mathematical calculations.'''
        if operation == "multiply":
            return a * b
        elif operation == "percentage":
            return (a / 100) * b
        return 0

    agent = Agent(
        system_prompt=(
            "You are a helpful assistant. Use the 'search' tool to find information "
            "about a user's question and the 'calculator' tool to perform math."
        ),
        chat_generator=OpenAIChatGenerator(),
        tools=[search, calculator],
        streaming_callback=print_streaming_chunk,
    )

    result = agent.run(
        messages=[ChatMessage.from_user("Calculate the appropriate tip for an €85 meal in France")]
    )

    # Access the final response from the Agent
    # print(result["last_message"].text)
    ```

    #### Using a `user_prompt` template with variables

    You can define a reusable `user_prompt` with Jinja2 template variables so the Agent can be invoked
    with different inputs without manually constructing `ChatMessage` objects each time.
    This is especially useful when embedding the Agent in a pipeline.

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
        messages=[],
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
        tool_concurrency_limit: int = 4,
        tool_streaming_callback_passthrough: bool = False,
        confirmation_strategies: dict[str | tuple[str, ...], ConfirmationStrategy] | None = None,
    ) -> None:
        """
        Initialize the agent component.

        :param chat_generator: An instance of the chat generator that your agent should use. It must support tools.
        :param tools: A list of Tool and/or Toolset objects, or a single Toolset that the agent can use.
        :param system_prompt: System prompt for the agent. Can be a plain string template or a Jinja2 message template.
            For details on the supported template syntax, refer to the
            [documentation](https://docs.haystack.deepset.ai/docs/chatpromptbuilder#string-templates).
        :param user_prompt: User prompt for the agent. Can be a plain string template or a Jinja2 message template.
            If provided, this is appended to the messages provided at runtime.
            For details on the supported template syntax, refer to the
            [documentation](https://docs.haystack.deepset.ai/docs/chatpromptbuilder#string-templates).
        :param required_variables:
            Lists the variables that must be provided as inputs to `user_prompt` or `system_prompt`.
            If a required variable is not provided at run time, an exception is raised.
            If set to `"*"`, all variables found in the prompts are required. Optional.
        :param exit_conditions: List of conditions that will cause the agent to return.
            Can include "text" if the agent should return when it generates a message without tool calls,
            or tool names that will cause the agent to return once the tool was executed. Defaults to ["text"].
        :param state_schema: A dictionary defining the agent's runtime state. Each key maps to a type config
            with `"type"` (required) and an optional `"handler"` for merging values across tool calls.
            Tools can read from and write to state keys using `inputs_from_state` and `outputs_to_state`.
        :param max_agent_steps: Maximum number of steps the agent will run before stopping. Defaults to 100.
            A step is one chat-generator call plus the execution of every tool call the model requested in
            that call (if any). If the agent reaches this number of steps it stops and returns the current state.
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
            The same callback can be configured to emit tool results when a tool is called.
        :param raise_on_tool_invocation_failure: Should the agent raise an exception when a tool invocation fails?
            If set to False, the exception will be turned into a chat message and passed to the LLM.
        :param tool_concurrency_limit: Maximum number of tool calls to execute at the same time.
            Defaults to 4. Set to 1 to disable parallel tool execution.
        :param tool_streaming_callback_passthrough: If True, pass the streaming callback to tools that accept it.
        :param confirmation_strategies: A dictionary mapping tool names to ConfirmationStrategy instances.
        :raises TypeError: If the chat_generator does not support tools parameter in its run method.
        :raises ValueError: If any `user_prompt` variable overlaps with the `state_schema` or `run` method parameters.
        """
        # --- Validation ---
        self._chat_generator_supports_tools: bool = "tools" in inspect.signature(chat_generator.run).parameters
        # We use an explicit None check for tools b/c testing for truthiness calls __len__, which for SearchableToolset
        # would iterate and prematurely warm it up at init.
        if tools is not None and not self._chat_generator_supports_tools:
            raise TypeError(
                f"{type(chat_generator).__name__} does not accept tools parameter in its run method. "
                "The Agent component requires a chat generator that supports tools when tools are provided."
            )

        if exit_conditions is None:
            exit_conditions = ["text"]

        if state_schema is not None:
            reserved_used = sorted(set(state_schema) & _INTERNAL_STATE_KEYS.keys())
            if reserved_used:
                raise ValueError(
                    f"state_schema keys {reserved_used} are reserved for Agent internal state and "
                    f"cannot be redefined. Reserved keys: {sorted(_INTERNAL_STATE_KEYS)}."
                )
            _validate_schema(state_schema)
        _validate_prompt_message_blocks(user_prompt, system_prompt)
        if tool_concurrency_limit < 1:
            raise ValueError("tool_concurrency_limit must be greater than or equal to 1.")

        # --- Attributes ---
        self.chat_generator = chat_generator
        # We use an explicit None check for tools b/c testing for truthiness calls __len__, which for SearchableToolset
        # would iterate and prematurely warm it up at init.
        self.tools = tools if tools is not None else []
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.required_variables = required_variables
        self.exit_conditions = exit_conditions
        self.max_agent_steps = max_agent_steps
        self.raise_on_tool_invocation_failure = raise_on_tool_invocation_failure
        self.streaming_callback = streaming_callback
        self.tool_concurrency_limit = tool_concurrency_limit
        self.tool_streaming_callback_passthrough = tool_streaming_callback_passthrough
        self._confirmation_strategies = confirmation_strategies or {}
        self._is_warmed_up = False

        # --- State schema ---
        # shallow copy is sufficient: we only add a top-level "messages" key, never mutate nested values
        self._state_schema = state_schema or {}
        self.state_schema = dict(self._state_schema)
        if self.state_schema.get("messages") is None:
            self.state_schema["messages"] = {"type": list[ChatMessage], "handler": merge_lists}
        for key, config in _INTERNAL_STATE_KEYS.items():
            self.state_schema[key] = dict(config)

        # --- Component I/O ---
        self._run_method_params = _get_run_method_params(self)
        output_types: dict[str, Any] = {"last_message": ChatMessage}
        for param, config in self.state_schema.items():
            output_types[param] = config["type"]
            # Internal state keys are populated internally by the Agent itself and are not exposed as inputs
            if param not in self._run_method_params and param not in _INTERNAL_STATE_KEYS:
                component.set_input_type(self, name=param, type=config["type"], default=None)
        component.set_output_types(self, **output_types)

        # --- Prompt builders ---
        # required_variables starts empty and is populated by _register_prompt_variables once
        # builder.variables are known
        self._user_chat_prompt_builder = (
            ChatPromptBuilder(template=_template_for_role(user_prompt, "user"), required_variables=[])
            if user_prompt is not None
            else None
        )
        self._system_chat_prompt_builder: ChatPromptBuilder | None = None
        if system_prompt is not None:
            self._system_chat_prompt_builder = ChatPromptBuilder(
                template=_template_for_role(system_prompt, "system"), required_variables=[]
            )
        self._register_prompt_variables()

    def _register_prompt_variables(self) -> None:
        """
        Collect variables from both Chat Prompt Builders and register Agent inputs.

        Sets `required_variables` for both Chat Prompt Builders, checks for conflicts with state schema and
        run parameters, and registers component inputs.
        """
        required_variables = self.required_variables

        prompt_builders = [
            builder for builder in (self._system_chat_prompt_builder, self._user_chat_prompt_builder) if builder
        ]
        if required_variables is not None and not any(builder.variables for builder in prompt_builders):
            logger.warning(
                "The parameter required_variables is provided but neither user_prompt nor system_prompt "
                "contains template variables. Either provide a prompt with Jinja2 template variables "
                "or remove required_variables, it has otherwise no effect."
            )

        all_variables: dict[str, list[str]] = {}
        for builder, label in [
            (self._system_chat_prompt_builder, "system_prompt"),
            (self._user_chat_prompt_builder, "user_prompt"),
        ]:
            if builder is not None:
                # set required_variables on the builder, filtered to its own variables
                if required_variables == "*":
                    builder.required_variables = "*"
                elif isinstance(self.required_variables, list):
                    builder.required_variables = [v for v in self.required_variables if v in builder.variables]

                for var_name in builder.variables:
                    all_variables.setdefault(var_name, []).append(label)

        for var_name, sources in all_variables.items():
            prompt_source = " and ".join(sources)
            if var_name in self.state_schema:
                raise ValueError(
                    f"Variable '{var_name}' from {prompt_source} is already defined in the state schema. "
                    "Please rename the variable or remove it from the prompt to avoid conflicts."
                )
            if var_name in self._run_method_params:
                raise ValueError(
                    f"Variable '{var_name}' from {prompt_source} conflicts with input names in the run method. "
                    "Please rename the variable or remove it from the prompt to avoid conflicts."
                )
            if required_variables == "*" or (isinstance(required_variables, list) and var_name in required_variables):
                component.set_input_type(self, name=var_name, type=Any)
            else:
                component.set_input_type(self, name=var_name, type=Any, default=None)

    def warm_up(self) -> None:
        """
        Warm up the Agent.
        """
        if not self._is_warmed_up:
            if hasattr(self.chat_generator, "warm_up"):
                self.chat_generator.warm_up()
            if self.tools:
                warm_up_tools(self.tools)
            self._is_warmed_up = True

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :returns: Dictionary with serialized data.
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
            tool_concurrency_limit=self.tool_concurrency_limit,
            tool_streaming_callback_passthrough=self.tool_streaming_callback_passthrough,
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

        :param data: Dictionary to deserialize from.
        :returns: Deserialized agent.
        """
        init_params = data.get("init_parameters", {})

        deserialize_component_inplace(init_params, key="chat_generator")

        if init_params.get("state_schema") is not None:
            init_params["state_schema"] = _schema_from_dict(init_params["state_schema"])

        if init_params.get("streaming_callback") is not None:
            init_params["streaming_callback"] = deserialize_callable(init_params["streaming_callback"])

        deserialize_tools_or_toolset_inplace(init_params, key="tools")

        if init_params.get("confirmation_strategies") is not None:
            init_params["confirmation_strategies"] = _deserialize_confirmation_strategies(
                init_params["confirmation_strategies"]
            )

        return default_from_dict(cls, data)

    def _create_agent_span(self, tools: ToolsType) -> Any:
        """
        Create a span for the agent run.

        If the agent is running as part of a pipeline, this span will be nested
        under the current active span (the pipeline's component span).

        :param tools: The tools selected for this run (init-time tools or the runtime override
            resolved by `_initialize_fresh_execution`), so the span reflects the tools actually used.
        """
        parent_span = tracing.tracer.current_span()
        return tracing.tracer.trace(
            "haystack.agent.run",
            tags={
                "haystack.agent.max_steps": self.max_agent_steps,
                "haystack.agent.tools": tools,
                "haystack.agent.exit_conditions": self.exit_conditions,
                "haystack.agent.state_schema": _schema_to_dict(self.state_schema),
            },
            parent_span=parent_span,
        )

    def _initialize_fresh_execution(
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None,
        requires_async: bool,
        *,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | list[str] | None = None,
        confirmation_strategy_context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _ExecutionContext:
        """
        Initialize execution context for a fresh run of the agent.

        :param messages: List of ChatMessage objects to start the agent with.
        :param streaming_callback: Optional callback for streaming responses.
        :param requires_async: Whether the agent run requires asynchronous execution.
        :param generation_kwargs: Additional keyword arguments for chat generator. These parameters will
            override the parameters passed during component initialization.
        :param tools: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
            When passing tool names, tools are selected from the Agent's originally configured tools.
        :param confirmation_strategy_context: Optional dictionary for passing request-scoped resources
            to confirmation strategies.
        :param kwargs: Additional data to pass to the State used by the Agent.
        """
        messages = messages or []

        if self._user_chat_prompt_builder is not None:
            user_messages = _render_prompt_messages(
                prompt_builder=self._user_chat_prompt_builder,
                expected_role=ChatRole.USER,
                prompt_label="user_prompt",
                kwargs=kwargs,
            )
            messages = messages + user_messages

        if self._system_chat_prompt_builder is not None:
            system_messages = _render_prompt_messages(
                prompt_builder=self._system_chat_prompt_builder,
                expected_role=ChatRole.SYSTEM,
                prompt_label="system_prompt",
                kwargs=kwargs,
            )
            messages = system_messages + messages

        if all(m.is_from(ChatRole.SYSTEM) for m in messages):
            logger.warning("All messages provided to the Agent component are system messages. This is not recommended.")

        selected_tools = self._select_tools(tools)
        flat_tools = flatten_tools_or_toolsets(selected_tools)
        # Validate tool support once for the run (covers both init-time and runtime tools)
        if flat_tools and not self._chat_generator_supports_tools:
            raise TypeError(
                f"{type(self.chat_generator).__name__} does not accept tools parameter in its run method. "
                "The Agent component requires a chat generator that supports tools when tools are provided."
            )

        state_kwargs: dict[str, Any] = {key: kwargs[key] for key in self.state_schema.keys() if key in kwargs}
        state = State(schema=self.state_schema, data=state_kwargs)
        state.set("messages", messages)
        state.set("step_count", 0)
        state.set("token_usage", {})
        state.set("tool_call_counts", {tool.name: 0 for tool in flat_tools})

        streaming_callback = select_streaming_callback(  # type: ignore[call-overload]
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=requires_async
        )
        generator_inputs: dict[str, Any] = {}
        if streaming_callback is not None:
            generator_inputs["streaming_callback"] = streaming_callback
        if generation_kwargs is not None:
            generator_inputs["generation_kwargs"] = generation_kwargs

        tool_execution_inputs: dict[str, Any] = {
            "raise_on_failure": self.raise_on_tool_invocation_failure,
            "streaming_callback": streaming_callback,
            "max_workers": self.tool_concurrency_limit,
            "enable_streaming_callback_passthrough": self.tool_streaming_callback_passthrough,
        }

        return _ExecutionContext(
            state=state,
            tools=selected_tools,
            chat_generator_inputs=generator_inputs,
            tool_execution_inputs=tool_execution_inputs,
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
            # Per-run tools are not covered by the Agent's own warm_up(), so warm them up here.
            # warm_up() is expected to be idempotent, so re-warming on every run is cheap.
            warm_up_tools(tools)
            return tools

        if isinstance(tools, list):
            selected = cast(list[Tool | Toolset], tools)  # mypy can't narrow the Union type from isinstance check
            # Per-run tools are not covered by the Agent's own warm_up(), so warm them up here.
            # warm_up() is expected to be idempotent, so re-warming on every run is cheap.
            warm_up_tools(selected)
            return selected

        raise TypeError(
            "tools must be a list of Tool and/or Toolset objects, a Toolset, or a list of tool names (strings)."
        )

    def run(
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        *,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | list[str] | None = None,
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
        :param tools: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
            When passing tool names, tools are selected from the Agent's originally configured tools.
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
            - "step_count": The number of steps the agent ran. A step is one chat-generator call plus the
              execution of every tool call the model requested in that call (if any). The counter is incremented
              after each step completes, including the final step that hits an exit condition or `max_agent_steps`.
            - "token_usage": Aggregated token usage from every LLM call in the run, summed from each LLM message's
              `meta["usage"]`.
            - "tool_call_counts": Mapping of tool name to the number of times that tool was invoked.
            - Any additional keys defined in the `state_schema`.
        """
        agent_inputs = {"messages": messages, "streaming_callback": streaming_callback, **kwargs}
        if not self._is_warmed_up:
            self.warm_up()

        exe_context = self._initialize_fresh_execution(
            messages=messages,
            streaming_callback=streaming_callback,
            requires_async=False,
            generation_kwargs=generation_kwargs,
            tools=tools,
            confirmation_strategy_context=confirmation_strategy_context,
            **kwargs,
        )

        with self._create_agent_span(exe_context.tools) as span:
            span.set_content_tag("haystack.agent.input", agent_inputs)
            while exe_context.counter < self.max_agent_steps:
                if not self._run_step(exe_context, span):
                    break
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
        streaming_callback: StreamingCallbackT | None = None,
        *,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | list[str] | None = None,
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
        :param tools: Optional list of Tool objects, a Toolset, or list of tool names to use for this run.
        :param kwargs: Additional data to pass to the State schema used by the Agent.
            The keys must match the schema defined in the Agent's `state_schema`.
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
            - "step_count": The number of steps the agent ran. A step is one chat-generator call plus the
              execution of every tool call the model requested in that call (if any). The counter is incremented
              after each step completes, including the final step that hits an exit condition or `max_agent_steps`.
            - "token_usage": Aggregated token usage from every LLM call in the run, summed from each LLM message's
              `meta["usage"]`.
            - "tool_call_counts": Mapping of tool name to the number of times that tool was invoked.
            - Any additional keys defined in the `state_schema`.
        """
        agent_inputs = {"messages": messages, "streaming_callback": streaming_callback, **kwargs}
        if not self._is_warmed_up:
            self.warm_up()

        exe_context = self._initialize_fresh_execution(
            messages=messages,
            streaming_callback=streaming_callback,
            requires_async=True,
            tools=tools,
            generation_kwargs=generation_kwargs,
            confirmation_strategy_context=confirmation_strategy_context,
            **kwargs,
        )

        with self._create_agent_span(exe_context.tools) as span:
            span.set_content_tag("haystack.agent.input", agent_inputs)
            while exe_context.counter < self.max_agent_steps:
                if not await self._run_step_async(exe_context, span):
                    break
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

    def _run_step(self, exe_context: _ExecutionContext, agent_span: tracing.Span) -> bool:
        """Execute one agent step. Returns True to continue the loop, False to stop."""
        with tracing.tracer.trace(
            "haystack.agent.step", tags={"haystack.agent.step": exe_context.counter}, parent_span=agent_span
        ) as step_span:
            # Re-flatten the tools every step so dynamic toolsets (e.g. SearchableToolset) surface tools discovered in
            # earlier steps. Validate names here so duplicates fail before starting the step.
            current_tools = flatten_tools_or_toolsets(exe_context.tools)
            _check_duplicate_tool_names(current_tools)

            chat_generator_inputs = {
                "messages": exe_context.state.data["messages"],
                **exe_context.chat_generator_inputs,
            }
            if current_tools:
                chat_generator_inputs["tools"] = current_tools
            with tracing.tracer.trace("haystack.agent.step.llm", parent_span=step_span) as llm_span:
                llm_span.set_content_tag("haystack.agent.step.llm.input", chat_generator_inputs)
                result = self.chat_generator.run(**chat_generator_inputs)
                llm_span.set_content_tag("haystack.agent.step.llm.output", result)
            llm_messages = result["replies"]
            exe_context.state.set("messages", llm_messages)
            _record_llm_usage(exe_context.state, llm_messages)

            if not any(msg.tool_call for msg in llm_messages) or not current_tools:
                exe_context.counter += 1
                exe_context.state.set("step_count", exe_context.counter)
                return False

            modified_tool_call_messages, new_chat_history = _process_confirmation_strategies(
                confirmation_strategies=self._confirmation_strategies,
                messages_with_tool_calls=llm_messages,
                state=exe_context.state,
                confirmation_strategy_context=exe_context.confirmation_strategy_context,
                tools=current_tools,
                streaming_callback=exe_context.tool_execution_inputs["streaming_callback"],
                enable_streaming_passthrough=exe_context.tool_execution_inputs["enable_streaming_callback_passthrough"],
            )
            exe_context.state.set(key="messages", value=new_chat_history, handler_override=replace_values)

            tool_execution_inputs = {
                "messages": modified_tool_call_messages,
                "state": exe_context.state,
                **exe_context.tool_execution_inputs,
                "tools": current_tools,
            }
            with tracing.tracer.trace("haystack.agent.step.tool", parent_span=step_span) as tool_span:
                tool_span.set_content_tag("haystack.agent.step.tool.input", tool_execution_inputs)
                tool_messages, exe_context.state = _run_tool(**tool_execution_inputs)
                tool_span.set_content_tag(
                    "haystack.agent.step.tool.output", {"tool_messages": tool_messages, "state": exe_context.state}
                )
            exe_context.state.set("messages", tool_messages)
            _record_tool_calls(exe_context.state, tool_messages)

            exe_context.counter += 1
            exe_context.state.set("step_count", exe_context.counter)
            exit_triggered = self.exit_conditions != ["text"] and self._check_exit_conditions(
                llm_messages, tool_messages
            )
            return not exit_triggered

    async def _run_step_async(self, exe_context: _ExecutionContext, agent_span: tracing.Span) -> bool:
        """Execute one agent step asynchronously. Returns True to continue the loop, False to stop."""
        with tracing.tracer.trace(
            "haystack.agent.step", tags={"haystack.agent.step": exe_context.counter}, parent_span=agent_span
        ) as step_span:
            # Re-flatten the tools every step so dynamic toolsets (e.g. SearchableToolset) surface tools discovered in
            # earlier steps. Validate names here so duplicates fail before starting the step.
            current_tools = flatten_tools_or_toolsets(exe_context.tools)
            _check_duplicate_tool_names(current_tools)

            chat_generator_inputs = {
                "messages": exe_context.state.data["messages"],
                **exe_context.chat_generator_inputs,
            }
            if current_tools:
                chat_generator_inputs["tools"] = current_tools
            with tracing.tracer.trace("haystack.agent.step.llm", parent_span=step_span) as llm_span:
                llm_span.set_content_tag("haystack.agent.step.llm.input", chat_generator_inputs)
                # For sync-only generators, _run_component_async dispatches to a thread via asyncio.to_thread,
                # which copies the current contextvars context — preserving the active tracing span.
                result = await _run_component_async(self.chat_generator, **chat_generator_inputs)
                llm_span.set_content_tag("haystack.agent.step.llm.output", result)
            llm_messages = result["replies"]
            exe_context.state.set("messages", llm_messages)
            _record_llm_usage(exe_context.state, llm_messages)

            if not any(msg.tool_call for msg in llm_messages) or not current_tools:
                exe_context.counter += 1
                exe_context.state.set("step_count", exe_context.counter)
                return False

            modified_tool_call_messages, new_chat_history = await _process_confirmation_strategies_async(
                confirmation_strategies=self._confirmation_strategies,
                messages_with_tool_calls=llm_messages,
                tools=current_tools,
                state=exe_context.state,
                streaming_callback=exe_context.tool_execution_inputs["streaming_callback"],
                enable_streaming_passthrough=exe_context.tool_execution_inputs["enable_streaming_callback_passthrough"],
                confirmation_strategy_context=exe_context.confirmation_strategy_context,
            )
            exe_context.state.set(key="messages", value=new_chat_history, handler_override=replace_values)

            tool_execution_inputs = {
                "messages": modified_tool_call_messages,
                "state": exe_context.state,
                **exe_context.tool_execution_inputs,
                "tools": current_tools,
            }
            with tracing.tracer.trace("haystack.agent.step.tool", parent_span=step_span) as tool_span:
                tool_span.set_content_tag("haystack.agent.step.tool.input", tool_execution_inputs)
                tool_messages, exe_context.state = await _run_tool_async(**tool_execution_inputs)
                tool_span.set_content_tag(
                    "haystack.agent.step.tool.output", {"tool_messages": tool_messages, "state": exe_context.state}
                )
            exe_context.state.set("messages", tool_messages)
            _record_tool_calls(exe_context.state, tool_messages)

            exe_context.counter += 1
            exe_context.state.set("step_count", exe_context.counter)
            exit_triggered = self.exit_conditions != ["text"] and self._check_exit_conditions(
                llm_messages, tool_messages
            )
            return not exit_triggered

    def _check_exit_conditions(self, llm_messages: list[ChatMessage], tool_messages: list[ChatMessage]) -> bool:
        """
        Decide whether the agent should stop looping.

        Returns True when the model called at least one tool listed in `exit_conditions` and
        that tool did not error. Every tool call in the message is checked, so the order of
        parallel tool calls does not matter.

        :param llm_messages: List of messages from the LLM
        :param tool_messages: List of messages from tool execution.
        :return: True if an exit condition is met and there are no errors, False otherwise
        """
        matched_exit_conditions: set[str] = set()
        has_errors = False

        for msg in llm_messages:
            for tool_call in msg.tool_calls:
                if tool_call.tool_name not in self.exit_conditions:
                    continue
                matched_exit_conditions.add(tool_call.tool_name)

                # Check if any error is specifically from the tool matching the exit condition
                tool_errors = [
                    tool_msg.tool_call_result.error
                    for tool_msg in tool_messages
                    if tool_msg.tool_call_result is not None
                    and tool_msg.tool_call_result.origin.tool_name == tool_call.tool_name
                ]
                if any(tool_errors):
                    has_errors = True
                    # No need to check further if we found an error
                    break
            if has_errors:
                break

        # Only return True if at least one exit condition was matched AND none had errors
        return bool(matched_exit_conditions) and not has_errors
