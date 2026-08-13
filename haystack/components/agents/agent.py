# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
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
from haystack.components.agents.utils import (
    _record_context_tokens,
    _record_llm_usage,
    _record_tool_calls,
    _render_prompt_messages,
    _select_tools_by_name,
    _spawn_tools,
    _template_for_role,
    _validate_prompt_message_blocks,
)
from haystack.components.builders import ChatPromptBuilder
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.serialization import component_to_dict, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole, StreamingCallbackT, select_streaming_callback
from haystack.hooks.invocation import _run_hooks, _run_hooks_async
from haystack.hooks.protocol import (
    AFTER_RUN,
    AFTER_TOOL,
    BEFORE_LLM,
    BEFORE_RUN,
    BEFORE_TOOL,
    ON_EXIT,
    VALID_HOOK_POINTS,
    Hook,
    HookPoint,
)
from haystack.hooks.utils import (
    _deserialize_hooks_dictionary,
    _serialize_hooks_dictionary,
    close_hooks,
    close_hooks_async,
    warm_up_hooks,
    warm_up_hooks_async,
)
from haystack.tools import (
    Toolset,
    ToolsType,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    flatten_tools_or_toolsets,
    serialize_tools_or_toolset,
    warm_up_tools,
)
from haystack.utils.async_utils import _execute_component_async
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable
from haystack.utils.deserialization import deserialize_component_inplace

logger = logging.getLogger(__name__)

# `exit_reason` values the Agent sets when it stops without a tool exit condition: a tool-call-free reply, or the
# `max_agent_steps` budget running out. A tool exit condition instead reports the tool's name.
_EXIT_REASON_TEXT = "text"
_EXIT_REASON_MAX_STEPS = "max_agent_steps"

# Run-metadata state keys the Agent populates automatically during a run. Users may not define them in their own
# `state_schema`, and they are exposed as Agent outputs only (not inputs).
_RUN_METADATA_STATE_KEYS: dict[str, dict[str, Any]] = {
    "step_count": {"type": int, "handler": replace_values},
    "token_usage": {"type": dict[str, Any], "handler": replace_values},
    "tool_call_counts": {"type": dict[str, int], "handler": replace_values},
    "exit_reason": {"type": str, "handler": replace_values},
}

# Internal state keys the Agent manages for run control and hooks. Like run-metadata keys they are reserved and cannot
# be redefined by users, but unlike them they are NOT exposed as Agent inputs or outputs (purely internal state):
# - `continue_run`: set by an `on_exit` hook to keep the Agent running instead of stopping (re-read each exit attempt).
# - `tools`: the flattened tools available in the current step, so a hook can inspect them (e.g. HITL confirmation).
# - `hook_context`: per-run request-scoped resources passed to `run`/`run_async` for hooks to read.
# - `context_tokens`: approximate current context-window size, refreshed after each LLM call, for hooks to read
#   (e.g. a `before_llm` hook that triggers compaction once the context grows too large). Kept internal rather than
#   exposed as an output because it is a best-effort snapshot; see `_record_context_tokens`.
_INTERNAL_STATE_KEYS: dict[str, dict[str, Any]] = {
    "continue_run": {"type": bool, "handler": replace_values},
    "tools": {"type": list, "handler": replace_values},
    "hook_context": {"type": dict[str, Any], "handler": replace_values},
    "context_tokens": {"type": int, "handler": replace_values},
}


def _get_run_method_params(instance: "Agent") -> set[str]:
    """Derive the parameter names of the Agent.run method via introspection."""
    sig = inspect.signature(instance.run)
    return {name for name, p in sig.parameters.items() if p.kind != inspect.Parameter.VAR_KEYWORD}


def _public_outputs(state: State) -> dict[str, Any]:
    """Return the State data excluding the internal state keys (i.e. the Agent's user-facing outputs)."""
    return {key: value for key, value in state.data.items() if key not in _INTERNAL_STATE_KEYS}


def _validate_hooks(hooks: dict[HookPoint, list[Hook]]) -> None:
    """
    Validate a hooks mapping: known hook points, real Hook objects, and hook-point restrictions.

    :param hooks: Mapping of hook point to the hooks registered under it.
    :raises ValueError: If a hook point is unknown, or a hook is registered under a hook point it does not support.
    :raises TypeError: If a registered hook has no callable `run(state)`.
    """
    for hook_point, hook_list in hooks.items():
        if hook_point not in VALID_HOOK_POINTS:
            raise ValueError(
                f"Invalid hook point '{hook_point}'. Valid hook points are: {', '.join(VALID_HOOK_POINTS)}."
            )
        for h in hook_list:
            if not callable(getattr(h, "run", None)):
                if callable(h):
                    raise TypeError(
                        f"Hook registered for hook point '{hook_point}' is callable but is not a Hook object. "
                        "If it is a function, wrap it with the @hook decorator."
                    )
                raise TypeError(
                    f"Hook registered for hook point '{hook_point}' must have a callable 'run(state)', "
                    f"got an object of type '{type(h).__name__}'."
                )
            # A hook may declare `allowed_hook_points` to restrict where it can run (e.g. ConfirmationHook only
            # makes sense at "before_tool"). Hooks without it can be registered under any hook point.
            allowed_points = getattr(h, "allowed_hook_points", None)
            if allowed_points is not None and hook_point not in allowed_points:
                raise ValueError(
                    f"Hook of type '{type(h).__name__}' is registered under hook point '{hook_point}' but only "
                    f"supports: {', '.join(allowed_points)}."
                )


def _consume_continue_run(state: State) -> bool:
    """Return the `continue_run` control flag and reset it so it does not carry over to the next exit attempt."""
    should_continue = state.data["continue_run"]
    state.set("continue_run", False)
    return should_continue


def _is_text_exit(messages: list[ChatMessage]) -> bool:
    """
    Return whether `messages` end in a plain assistant text reply with no tool calls anywhere in the batch.

    This is the "no tool call" exit for the model's own replies. The last message must be a non-empty assistant text
    message, so an invalid response (e.g. one with no tool calls and no text) does not trigger an exit.
    """
    if not messages:
        return False
    last = messages[-1]
    return not any(m.tool_call for m in messages) and last.is_from(ChatRole.ASSISTANT) and bool(last.text)


def _pending_tool_call_messages_from_state(state: State) -> list[ChatMessage]:
    """
    Return the pending tool-call message after `before_tool` hooks have run.

    `before_tool` hooks may mutate `state.data["messages"]`. After they run, the Agent intentionally inspects only the
    current last message in the state. If that message has tool calls, those calls are executed. If it has no tool
    calls, no tools run for the step, no tool-based exit condition is triggered, and the Agent loops back to the next
    LLM call unless `max_agent_steps` has been reached.
    """
    messages = state.data.get("messages") or []
    if not messages:
        return []
    last_message = messages[-1]
    return [last_message] if last_message.tool_calls else []


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
    """

    state: State
    tools: ToolsType
    chat_generator_inputs: dict
    tool_execution_inputs: dict
    counter: int = 0


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
    )

    # The template variables 'language' and 'document' become inputs to the run method
    result = agent.run(
        messages=[],
        language="French",
        document="The weather is lovely today and the sun is shining.",
    )

    print(result["last_message"].text)
    ```

    #### Using hooks to influence the run loop

    Hooks are callables that receive the live `State` and run at specific points in the Agent loop:

    - `before_llm`: runs before each chat-generator call.
    - `before_tool`: runs after the model requests tool calls, before any tools run. After these hooks run, the Agent
      re-reads the current last message from `state.data["messages"]`. If that message has tool calls, those calls are
      executed. If it has no tool calls, no tools run for that step, no tool-based exit condition is triggered, and the
      Agent loops back to the next LLM call unless `max_agent_steps` has been reached.
    - `after_tool`: runs after tools execute, once their result messages are in `state.data["messages"]`, before the
      exit check and the next LLM call. Use it to rewrite the freshly produced tool-result messages (e.g. offload,
      redact, truncate, or summarize results). It does not run on the plain-text exit step, where no tools run.
    - `on_exit`: runs when the Agent is about to stop on an exit condition. An `on_exit` hook can keep the Agent
      running by setting `state.set("continue_run", True)`.

    Use the `@hook` decorator to build a hook from a function. This `on_exit` hook keeps the Agent running until a
    required tool has been called.

    ```python
    from haystack.components.agents import Agent
    from haystack.components.agents.state import State
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.hooks import hook
    from haystack.tools import tool
    from typing import Annotated


    @tool
    def save_result(content: Annotated[str, "The result to save"]) -> str:
        \"\"\"Save the final result.\"\"\"
        # Placeholder: would persist `content` to a database or the file system
        return "saved"


    @hook
    def require_save(state: State) -> None:
        if state.get("tool_call_counts", {}).get("save_result", 0) == 0:
            state.set("messages", [ChatMessage.from_system("Call `save_result` before finishing.")])
            state.set("continue_run", True)  # keep the Agent running instead of stopping


    agent = Agent(
        chat_generator=OpenAIChatGenerator(),
        tools=[save_result],
        hooks={"on_exit": [require_save]},
    )
    ```

    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        chat_generator: ChatGenerator,
        tools: ToolsType | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        required_variables: list[str] | Literal["*"] | None = "*",
        exit_conditions: list[str] | None = None,
        state_schema: dict[str, Any] | None = None,
        max_agent_steps: int = 100,
        streaming_callback: StreamingCallbackT | None = None,
        raise_on_tool_invocation_failure: bool = False,
        tool_concurrency_limit: int = 4,
        tool_streaming_callback_passthrough: bool = False,
        hooks: dict[HookPoint, list[Hook]] | None = None,
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
            If set to `"*"`, all variables found in the prompts are required. Defaults to `"*"`.
            Set to `None` to make all variables optional; missing ones render as empty strings.
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
        :param hooks: A dictionary mapping a hook point to a list of hooks the Agent runs at that point. Each hook
            receives the live `State` and influences the run by mutating it in place; hooks for a hook point run in
            list order. Valid hook points are:
            - "before_run": Runs once per run, after the state is initialized and before the first chat-generator
              call. Use it to rewrite the initial messages or seed state (e.g. turn the user query into a task
              brief) without re-running on every step like "before_llm" does.
            - "before_llm": Runs before each chat-generator call.
            - "before_tool": Runs after the model requests tool calls, before any tools run. After these hooks run,
              the Agent re-reads the current last message from `state.data["messages"]`. If that message contains tool
              calls, those calls are executed. If it does not, no tools run for that step, no tool-based exit condition
              is triggered, and the Agent loops back to the next LLM call unless `max_agent_steps` has been reached.
            - "after_tool": Runs after tools execute, once their result messages are in `state.data["messages"]`,
              before the exit check and the next LLM call. Use it to rewrite the freshly produced tool-result messages
              (e.g. offload, redact, truncate, or summarize results). It does not run on the plain-text exit step,
              where no tools run.
            - "on_exit": Runs when the Agent is about to stop on an exit condition. An "on_exit" hook can keep the
              Agent running by setting the `continue_run` control flag (`state.set("continue_run", True)`), usually
              alongside a message telling the model what to do next. "on_exit" hooks run when the Agent stops on an
              exit condition, but not when it stops because `max_agent_steps` is reached.
            - "after_run": Runs once per run, after the step loop has ended and before the Agent builds its return
              value — regardless of whether the run stopped on an exit condition or because `max_agent_steps` was
              reached (unlike "on_exit"). Mutations to the state (e.g. appending a final message) are reflected in
              the returned `messages` / `last_message` and `state_schema` outputs. Setting `continue_run` here has
              no effect.
        :raises TypeError: If the chat_generator does not support tools parameter in its run method.
        :raises ValueError: If any `user_prompt` variable overlaps with the `state_schema` or `run` method parameters,
            if a hook is registered under an unknown hook point, or if a hook is registered under a hook point it does
            not support (via its `allowed_hook_points`).
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
            reserved_keys = _RUN_METADATA_STATE_KEYS.keys() | _INTERNAL_STATE_KEYS.keys()
            reserved_used = sorted(set(state_schema) & reserved_keys)
            if reserved_used:
                raise ValueError(
                    f"state_schema keys {reserved_used} are reserved for Agent internal state and "
                    f"cannot be redefined. Reserved keys: {sorted(reserved_keys)}."
                )
            _validate_schema(state_schema)
        _validate_prompt_message_blocks(user_prompt, system_prompt)
        if tool_concurrency_limit < 1:
            raise ValueError("tool_concurrency_limit must be greater than or equal to 1.")

        hooks = hooks or {}
        _validate_hooks(hooks)

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
        self.hooks = hooks

        # --- State schema ---
        # shallow copy is sufficient: we only add a top-level "messages" key, never mutate nested values
        self.state_schema = state_schema or {}
        self.resolved_state_schema = dict(self.state_schema)
        if self.resolved_state_schema.get("messages") is None:
            self.resolved_state_schema["messages"] = {"type": list[ChatMessage], "handler": merge_lists}
        for key, config in {**_RUN_METADATA_STATE_KEYS, **_INTERNAL_STATE_KEYS}.items():
            self.resolved_state_schema[key] = dict(config)

        # --- Component I/O ---
        self._run_method_params = _get_run_method_params(self)
        output_types: dict[str, Any] = {"last_message": ChatMessage}
        for param, config in self.resolved_state_schema.items():
            # Internal keys are run-control / hook-facing state, not exposed as inputs or outputs.
            if param in _INTERNAL_STATE_KEYS:
                continue
            output_types[param] = config["type"]
            # Run-metadata keys are populated by the Agent itself and exposed as outputs only, not inputs.
            if param not in self._run_method_params and param not in _RUN_METADATA_STATE_KEYS:
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
        if isinstance(required_variables, list) and not any(builder.variables for builder in prompt_builders):
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
            if var_name in self.resolved_state_schema:
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
        """Warm up the tools, hooks, and the underlying chat generator."""
        warm_up_tools(tools=self.tools)
        warm_up_hooks(self.hooks)
        if hasattr(self.chat_generator, "warm_up"):
            self.chat_generator.warm_up()

    async def warm_up_async(self) -> None:
        """Warm up the tools, hooks, and the underlying chat generator on the serving event loop."""
        warm_up_tools(tools=self.tools)
        await warm_up_hooks_async(self.hooks)
        if hasattr(self.chat_generator, "warm_up_async"):
            await self.chat_generator.warm_up_async()
        elif hasattr(self.chat_generator, "warm_up"):
            self.chat_generator.warm_up()

    def close(self) -> None:
        """Release the hooks' and the underlying chat generator's resources."""
        close_hooks(self.hooks)
        if hasattr(self.chat_generator, "close"):
            self.chat_generator.close()

    async def close_async(self) -> None:
        """Release the hooks' and the underlying chat generator's async resources."""
        await close_hooks_async(self.hooks)
        if hasattr(self.chat_generator, "close_async"):
            await self.chat_generator.close_async()
        elif hasattr(self.chat_generator, "close"):
            self.chat_generator.close()

    def clone(self, **overrides: Any) -> "Agent":
        """
        Return a new Agent configured like this one, with the given init parameters replaced.

        :param overrides: Init parameters to replace, e.g. `agent.clone(system_prompt="...")`.
        :returns: The new Agent.
        """
        init_params = inspect.signature(type(self).__init__).parameters
        params: dict[str, Any] = {name: getattr(self, name) for name in init_params if name != "self"}
        return type(self)(**{**params, **overrides})

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
            state_schema=_schema_to_dict(self.state_schema),
            max_agent_steps=self.max_agent_steps,
            streaming_callback=serialize_callable(self.streaming_callback) if self.streaming_callback else None,
            raise_on_tool_invocation_failure=self.raise_on_tool_invocation_failure,
            tool_concurrency_limit=self.tool_concurrency_limit,
            tool_streaming_callback_passthrough=self.tool_streaming_callback_passthrough,
            hooks=_serialize_hooks_dictionary(self.hooks) if self.hooks else None,
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

        if init_params.get("hooks") is not None:
            init_params["hooks"] = _deserialize_hooks_dictionary(init_params["hooks"])

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
                "haystack.agent.state_schema": _schema_to_dict(self.resolved_state_schema),
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
        hook_context: dict[str, Any] | None = None,
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
        :param hook_context: Optional dictionary of request-scoped resources made available to hooks via
            `state.data.get("hook_context")`.
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

        selected_tools = self._select_tools(tools=tools)
        flat_tools = flatten_tools_or_toolsets(tools=selected_tools)
        # Validate tool support once for the run (covers both init-time and runtime tools)
        if flat_tools and not self._chat_generator_supports_tools:
            raise TypeError(
                f"{type(self.chat_generator).__name__} does not accept tools parameter in its run method. "
                "The Agent component requires a chat generator that supports tools when tools are provided."
            )

        state_kwargs: dict[str, Any] = {key: kwargs[key] for key in self.resolved_state_schema.keys() if key in kwargs}
        state = State(schema=self.resolved_state_schema, data=state_kwargs)
        state.set("messages", messages)
        state.set("step_count", 0)
        state.set("token_usage", {})
        state.set("context_tokens", 0)
        state.set("tool_call_counts", {tool.name: 0 for tool in flat_tools})
        state.set("exit_reason", None)
        state.set("continue_run", False)
        state.set("hook_context", hook_context or {})

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
        # Toolsets are spawned per run (see _spawn_tools / _select_tools_by_name) so concurrent runs
        # sharing the same configured Toolset don't corrupt each other's run-scoped state.
        if tools is None:
            return _spawn_tools(tools=self.tools)

        if isinstance(tools, list) and all(isinstance(t, str) for t in tools):
            return _select_tools_by_name(self.tools, cast(list[str], tools))

        if isinstance(tools, (Toolset, list)):
            selected = cast(ToolsType, tools)  # mypy can't narrow the Union type from the isinstance checks
            # Per-run tools are not covered by the Agent's own warm_up(), so warm them up here.
            # warm_up() is expected to be idempotent, so re-warming on every run is cheap.
            warm_up_tools(tools=selected)
            return _spawn_tools(tools=selected)

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
        hook_context: dict[str, Any] | None = None,
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
        :param hook_context: Optional dictionary of request-scoped resources made available to hooks via
            `state.data.get("hook_context")`. Useful in web/server environments to provide per-request objects
            (e.g., WebSocket connections, async queues, Redis pub/sub clients) that a hook can use, for
            example a ConfirmationHook driving non-blocking user interaction.
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
            - "exit_reason": Why the Agent stopped, useful for routing the output downstream (e.g. with a
              `ConditionalRouter`). One of: `"text"` (the model returned a reply with no tool calls), the name of
              the tool that satisfied a tool exit condition (in which case `last_message` is that tool's result),
              or `"max_agent_steps"` (the Agent hit `max_agent_steps` before meeting an exit condition).
            - Any additional keys defined in the `state_schema`.
        """
        agent_inputs = {"messages": messages, "streaming_callback": streaming_callback, **kwargs}
        self.warm_up()

        exe_context = self._initialize_fresh_execution(
            messages=messages,
            streaming_callback=streaming_callback,
            requires_async=False,
            generation_kwargs=generation_kwargs,
            tools=tools,
            hook_context=hook_context,
            **kwargs,
        )

        with self._create_agent_span(tools=exe_context.tools) as span:
            span.set_content_tag("haystack.agent.input", agent_inputs)
            _run_hooks(hooks=self.hooks, hook_point=BEFORE_RUN, state=exe_context.state)
            # A before_run hook can restore a saved State, so resume the execution counter from its step count.
            exe_context.counter = exe_context.state.data.get("step_count", 0)
            while exe_context.counter < self.max_agent_steps:
                if not self._run_step(exe_context=exe_context, agent_span=span):
                    break
            else:
                # Reached only when the loop ends without a `break`. A `break` means a step already set its own
                # `exit_reason`, so this branch runs only when `max_agent_steps` is why the Agent stopped.
                logger.warning(
                    "Agent reached maximum agent steps of {max_agent_steps}, stopping.",
                    max_agent_steps=self.max_agent_steps,
                )
                exe_context.state.set("exit_reason", _EXIT_REASON_MAX_STEPS)
            _run_hooks(hooks=self.hooks, hook_point=AFTER_RUN, state=exe_context.state)
            result = _public_outputs(state=exe_context.state)
            if msgs := result.get("messages"):
                result["last_message"] = msgs[-1]
            span.set_content_tag("haystack.agent.output", result)
            span.set_tag("haystack.agent.steps_taken", exe_context.counter)

        return result

    async def run_async(
        self,
        messages: list[ChatMessage],
        streaming_callback: StreamingCallbackT | None = None,
        *,
        generation_kwargs: dict[str, Any] | None = None,
        tools: ToolsType | list[str] | None = None,
        hook_context: dict[str, Any] | None = None,
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
        :param hook_context: Optional dictionary of request-scoped resources made available to hooks via
            `state.data.get("hook_context")`. Useful in web/server environments to provide per-request objects
            (e.g., WebSocket connections, async queues, Redis pub/sub clients) that a hook can use, for
            example a ConfirmationHook driving non-blocking user interaction.
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
            - "exit_reason": Why the Agent stopped, useful for routing the output downstream (e.g. with a
              `ConditionalRouter`). One of: `"text"` (the model returned a reply with no tool calls), the name of
              the tool that satisfied a tool exit condition (in which case `last_message` is that tool's result),
              or `"max_agent_steps"` (the Agent hit `max_agent_steps` before meeting an exit condition).
            - Any additional keys defined in the `state_schema`.
        """
        agent_inputs = {"messages": messages, "streaming_callback": streaming_callback, **kwargs}
        await self.warm_up_async()

        exe_context = self._initialize_fresh_execution(
            messages=messages,
            streaming_callback=streaming_callback,
            requires_async=True,
            tools=tools,
            generation_kwargs=generation_kwargs,
            hook_context=hook_context,
            **kwargs,
        )

        with self._create_agent_span(tools=exe_context.tools) as span:
            span.set_content_tag("haystack.agent.input", agent_inputs)
            await _run_hooks_async(hooks=self.hooks, hook_point=BEFORE_RUN, state=exe_context.state)
            # A before_run hook can restore a saved State, so resume the execution counter from its step count.
            exe_context.counter = exe_context.state.data.get("step_count", 0)
            while exe_context.counter < self.max_agent_steps:
                if not await self._run_step_async(exe_context=exe_context, agent_span=span):
                    break
            else:
                # Reached only when the loop ends without a `break`. A `break` means a step already set its own
                # `exit_reason`, so this branch runs only when `max_agent_steps` is why the Agent stopped.
                logger.warning(
                    "Agent reached maximum agent steps of {max_agent_steps}, stopping.",
                    max_agent_steps=self.max_agent_steps,
                )
                exe_context.state.set("exit_reason", _EXIT_REASON_MAX_STEPS)
            await _run_hooks_async(hooks=self.hooks, hook_point=AFTER_RUN, state=exe_context.state)
            result = _public_outputs(state=exe_context.state)
            if msgs := result.get("messages"):
                result["last_message"] = msgs[-1]
            span.set_content_tag("haystack.agent.output", result)
            span.set_tag("haystack.agent.steps_taken", exe_context.counter)

        return result

    def _run_step(self, exe_context: _ExecutionContext, agent_span: tracing.Span) -> bool:
        """Execute one agent step. Returns True to continue the loop, False to stop."""
        with tracing.tracer.trace(
            "haystack.agent.step", tags={"haystack.agent.step": exe_context.counter}, parent_span=agent_span
        ) as step_span:
            # Re-flatten the tools every step so dynamic toolsets (e.g. SearchableToolset) surface tools discovered in
            # earlier steps. Validate names here so duplicates fail before starting the step.
            current_tools = flatten_tools_or_toolsets(tools=exe_context.tools)
            _check_duplicate_tool_names(tools=current_tools)
            # Expose the current tools to hooks (e.g. ConfirmationHook) via State.
            exe_context.state.set("tools", current_tools, handler_override=replace_values)

            _run_hooks(hooks=self.hooks, hook_point=BEFORE_LLM, state=exe_context.state)
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
            _record_llm_usage(state=exe_context.state, llm_messages=llm_messages)
            _record_context_tokens(state=exe_context.state, llm_messages=llm_messages)

            # Stop on the "no tool call" exit: no tools available, or a plain assistant text reply (see _is_text_exit).
            if not current_tools or _is_text_exit(messages=llm_messages):
                exe_context.counter += 1
                exe_context.state.set("step_count", exe_context.counter)
                exe_context.state.set("exit_reason", _EXIT_REASON_TEXT)
                return self._continue_after_exit_hooks(exe_context=exe_context)

            _run_hooks(hooks=self.hooks, hook_point=BEFORE_TOOL, state=exe_context.state)
            # Re-read the pending tool calls from State so that any rewrites a before_tool hook made (e.g.
            # ConfirmationHook rejecting or modifying calls) are honored by the executor.
            pending_tool_call_messages = _pending_tool_call_messages_from_state(exe_context.state)

            tool_execution_inputs = {
                "messages": pending_tool_call_messages,
                "state": exe_context.state,
                **exe_context.tool_execution_inputs,
                "tools": current_tools,
            }
            tool_messages, exe_context.state = _run_tool(**tool_execution_inputs)
            exe_context.state.set("messages", tool_messages)
            _record_tool_calls(state=exe_context.state, tool_messages=tool_messages)
            _run_hooks(hooks=self.hooks, hook_point=AFTER_TOOL, state=exe_context.state)

            exe_context.counter += 1
            exe_context.state.set("step_count", exe_context.counter)
            exit_condition_tool = (
                None
                if self.exit_conditions == ["text"]
                else self._check_exit_conditions(llm_messages=pending_tool_call_messages, tool_messages=tool_messages)
            )
            if exit_condition_tool is not None:
                exe_context.state.set("exit_reason", exit_condition_tool)
                return self._continue_after_exit_hooks(exe_context=exe_context)
            return True

    async def _run_step_async(self, exe_context: _ExecutionContext, agent_span: tracing.Span) -> bool:
        """Execute one agent step asynchronously. Returns True to continue the loop, False to stop."""
        with tracing.tracer.trace(
            "haystack.agent.step", tags={"haystack.agent.step": exe_context.counter}, parent_span=agent_span
        ) as step_span:
            # Re-flatten the tools every step so dynamic toolsets (e.g. SearchableToolset) surface tools discovered in
            # earlier steps. Validate names here so duplicates fail before starting the step.
            current_tools = flatten_tools_or_toolsets(tools=exe_context.tools)
            _check_duplicate_tool_names(tools=current_tools)
            # Expose the current tools to hooks (e.g. ConfirmationHook) via State.
            exe_context.state.set("tools", current_tools, handler_override=replace_values)

            await _run_hooks_async(hooks=self.hooks, hook_point=BEFORE_LLM, state=exe_context.state)
            chat_generator_inputs = {
                "messages": exe_context.state.data["messages"],
                **exe_context.chat_generator_inputs,
            }
            if current_tools:
                chat_generator_inputs["tools"] = current_tools
            with tracing.tracer.trace("haystack.agent.step.llm", parent_span=step_span) as llm_span:
                llm_span.set_content_tag("haystack.agent.step.llm.input", chat_generator_inputs)
                # For sync-only generators, _execute_component_async dispatches to a thread via asyncio.to_thread,
                # which copies the current contextvars context — preserving the active tracing span.
                result = await _execute_component_async(self.chat_generator, **chat_generator_inputs)
                llm_span.set_content_tag("haystack.agent.step.llm.output", result)
            llm_messages = result["replies"]
            exe_context.state.set("messages", llm_messages)
            _record_llm_usage(state=exe_context.state, llm_messages=llm_messages)
            _record_context_tokens(state=exe_context.state, llm_messages=llm_messages)

            # Stop on the "no tool call" exit: no tools available, or a plain assistant text reply (see _is_text_exit).
            if not current_tools or _is_text_exit(messages=llm_messages):
                exe_context.counter += 1
                exe_context.state.set("step_count", exe_context.counter)
                exe_context.state.set("exit_reason", _EXIT_REASON_TEXT)
                return await self._continue_after_exit_hooks_async(exe_context=exe_context)

            await _run_hooks_async(hooks=self.hooks, hook_point=BEFORE_TOOL, state=exe_context.state)
            # Re-read the pending tool calls from State so that any rewrites a before_tool hook made (e.g.
            # ConfirmationHook rejecting or modifying calls) are honored by the executor.
            pending_tool_call_messages = _pending_tool_call_messages_from_state(exe_context.state)

            tool_execution_inputs = {
                "messages": pending_tool_call_messages,
                "state": exe_context.state,
                **exe_context.tool_execution_inputs,
                "tools": current_tools,
            }
            tool_messages, exe_context.state = await _run_tool_async(**tool_execution_inputs)
            exe_context.state.set("messages", tool_messages)
            _record_tool_calls(state=exe_context.state, tool_messages=tool_messages)
            await _run_hooks_async(hooks=self.hooks, hook_point=AFTER_TOOL, state=exe_context.state)

            exe_context.counter += 1
            exe_context.state.set("step_count", exe_context.counter)
            exit_condition_tool = (
                None
                if self.exit_conditions == ["text"]
                else self._check_exit_conditions(llm_messages=pending_tool_call_messages, tool_messages=tool_messages)
            )
            if exit_condition_tool is not None:
                exe_context.state.set("exit_reason", exit_condition_tool)
                return await self._continue_after_exit_hooks_async(exe_context=exe_context)
            return True

    def _check_exit_conditions(self, llm_messages: list[ChatMessage], tool_messages: list[ChatMessage]) -> str | None:
        """
        Decide whether the agent should stop looping and, if so, on which tool.

        Returns the name of the tool that triggered an exit condition: the model called at least one tool listed in
        `exit_conditions` and that tool did not error. Every tool call in the message is checked, so the order of
        parallel tool calls does not matter. When several exit-condition tools are called in the same step, the first
        one encountered is returned.

        :param llm_messages: List of messages from the LLM
        :param tool_messages: List of messages from tool execution.
        :return: The name of the tool that satisfied an exit condition, or None if none did (or one errored).
        """
        errored_tools = {
            tool_msg.tool_call_result.origin.tool_name
            for tool_msg in tool_messages
            if tool_msg.tool_call_result is not None and tool_msg.tool_call_result.error
        }

        first_match: str | None = None
        for msg in llm_messages:
            for tool_call in msg.tool_calls:
                if tool_call.tool_name not in self.exit_conditions:
                    continue
                # An errored exit-condition tool cancels the exit, even if another exit-condition tool succeeded.
                if tool_call.tool_name in errored_tools:
                    return None
                first_match = first_match or tool_call.tool_name
        return first_match

    def _continue_after_exit_hooks(self, exe_context: _ExecutionContext) -> bool:
        """
        Run `on_exit` hooks and return whether the loop should keep going.

        A hook keeps the Agent running by setting the `continue_run` control flag (`state.set("continue_run", True)`),
        usually alongside a message telling the model what to do next. The flag is consumed on each exit attempt and
        the loop stays bounded by `max_agent_steps`.
        """
        if not self.hooks.get(ON_EXIT):
            return False
        exe_context.state.set("continue_run", False)
        _run_hooks(hooks=self.hooks, hook_point=ON_EXIT, state=exe_context.state)
        return _consume_continue_run(exe_context.state)

    async def _continue_after_exit_hooks_async(self, exe_context: _ExecutionContext) -> bool:
        """Async version of `_continue_after_exit_hooks`."""
        if not self.hooks.get(ON_EXIT):
            return False
        exe_context.state.set("continue_run", False)
        await _run_hooks_async(hooks=self.hooks, hook_point=ON_EXIT, state=exe_context.state)
        return _consume_continue_run(exe_context.state)
