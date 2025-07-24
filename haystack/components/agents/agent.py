# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Any, Dict, List, Optional, Union

from haystack import logging, tracing
from haystack.components.generators.chat.types import ChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.core.component.component import component
from haystack.core.pipeline.async_pipeline import AsyncPipeline
from haystack.core.pipeline.breakpoint import (
    _check_chat_generator_breakpoint,
    _check_tool_invoker_breakpoint,
    _create_agent_snapshot,
    _validate_tool_breakpoint_is_valid,
)
from haystack.core.pipeline.pipeline import Pipeline
from haystack.core.pipeline.utils import _deepcopy_with_exceptions
from haystack.core.serialization import component_to_dict, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.dataclasses.breakpoints import AgentBreakpoint, AgentSnapshot, ToolBreakpoint
from haystack.dataclasses.streaming_chunk import StreamingCallbackT, select_streaming_callback
from haystack.tools import Tool, Toolset, deserialize_tools_or_toolset_inplace, serialize_tools_or_toolset
from haystack.utils import _deserialize_value_with_schema
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable
from haystack.utils.deserialization import deserialize_chatgenerator_inplace

from .state.state import State, _schema_from_dict, _schema_to_dict, _validate_schema
from .state.state_utils import merge_lists

logger = logging.getLogger(__name__)


@component
class Agent:
    """
    A Haystack component that implements a tool-using agent with provider-agnostic chat model support.

    The component processes messages and executes tools until an exit_condition condition is met.
    The exit_condition can be triggered either by a direct text response or by invoking a specific designated tool.

    When you call an Agent without tools, it acts as a ChatGenerator, produces one response, then exits.

    ### Usage example
    ```python
    from haystack.components.agents import Agent
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.tools.tool import Tool

    tools = [Tool(name="calculator", description="..."), Tool(name="search", description="...")]

    agent = Agent(
        chat_generator=OpenAIChatGenerator(),
        tools=tools,
        exit_condition="search",
    )

    # Run the agent
    result = agent.run(
        messages=[ChatMessage.from_user("Find information about Haystack")]
    )

    assert "messages" in result  # Contains conversation history
    ```
    """

    def __init__(
        self,
        *,
        chat_generator: ChatGenerator,
        tools: Optional[Union[List[Tool], Toolset]] = None,
        system_prompt: Optional[str] = None,
        exit_conditions: Optional[List[str]] = None,
        state_schema: Optional[Dict[str, Any]] = None,
        max_agent_steps: int = 100,
        streaming_callback: Optional[StreamingCallbackT] = None,
        raise_on_tool_invocation_failure: bool = False,
        tool_invoker_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the agent component.

        :param chat_generator: An instance of the chat generator that your agent should use. It must support tools.
        :param tools: List of Tool objects or a Toolset that the agent can use.
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

        valid_exits = ["text"] + [tool.name for tool in tools or []]
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
            resolved_state_schema["messages"] = {"type": List[ChatMessage], "handler": merge_lists}
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
            self._is_warmed_up = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :return: Dictionary with serialized data
        """
        if self.streaming_callback is not None:
            streaming_callback = serialize_callable(self.streaming_callback)
        else:
            streaming_callback = None

        return default_to_dict(
            self,
            chat_generator=component_to_dict(obj=self.chat_generator, name="chat_generator"),
            tools=serialize_tools_or_toolset(self.tools),
            system_prompt=self.system_prompt,
            exit_conditions=self.exit_conditions,
            # We serialize the original state schema, not the resolved one to reflect the original user input
            state_schema=_schema_to_dict(self._state_schema),
            max_agent_steps=self.max_agent_steps,
            streaming_callback=streaming_callback,
            raise_on_tool_invocation_failure=self.raise_on_tool_invocation_failure,
            tool_invoker_kwargs=self.tool_invoker_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        """
        Deserialize the agent from a dictionary.

        :param data: Dictionary to deserialize from
        :return: Deserialized agent
        """
        init_params = data.get("init_parameters", {})

        deserialize_chatgenerator_inplace(init_params, key="chat_generator")

        if "state_schema" in init_params:
            init_params["state_schema"] = _schema_from_dict(init_params["state_schema"])

        if init_params.get("streaming_callback") is not None:
            init_params["streaming_callback"] = deserialize_callable(init_params["streaming_callback"])

        deserialize_tools_or_toolset_inplace(init_params, key="tools")

        return default_from_dict(cls, data)

    def _prepare_generator_inputs(self, streaming_callback: Optional[StreamingCallbackT] = None) -> Dict[str, Any]:
        """Prepare inputs for the chat generator."""
        generator_inputs: Dict[str, Any] = {"tools": self.tools}
        if streaming_callback is not None:
            generator_inputs["streaming_callback"] = streaming_callback
        return generator_inputs

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

    def run(  # noqa: PLR0915
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        *,
        break_point: Optional[AgentBreakpoint] = None,
        snapshot: Optional[AgentSnapshot] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Process messages and execute tools until an exit condition is met.

        :param messages: List of Haystack ChatMessage objects to process.
            If a list of dictionaries is provided, each dictionary will be converted to a ChatMessage object.
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
            The same callback can be configured to emit tool results when a tool is called.
        :param break_point: An AgentBreakpoint, can be a Breakpoint for the "chat_generator" or a ToolBreakpoint
                           for "tool_invoker".
        :param snapshot: A dictionary containing a snapshot of a previously saved agent execution. The snapshot contains
                         the relevant information to restart the Agent execution from where it left off.
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
        # kwargs can contain the key parent_snapshot.
        # We pop it here to avoid passing it into State. We explicitly pass it on if a break point is
        # triggered.
        parent_snapshot = kwargs.pop("parent_snapshot", None)

        if not self._is_warmed_up and hasattr(self.chat_generator, "warm_up"):
            raise RuntimeError("The component Agent wasn't warmed up. Run 'warm_up()' before calling 'run()'.")

        if break_point and snapshot:
            raise ValueError(
                "break_point and snapshot cannot be provided at the same time. The agent run will be aborted."
            )

        # validate breakpoints
        if break_point and isinstance(break_point.break_point, ToolBreakpoint):
            _validate_tool_breakpoint_is_valid(agent_breakpoint=break_point, tools=self.tools)

        # Handle agent snapshot if provided
        if snapshot:
            component_visits = snapshot.component_visits
            current_inputs = {
                "chat_generator": _deserialize_value_with_schema(snapshot.component_inputs["chat_generator"]),
                "tool_invoker": _deserialize_value_with_schema(snapshot.component_inputs["tool_invoker"]),
            }
            state_data = current_inputs["tool_invoker"]["state"].data
            if isinstance(snapshot.break_point.break_point, ToolBreakpoint):
                # If the break point is a ToolBreakpoint, we need to get the messages from the tool invoker inputs
                messages = current_inputs["tool_invoker"]["messages"]
                # Needed to know if we should start with the ToolInvoker or ChatGenerator
                skip_chat_generator = True
            else:
                messages = current_inputs["chat_generator"]["messages"]
                skip_chat_generator = False
            # We also load the streaming_callback from the snapshot if it exists
            streaming_callback = current_inputs["chat_generator"].get("streaming_callback", None)
        else:
            skip_chat_generator = False
            if self.system_prompt is not None:
                messages = [ChatMessage.from_system(self.system_prompt)] + messages

            if all(m.is_from(ChatRole.SYSTEM) for m in messages):
                logger.warning(
                    "All messages provided to the Agent component are system messages. This is not recommended as the "
                    "Agent will not perform any actions specific to user input. Consider adding user messages to the "
                    "input."
                )
            component_visits = dict.fromkeys(["chat_generator", "tool_invoker"], 0)
            state_data = kwargs

        state = State(schema=self.state_schema, data=state_data)
        state.set("messages", messages)

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=False
        )
        generator_inputs = self._prepare_generator_inputs(streaming_callback=streaming_callback)
        with self._create_agent_span() as span:
            span.set_content_tag(
                "haystack.agent.input",
                _deepcopy_with_exceptions({"messages": messages, "streaming_callback": streaming_callback, **kwargs}),
            )
            counter = 0

            while counter < self.max_agent_steps:
                # check for breakpoint before ChatGenerator
                if (
                    break_point
                    and break_point.break_point.component_name == "chat_generator"
                    and component_visits["chat_generator"] == break_point.break_point.visit_count
                ):
                    agent_snapshot = _create_agent_snapshot(
                        component_visits=component_visits,
                        agent_breakpoint=break_point,
                        component_inputs={
                            "chat_generator": {"messages": messages, **generator_inputs},
                            "tool_invoker": {"messages": [], "state": state, "streaming_callback": streaming_callback},
                        },
                    )
                    _check_chat_generator_breakpoint(agent_snapshot=agent_snapshot, parent_snapshot=parent_snapshot)

                # 1. Call the ChatGenerator
                # We skip the chat generator when restarting from a snapshot where we restart at the ToolInvoker.
                if skip_chat_generator:
                    llm_messages = state.get("messages", [])[-1:]
                    # We set it to False to ensure that the next iteration will call the chat generator again
                    skip_chat_generator = False
                else:
                    result = Pipeline._run_component(
                        component_name="chat_generator",
                        component={"instance": self.chat_generator},
                        inputs={"messages": messages, **generator_inputs},
                        component_visits=component_visits,
                        parent_span=span,
                    )
                    llm_messages = result["replies"]
                    state.set("messages", llm_messages)

                # 2. Check if any of the LLM responses contain a tool call or if the LLM is not using tools
                if not any(msg.tool_call for msg in llm_messages) or self._tool_invoker is None:
                    counter += 1
                    break

                # check for breakpoint before ToolInvoker
                if (
                    break_point
                    and break_point.break_point.component_name == "tool_invoker"
                    and break_point.break_point.visit_count == component_visits["tool_invoker"]
                ):
                    agent_snapshot = _create_agent_snapshot(
                        component_visits=component_visits,
                        agent_breakpoint=break_point,
                        component_inputs={
                            "chat_generator": {"messages": messages[:-1], **generator_inputs},
                            "tool_invoker": {
                                "messages": llm_messages,
                                "state": state,
                                "streaming_callback": streaming_callback,
                            },
                        },
                    )
                    _check_tool_invoker_breakpoint(
                        llm_messages=llm_messages, agent_snapshot=agent_snapshot, parent_snapshot=parent_snapshot
                    )

                # 3. Call the ToolInvoker
                # We only send the messages from the LLM to the tool invoker
                tool_invoker_result = Pipeline._run_component(
                    component_name="tool_invoker",
                    component={"instance": self._tool_invoker},
                    inputs={"messages": llm_messages, "state": state, "streaming_callback": streaming_callback},
                    component_visits=component_visits,
                    parent_span=span,
                )
                tool_messages = tool_invoker_result["tool_messages"]
                state = tool_invoker_result["state"]
                state.set("messages", tool_messages)

                # 4. Check if any LLM message's tool call name matches an exit condition
                if self.exit_conditions != ["text"] and self._check_exit_conditions(llm_messages, tool_messages):
                    counter += 1
                    break

                # 5. Fetch the combined messages and send them back to the LLM
                messages = state.get("messages")
                counter += 1

            if counter >= self.max_agent_steps:
                logger.warning(
                    "Agent reached maximum agent steps of {max_agent_steps}, stopping.",
                    max_agent_steps=self.max_agent_steps,
                )
            span.set_content_tag("haystack.agent.output", state.data)
            span.set_tag("haystack.agent.steps_taken", counter)

        result = {**state.data}
        all_messages = state.get("messages")
        if all_messages:
            result.update({"last_message": all_messages[-1]})
        return result

    async def run_async(  # noqa: PLR0915
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[StreamingCallbackT] = None,
        *,
        break_point: Optional[AgentBreakpoint] = None,
        snapshot: Optional[AgentSnapshot] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Asynchronously process messages and execute tools until the exit condition is met.

        This is the asynchronous version of the `run` method. It follows the same logic but uses
        asynchronous operations where possible, such as calling the `run_async` method of the ChatGenerator
        if available.

        :param messages: List of chat messages to process
        :param streaming_callback: An asynchronous callback that will be invoked when a response
        is streamed from the LLM. The same callback can be configured to emit tool results when a tool is called.
        :param break_point: An AgentBreakpoint, can be a Breakpoint for the "chat_generator" or a ToolBreakpoint
                           for "tool_invoker".
        :param snapshot: A dictionary containing the state of a previously saved agent execution.
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
        # kwargs can contain the key parent_snapshot.
        # We pop it here to avoid passing it into State. We explicitly handle it pass it on if a break point is
        # triggered.
        parent_snapshot = kwargs.pop("parent_snapshot", None)

        if not self._is_warmed_up and hasattr(self.chat_generator, "warm_up"):
            raise RuntimeError("The component Agent wasn't warmed up. Run 'warm_up()' before calling 'run_async()'.")

        if break_point and snapshot:
            raise ValueError(
                "break_point and snapshot cannot be provided at the same time. The agent run will be aborted."
            )

        # validate breakpoints
        if break_point and isinstance(break_point.break_point, ToolBreakpoint):
            _validate_tool_breakpoint_is_valid(agent_breakpoint=break_point, tools=self.tools)

        # Handle agent snapshot if provided
        if snapshot:
            component_visits = snapshot.component_visits
            current_inputs = {
                "chat_generator": _deserialize_value_with_schema(snapshot.component_inputs["chat_generator"]),
                "tool_invoker": _deserialize_value_with_schema(snapshot.component_inputs["tool_invoker"]),
            }
            state_data = current_inputs["tool_invoker"]["state"].data
            if isinstance(snapshot.break_point.break_point, ToolBreakpoint):
                # If the break point is a ToolBreakpoint, we need to get the messages from the tool invoker inputs
                messages = current_inputs["tool_invoker"]["messages"]
                # Needed to know if we should start with the ToolInvoker or ChatGenerator
                skip_chat_generator = True
            else:
                messages = current_inputs["chat_generator"]["messages"]
                skip_chat_generator = False
            # We also load the streaming_callback from the snapshot if it exists
            streaming_callback = current_inputs["chat_generator"].get("streaming_callback", None)
        else:
            skip_chat_generator = False
            if self.system_prompt is not None:
                messages = [ChatMessage.from_system(self.system_prompt)] + messages

            if all(m.is_from(ChatRole.SYSTEM) for m in messages):
                logger.warning(
                    "All messages provided to the Agent component are system messages. This is not recommended as the "
                    "Agent will not perform any actions specific to user input. Consider adding user messages to the "
                    "input."
                )
            component_visits = dict.fromkeys(["chat_generator", "tool_invoker"], 0)
            state_data = kwargs

        state = State(schema=self.state_schema, data=state_data)
        state.set("messages", messages)

        streaming_callback = select_streaming_callback(
            init_callback=self.streaming_callback, runtime_callback=streaming_callback, requires_async=True
        )
        generator_inputs = self._prepare_generator_inputs(streaming_callback=streaming_callback)
        with self._create_agent_span() as span:
            span.set_content_tag(
                "haystack.agent.input",
                _deepcopy_with_exceptions({"messages": messages, "streaming_callback": streaming_callback, **kwargs}),
            )
            counter = 0

            while counter < self.max_agent_steps:
                # check for breakpoint before ChatGenerator
                if (
                    break_point
                    and break_point.break_point.component_name == "chat_generator"
                    and component_visits["chat_generator"] == break_point.break_point.visit_count
                ):
                    agent_snapshot = _create_agent_snapshot(
                        component_visits=component_visits,
                        agent_breakpoint=break_point,
                        component_inputs={
                            "chat_generator": {"messages": messages, **generator_inputs},
                            "tool_invoker": {"messages": [], "state": state, "streaming_callback": streaming_callback},
                        },
                    )
                    _check_chat_generator_breakpoint(agent_snapshot=agent_snapshot, parent_snapshot=parent_snapshot)

                # 1. Call the ChatGenerator
                # If skip_chat_generator is True, we skip the chat generator and use the messages from the state
                # This is useful when the agent is resumed from a snapshot where the chat generator already ran.
                if skip_chat_generator:
                    llm_messages = state.get("messages", [])[-1:]  # Get the last message from the state
                    # We set it to False to ensure that the next iteration will call the chat generator again
                    skip_chat_generator = False
                else:
                    result = await AsyncPipeline._run_component_async(
                        component_name="chat_generator",
                        component={"instance": self.chat_generator},
                        component_inputs={"messages": messages, **generator_inputs},
                        component_visits=component_visits,
                        parent_span=span,
                    )
                    llm_messages = result["replies"]
                    state.set("messages", llm_messages)

                # 2. Check if any of the LLM responses contain a tool call or if the LLM is not using tools
                if not any(msg.tool_call for msg in llm_messages) or self._tool_invoker is None:
                    counter += 1
                    break

                # Check for breakpoint before ToolInvoker
                if (
                    break_point
                    and break_point.break_point.component_name == "tool_invoker"
                    and break_point.break_point.visit_count == component_visits["tool_invoker"]
                ):
                    agent_snapshot = _create_agent_snapshot(
                        component_visits=component_visits,
                        agent_breakpoint=break_point,
                        component_inputs={
                            "chat_generator": {"messages": messages[:-1], **generator_inputs},
                            "tool_invoker": {
                                "messages": llm_messages,
                                "state": state,
                                "streaming_callback": streaming_callback,
                            },
                        },
                    )
                    _check_tool_invoker_breakpoint(
                        llm_messages=llm_messages, agent_snapshot=agent_snapshot, parent_snapshot=parent_snapshot
                    )

                # 3. Call the ToolInvoker
                # We only send the messages from the LLM to the tool invoker
                tool_invoker_result = await AsyncPipeline._run_component_async(
                    component_name="tool_invoker",
                    component={"instance": self._tool_invoker},
                    component_inputs={
                        "messages": llm_messages,
                        "state": state,
                        "streaming_callback": streaming_callback,
                    },
                    component_visits=component_visits,
                    parent_span=span,
                )
                tool_messages = tool_invoker_result["tool_messages"]
                state = tool_invoker_result["state"]
                state.set("messages", tool_messages)

                # 4. Check if any LLM message's tool call name matches an exit condition
                if self.exit_conditions != ["text"] and self._check_exit_conditions(llm_messages, tool_messages):
                    counter += 1
                    break

                # 5. Fetch the combined messages and send them back to the LLM
                messages = state.get("messages")
                counter += 1

            if counter >= self.max_agent_steps:
                logger.warning(
                    "Agent reached maximum agent steps of {max_agent_steps}, stopping.",
                    max_agent_steps=self.max_agent_steps,
                )
            span.set_content_tag("haystack.agent.output", state.data)
            span.set_tag("haystack.agent.steps_taken", counter)

        result = {**state.data}
        all_messages = state.get("messages")
        if all_messages:
            result.update({"last_message": all_messages[-1]})
        return result

    def _check_exit_conditions(self, llm_messages: List[ChatMessage], tool_messages: List[ChatMessage]) -> bool:
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
