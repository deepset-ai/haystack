# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.generators.chat.types import ChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.state import State, _schema_from_dict, _schema_to_dict, _validate_schema
from haystack.dataclasses.streaming_chunk import SyncStreamingCallbackT
from haystack.tools import Tool, deserialize_tools_inplace
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable
from haystack.utils.deserialization import deserialize_chatgenerator_inplace

logger = logging.getLogger(__name__)


@component
class Agent:
    """
    A Haystack component that implements a tool-using agent with provider-agnostic chat model support.

    The component processes messages and executes tools until a exit_condition condition is met.
    The exit_condition can be triggered either by a direct text response or by invoking a specific designated tool.

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
        tools: Optional[List[Tool]] = None,
        system_prompt: Optional[str] = None,
        exit_conditions: Optional[List[str]] = None,
        state_schema: Optional[Dict[str, Any]] = None,
        max_agent_steps: int = 100,
        raise_on_tool_invocation_failure: bool = False,
        streaming_callback: Optional[SyncStreamingCallbackT] = None,
    ):
        """
        Initialize the agent component.

        :param chat_generator: An instance of the chat generator that your agent should use. It must support tools.
        :param tools: List of Tool objects available to the agent
        :param system_prompt: System prompt for the agent.
        :param exit_conditions: List of conditions that will cause the agent to return.
            Can include "text" if the agent should return when it generates a message without tool calls,
            or tool names that will cause the agent to return once the tool was executed. Defaults to ["text"].
        :param state_schema: The schema for the runtime state used by the tools.
        :param max_agent_steps: Maximum number of steps the agent will run before stopping. Defaults to 100.
            If the agent exceeds this number of steps, it will stop and return the current state.
        :param raise_on_tool_invocation_failure: Should the agent raise an exception when a tool invocation fails?
            If set to False, the exception will be turned into a chat message and passed to the LLM.
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
        :raises TypeError: If the chat_generator does not support tools parameter in its run method.
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

        if state_schema is not None:
            _validate_schema(state_schema)
        self.state_schema = state_schema or {}

        self.chat_generator = chat_generator
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.exit_conditions = exit_conditions
        self.max_agent_steps = max_agent_steps
        self.raise_on_tool_invocation_failure = raise_on_tool_invocation_failure
        self.streaming_callback = streaming_callback

        output_types = {}
        for param, config in self.state_schema.items():
            component.set_input_type(self, name=param, type=config["type"], default=None)
            output_types[param] = config["type"]
        component.set_output_types(self, **output_types)

        self._tool_invoker = ToolInvoker(tools=self.tools, raise_on_failure=self.raise_on_tool_invocation_failure)
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
            chat_generator=self.chat_generator.to_dict(),
            tools=[t.to_dict() for t in self.tools],
            system_prompt=self.system_prompt,
            exit_conditions=self.exit_conditions,
            state_schema=_schema_to_dict(self.state_schema),
            max_agent_steps=self.max_agent_steps,
            raise_on_tool_invocation_failure=self.raise_on_tool_invocation_failure,
            streaming_callback=streaming_callback,
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

        deserialize_tools_inplace(init_params, key="tools")

        return default_from_dict(cls, data)

    def run(
        self,
        messages: List[ChatMessage],
        streaming_callback: Optional[SyncStreamingCallbackT] = None,
        **kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process messages and execute tools until the exit condition is met.

        :param messages: List of chat messages to process
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
        :param kwargs: Additional data to pass to the State schema used by the Agent.
            The keys must match the schema defined in the Agent's `state_schema`.
        :return: Dictionary containing messages and outputs matching the defined output types
        """
        if not self._is_warmed_up and hasattr(self.chat_generator, "warm_up"):
            raise RuntimeError("The component Agent wasn't warmed up. Run 'warm_up()' before calling 'run()'.")

        state = State(schema=self.state_schema, data=kwargs)

        if self.system_prompt is not None:
            messages = [ChatMessage.from_system(self.system_prompt)] + messages
        state.set("messages", messages)

        generator_inputs: Dict[str, Any] = {"tools": self.tools}

        selected_callback = streaming_callback or self.streaming_callback
        if selected_callback is not None:
            generator_inputs["streaming_callback"] = selected_callback

        # Repeat until the exit condition is met
        counter = 0
        while counter < self.max_agent_steps:
            # 1. Call the ChatGenerator
            llm_messages = self.chat_generator.run(messages=messages, **generator_inputs)["replies"]
            state.set("messages", llm_messages)

            # 2. Check if any of the LLM responses contain a tool call
            if not any(msg.tool_call for msg in llm_messages):
                return {**state.data}

            # 3. Call the ToolInvoker
            # We only send the messages from the LLM to the tool invoker
            tool_invoker_result = self._tool_invoker.run(messages=llm_messages, state=state)
            tool_messages = tool_invoker_result["tool_messages"]
            state = tool_invoker_result["state"]
            state.set("messages", tool_messages)

            # 4. Check if any LLM message's tool call name matches an exit condition
            if self.exit_conditions != ["text"]:
                matched_exit_conditions = set()
                has_errors = False

                for msg in llm_messages:
                    if msg.tool_call and msg.tool_call.tool_name in self.exit_conditions:
                        matched_exit_conditions.add(msg.tool_call.tool_name)

                        # Check if any error is specifically from the tool matching the exit condition
                        tool_errors = [
                            tool_msg.tool_call_result.error
                            for tool_msg in tool_messages
                            if tool_msg.tool_call_result.origin.tool_name == msg.tool_call.tool_name
                        ]
                        if any(tool_errors):
                            has_errors = True
                            # No need to check further if we found an error
                            break

                # Only return if at least one exit condition was matched AND none had errors
                if matched_exit_conditions and not has_errors:
                    return {**state.data}

            # 5. Fetch the combined messages and send them back to the LLM
            messages = state.get("messages")
            counter += 1

        logger.warning(
            "Agent exceeded maximum agent steps of {max_agent_steps}, stopping.", max_agent_steps=self.max_agent_steps
        )
        return {**state.data}
