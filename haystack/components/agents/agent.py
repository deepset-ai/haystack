# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.components.generators.chat.types import ChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.core.serialization import import_class_by_name
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.state import State, _schema_from_dict, _schema_to_dict, _validate_schema
from haystack.dataclasses.streaming_chunk import SyncStreamingCallbackT
from haystack.tools import Tool, deserialize_tools_inplace
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable

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
        exit_condition: str = "text",
        state_schema: Optional[Dict[str, Any]] = None,
        max_runs_per_component: int = 100,
        raise_on_tool_invocation_failure: bool = False,
        streaming_callback: Optional[SyncStreamingCallbackT] = None,
    ):
        """
        Initialize the agent component.

        :param chat_generator: An instance of the chat generator that your agent should use. It must support tools.
        :param tools: List of Tool objects available to the agent
        :param system_prompt: System prompt for the agent.
        :param exit_condition: Either "text" if the agent should return when it generates a message without tool calls
            or the name of a tool that will cause the agent to return once the tool was executed
        :param state_schema: The schema for the runtime state used by the tools.
        :param max_runs_per_component: Maximum number of runs per component. Agent will raise an exception if a
            component exceeds the maximum number of runs per component.
        :param raise_on_tool_invocation_failure: Should the agent raise an exception when a tool invocation fails?
            If set to False, the exception will be turned into a chat message and passed to the LLM.
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
        """
        valid_exits = ["text"] + [tool.name for tool in tools or []]
        if exit_condition not in valid_exits:
            raise ValueError(f"Exit condition must be one of {valid_exits}")

        if state_schema is not None:
            _validate_schema(state_schema)
        self.state_schema = state_schema or {}

        self.chat_generator = chat_generator
        self.tools = tools or []
        self.system_prompt = system_prompt
        self.exit_condition = exit_condition
        self.max_runs_per_component = max_runs_per_component
        self.raise_on_tool_invocation_failure = raise_on_tool_invocation_failure
        self.streaming_callback = streaming_callback

        output_types = {"messages": List[ChatMessage]}
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
            exit_condition=self.exit_condition,
            state_schema=_schema_to_dict(self.state_schema),
            max_runs_per_component=self.max_runs_per_component,
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

        chat_generator_class = import_class_by_name(init_params["chat_generator"]["type"])
        assert hasattr(chat_generator_class, "from_dict")  # we know but mypy doesn't
        chat_generator_instance = chat_generator_class.from_dict(init_params["chat_generator"])
        data["init_parameters"]["chat_generator"] = chat_generator_instance

        if "state_schema" in init_params:
            init_params["state_schema"] = _schema_from_dict(init_params["state_schema"])

        if init_params.get("streaming_callback") is not None:
            init_params["streaming_callback"] = deserialize_callable(init_params["streaming_callback"])

        deserialize_tools_inplace(init_params, key="tools")

        return default_from_dict(cls, data)

    def run(
        self, messages: List[ChatMessage], streaming_callback: Optional[SyncStreamingCallbackT] = None, **kwargs
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

        generator_inputs: Dict[str, Any] = {"tools": self.tools}

        selected_callback = streaming_callback or self.streaming_callback
        if selected_callback is not None:
            generator_inputs["streaming_callback"] = selected_callback

        # Repeat until the exit condition is met
        counter = 0
        while counter < self.max_runs_per_component:
            # 1. Call the ChatGenerator
            llm_messages = self.chat_generator.run(messages=messages, **generator_inputs)["replies"]

            # TODO Possible for LLM to return multiple messages (e.g. multiple tool calls)
            #      Would a better check be to see if any of the messages contain a tool call?
            # 2. Check if the LLM response contains a tool call
            if llm_messages[0].tool_call is None:
                return {"messages": messages + llm_messages, **state.data}

            # 3. Call the ToolInvoker
            # We only send the messages from the LLM to the tool invoker
            tool_invoker_result = self._tool_invoker.run(messages=llm_messages, state=state)
            tool_messages = tool_invoker_result["messages"]
            state = tool_invoker_result["state"]

            # 4. Check the LLM and Tool response for the exit condition, if exit_condition is a tool name
            # TODO Possible for LLM to return multiple messages (e.g. multiple tool calls)
            #      So exit condition could be missed if it's not the first message
            if self.exit_condition != "text" and (
                llm_messages[0].tool_call.tool_name == self.exit_condition
                and not tool_messages[0].tool_call_result.error
            ):
                return {"messages": messages + llm_messages + tool_messages, **state.data}

            # 5. Combine messages, llm_messages and tool_messages and send to the ChatGenerator
            messages = messages + llm_messages + tool_messages
            counter += 1

        logger.warning(
            "Agent exceeded maximum runs per component ({max_loops}), stopping.", max_loops=self.max_runs_per_component
        )
        return {"messages": messages, **state.data}
