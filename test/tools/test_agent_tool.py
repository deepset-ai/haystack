# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest

from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator, OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole, ToolCall
from haystack.tools import AgentTool, tool
from haystack.tools.agent_tool import _required_tool_parameters, agent_result_to_string


def _echo_last_message(messages: list[ChatMessage]) -> str:
    return messages[-1].text or ""


MESSAGES_SCHEMA = {
    "type": "array",
    "description": "Exactly one user message.",
    "minItems": 1,
    "maxItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "role": {"type": "string", "enum": ["user"]},
            "content": {"type": "string", "description": "The task to delegate to this tool."},
        },
        "required": ["role", "content"],
    },
}


class TestRequiredToolParameters:
    def test_agent_without_prompt_variables(self):
        agent = Agent(chat_generator=MockChatGenerator(responses=["done"]), system_prompt="You research things.")
        assert _required_tool_parameters(agent=agent, inputs_from_state=None) == []

    def test_agent_with_prompt_variables(self):
        agent = Agent(chat_generator=MockChatGenerator(responses=["done"]), system_prompt="You research {{topic}}.")
        assert _required_tool_parameters(agent=agent, inputs_from_state=None) == ["topic"]

    def test_prompt_variable_mapped_from_state(self):
        agent = Agent(chat_generator=MockChatGenerator(responses=["done"]), system_prompt="You research {{topic}}.")
        assert _required_tool_parameters(agent=agent, inputs_from_state={"subject": "topic"}) == []


class TestAgentResultToString:
    def test_exit_reason_text(self):
        result = {"last_message": ChatMessage.from_assistant("Response"), "exit_reason": "text"}
        assert agent_result_to_string(result=result) == "Response"

    def test_exit_reason_tool(self):
        message = ChatMessage.from_tool(tool_result="42", origin=ToolCall(id="1", tool_name="lookup", arguments={}))
        result = {"last_message": message, "exit_reason": "lookup"}
        assert agent_result_to_string(result=result) == json.dumps(message.to_dict())

    def test_exit_reason_max_agent_steps(self):
        message = ChatMessage.from_tool(tool_result="42", origin=ToolCall(id="1", tool_name="lookup", arguments={}))
        result = {"last_message": message, "exit_reason": "max_agent_steps"}
        assert agent_result_to_string(result=result) == json.dumps(message.to_dict()) + (
            "\n\n[The Agent reached max_agent_steps and stopped, so this result may be incomplete.]"
        )


class TestAgentTool:
    def test_init(self):
        sub_agent = Agent(chat_generator=MockChatGenerator(response_fn=_echo_last_message))

        agent_tool = AgentTool(
            agent=sub_agent, name="research", description="Research a question and report the findings."
        )

        assert agent_tool.name == "research"
        assert agent_tool.description == "Research a question and report the findings."
        assert agent_tool._component is sub_agent
        assert agent_tool.parameters == {
            "type": "object",
            "properties": {"messages": MESSAGES_SCHEMA},
            "required": ["messages"],
        }
        assert agent_tool.outputs_to_string == {"handler": agent_result_to_string}

    def test_init_with_non_agent(self):
        with pytest.raises(TypeError, match="must be an instance of Agent"):
            AgentTool(agent="not an agent", name="research", description="Research a question.")  # type: ignore[arg-type]

    def test_init_with_prompt_template(self):
        sub_agent = Agent(chat_generator=MockChatGenerator(responses=["done"]), system_prompt="You research {{topic}}.")

        agent_tool = AgentTool(agent=sub_agent, name="research", description="Research a question.")

        assert agent_tool.parameters == {
            "type": "object",
            "properties": {"messages": MESSAGES_SCHEMA, "topic": {"type": "string"}},
            "required": ["messages", "topic"],
        }

    def test_init_with_prompt_template_and_inputs_from_state(self):
        sub_agent = Agent(chat_generator=MockChatGenerator(responses=["done"]), system_prompt="You research {{topic}}.")

        agent_tool = AgentTool(
            agent=sub_agent, name="research", description="Research a question.", inputs_from_state={"subject": "topic"}
        )

        assert agent_tool.parameters == {
            "type": "object",
            "properties": {"messages": MESSAGES_SCHEMA},
            "required": ["messages"],
        }

    def test_init_with_custom_parameters(self):
        sub_agent = Agent(chat_generator=MockChatGenerator(responses=["done"]))
        parameters = {"type": "object", "properties": {"messages": {"type": "array"}}, "required": ["messages"]}

        agent_tool = AgentTool(
            agent=sub_agent, name="research", description="Research a question.", parameters=parameters
        )

        assert agent_tool.parameters == parameters

    def test_init_with_custom_parameters_missing_a_required_input(self):
        sub_agent = Agent(chat_generator=MockChatGenerator(responses=["done"]), system_prompt="You research {{topic}}.")
        parameters = {"type": "object", "properties": {"messages": {"type": "array"}}, "required": ["messages"]}

        with pytest.raises(ValueError, match=r"requires the inputs \['topic'\]"):
            AgentTool(agent=sub_agent, name="research", description="Research a question.", parameters=parameters)

    def test_to_dict(self):
        sub_agent = Agent(chat_generator=MockChatGenerator(response_fn=_echo_last_message))
        agent_tool = AgentTool(
            agent=sub_agent,
            name="research",
            description="Research a question.",
            outputs_to_state={"notes": {"source": "last_message"}},
        )

        assert agent_tool.to_dict() == {
            "type": "haystack.tools.agent_tool.AgentTool",
            "data": {
                "name": "research",
                "description": "Research a question.",
                "parameters": None,
                "inputs_from_state": None,
                "outputs_to_state": {"notes": {"source": "last_message"}},
                "outputs_to_string": {"handler": "haystack.tools.agent_tool.agent_result_to_string"},
                "agent": {
                    "type": "haystack.components.agents.agent.Agent",
                    "init_parameters": {
                        "chat_generator": {
                            "type": "haystack.components.generators.chat.mock.MockChatGenerator",
                            "init_parameters": {
                                "responses": None,
                                "response_fn": "test_agent_tool._echo_last_message",
                                "model": "mock-model",
                                "meta": {},
                                "streaming_callback": None,
                            },
                        },
                        "tools": [],
                        "system_prompt": None,
                        "user_prompt": None,
                        "required_variables": "*",
                        "exit_conditions": ["text"],
                        "state_schema": {},
                        "max_agent_steps": 100,
                        "streaming_callback": None,
                        "raise_on_tool_invocation_failure": False,
                        "tool_concurrency_limit": 4,
                        "tool_streaming_callback_passthrough": False,
                        "hooks": None,
                    },
                },
            },
        }

    def test_to_dict_with_explicit_parameters(self):
        sub_agent = Agent(chat_generator=MockChatGenerator(response_fn=_echo_last_message))
        parameters = {"type": "object", "properties": {"messages": {"type": "array"}}, "required": ["messages"]}
        agent_tool = AgentTool(
            agent=sub_agent, name="research", description="Research a question.", parameters=parameters
        )

        assert agent_tool.to_dict()["data"]["parameters"] == parameters

    def test_from_dict(self):
        data = {
            "type": "haystack.tools.agent_tool.AgentTool",
            "data": {
                "name": "research",
                "description": "Research a question.",
                "parameters": None,
                "inputs_from_state": None,
                "outputs_to_state": {"notes": {"source": "last_message"}},
                "outputs_to_string": {"handler": "haystack.tools.agent_tool.agent_result_to_string"},
                "agent": {
                    "type": "haystack.components.agents.agent.Agent",
                    "init_parameters": {
                        "chat_generator": {
                            "type": "haystack.components.generators.chat.mock.MockChatGenerator",
                            "init_parameters": {"response_fn": "test_agent_tool._echo_last_message"},
                        }
                    },
                },
            },
        }

        agent_tool = AgentTool.from_dict(data)

        assert isinstance(agent_tool, AgentTool)
        assert isinstance(agent_tool._component, Agent)
        assert agent_tool.name == "research"
        assert agent_tool.description == "Research a question."
        assert agent_tool.parameters == {
            "type": "object",
            "properties": {"messages": MESSAGES_SCHEMA},
            "required": ["messages"],
        }
        assert agent_tool.outputs_to_string == {"handler": agent_result_to_string}
        assert agent_tool.outputs_to_state == {"notes": {"source": "last_message"}}
        assert agent_tool.to_dict()["data"]["parameters"] is None

    def test_from_dict_with_explicit_parameters(self):
        parameters = {
            "type": "object",
            "properties": {"messages": {"type": "array", "description": "Custom messages schema."}},
            "required": ["messages"],
        }
        data = AgentTool(
            agent=Agent(chat_generator=MockChatGenerator(response_fn=_echo_last_message)),
            name="research",
            description="Research a question.",
            parameters=parameters,
        ).to_dict()

        agent_tool = AgentTool.from_dict(data)

        assert agent_tool.parameters == parameters
        assert agent_tool.to_dict()["data"]["parameters"] == parameters

    @pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
    @pytest.mark.integration
    def test_live_run(self):
        @tool
        def order_status(order_id: str) -> str:
            """Look up the status of an order."""
            return f"Order {order_id} shipped, tracking code ZX9-4471."

        order_lookup_agent = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4.1-nano"),
            tools=[order_status],
            system_prompt="Look up the order and report the tracking code exactly as the tool returned it.",
            max_agent_steps=3,
        )
        order_lookup_agent_as_tool = AgentTool(
            agent=order_lookup_agent, name="order_lookup", description="Look up the status of an order by its ID."
        )
        coordinator = Agent(
            chat_generator=OpenAIChatGenerator(model="gpt-4.1-mini"),
            tools=[order_lookup_agent_as_tool],
            system_prompt="Delegate every order question to the order_lookup tool, then answer the user.",
            max_agent_steps=5,
        )

        result = coordinator.run([ChatMessage.from_user("What is the status and the tracking code of order A-17?")])
        print(result)

        tool_messages = [message for message in result["messages"] if message.is_from(ChatRole.TOOL)]
        assert tool_messages
        assert all(not message.tool_call_results[0].error for message in tool_messages)
        assert any("ZX9-4471" in message.tool_call_results[0].result for message in tool_messages)
        assert result["tool_call_counts"].keys() == {"order_lookup"}
        assert "ZX9-4471" in result["messages"][-1].text
