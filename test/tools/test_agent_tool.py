# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import AgentTool, ComponentTool


def _echo_task(messages: list[ChatMessage]) -> str:
    return f"findings about {messages[-1].text}"


def _echo_system_prompt(messages: list[ChatMessage]) -> str:
    return messages[0].text or ""


def _delegation(task: str) -> dict:
    """The arguments an LLM produces for the default schema."""
    return {"messages": [{"role": "user", "content": task}]}


@pytest.fixture
def researcher():
    return Agent(chat_generator=MockChatGenerator(response_fn=_echo_task), system_prompt="You research things.")


@pytest.fixture
def research_tool(researcher):
    return AgentTool(researcher, name="research", description="Delegate a focused research question.")


def _coordinator(tool, **kwargs):
    """An Agent that delegates once to `tool`, then replies with plain text."""
    return Agent(
        chat_generator=MockChatGenerator(
            responses=[
                ChatMessage.from_assistant(
                    tool_calls=[ToolCall(tool_name=tool.name, arguments=_delegation("the history of Tesla"))]
                ),
                "Done.",
            ]
        ),
        tools=[tool],
        **kwargs,
    )


class TestAgentTool:
    def test_init(self, research_tool, researcher):
        assert research_tool.name == "research"
        assert research_tool.description == "Delegate a focused research question."
        assert research_tool._component is researcher

    def test_default_parameters_ask_for_one_user_message(self, research_tool):
        messages = research_tool.parameters["properties"]["messages"]
        assert research_tool.parameters["required"] == ["messages"]
        assert (messages["minItems"], messages["maxItems"]) == (1, 1)
        assert messages["items"]["properties"]["role"]["enum"] == ["user"]
        assert messages["items"]["required"] == ["role", "content"]

    def test_parameters_none_falls_back_to_the_derived_schema(self, researcher):
        agent_tool = AgentTool(
            researcher, name="research", description="Delegate a research question.", parameters=None
        )
        assert "$defs" in agent_tool.parameters
        assert "generation_kwargs" in agent_tool.parameters["properties"]

    def test_default_parameters_are_much_smaller_than_the_derived_ones(self, researcher):
        agent_tool = AgentTool(researcher, name="research", description="Delegate a research question.")
        component_tool = ComponentTool(component=researcher, name="research", description="Delegate a question.")
        assert "$defs" not in agent_tool.parameters
        assert len(str(agent_tool.parameters)) < len(str(component_tool.parameters)) / 10

    def test_custom_parameters(self, researcher):
        parameters = {"type": "object", "properties": {"messages": {"type": "array"}}, "required": ["messages"]}
        agent_tool = AgentTool(
            researcher, name="research", description="Delegate a research question.", parameters=parameters
        )
        assert agent_tool.parameters == parameters

    def test_init_with_non_agent(self):
        with pytest.raises(TypeError, match="must be an instance of Agent"):
            AgentTool("not an agent", name="research", description="Delegate a research question.")  # type: ignore[arg-type]

    def test_invoke(self, research_tool):
        result = research_tool.invoke(**_delegation("the history of Tesla"))
        assert result["last_message"].text == "findings about the history of Tesla"

    @pytest.mark.asyncio
    async def test_invoke_async(self, research_tool):
        result = await research_tool.invoke_async(**_delegation("the history of Tesla"))
        assert result["last_message"].text == "findings about the history of Tesla"

    def test_tool_result_is_the_final_reply_text(self, research_tool):
        result = _coordinator(research_tool).run([ChatMessage.from_user("Tell me about Tesla")])
        assert result["messages"][2].tool_call_results[0].result == "findings about the history of Tesla"

    def test_custom_outputs_to_string(self, researcher):
        agent_tool = AgentTool(
            researcher,
            name="research",
            description="Delegate a research question.",
            outputs_to_string={"source": "step_count"},
        )
        result = _coordinator(agent_tool).run([ChatMessage.from_user("Tell me about Tesla")])
        assert result["messages"][2].tool_call_results[0].result == "1"

    def test_inputs_from_state_reach_the_agent(self):
        researcher = Agent(
            chat_generator=MockChatGenerator(response_fn=_echo_system_prompt), system_prompt="Answer in {{ language }}."
        )
        agent_tool = AgentTool(
            researcher,
            name="research",
            description="Delegate a research question.",
            inputs_from_state={"language": "language"},
        )
        coordinator = _coordinator(agent_tool, state_schema={"language": {"type": str}})
        result = coordinator.run([ChatMessage.from_user("Tell me about Tesla")], language="Italian")
        assert result["messages"][2].tool_call_results[0].result == "Answer in Italian."

    def test_outputs_to_state(self, researcher):
        agent_tool = AgentTool(
            researcher,
            name="research",
            description="Delegate a research question.",
            outputs_to_state={"notes": {"source": "last_message"}},
        )
        coordinator = _coordinator(agent_tool, state_schema={"notes": {"type": ChatMessage}})
        result = coordinator.run([ChatMessage.from_user("Tell me about Tesla")])
        assert result["notes"].text == "findings about the history of Tesla"

    def test_to_dict(self, research_tool):
        data = research_tool.to_dict()
        assert data["type"] == "haystack.tools.agent_tool.AgentTool"
        assert data["data"]["name"] == "research"
        assert data["data"]["description"] == "Delegate a focused research question."
        assert data["data"]["parameters"] == research_tool.parameters
        assert data["data"]["outputs_to_string"] == {
            "source": "last_message",
            "handler": "haystack.tools.agent_tool._last_message_text",
        }
        assert "component" not in data["data"]
        assert data["data"]["agent"]["init_parameters"]["system_prompt"] == "You research things."

    def test_from_dict(self, research_tool):
        deserialized = AgentTool.from_dict(research_tool.to_dict())
        assert isinstance(deserialized, AgentTool)
        assert deserialized.name == research_tool.name
        assert deserialized.parameters == research_tool.parameters
        assert isinstance(deserialized._component, Agent)
        assert deserialized._component.system_prompt == "You research things."
        assert deserialized.invoke(**_delegation("the history of Tesla"))["last_message"].text == (
            "findings about the history of Tesla"
        )

    def test_serde_in_pipeline(self, research_tool):
        pipeline = Pipeline()
        pipeline.add_component("coordinator", _coordinator(research_tool))

        deserialized = Pipeline.loads(pipeline.dumps())
        result = deserialized.run({"coordinator": {"messages": [ChatMessage.from_user("Tell me about Tesla")]}})
        assert result["coordinator"]["messages"][2].tool_call_results[0].result == (
            "findings about the history of Tesla"
        )
