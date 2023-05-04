import pytest
from unittest.mock import MagicMock

from haystack.agents.base import ToolsManager, Tool
from haystack.agents.conversational import ConversationalAgent, ConversationalAgentWithTools
from haystack.agents.memory import ConversationSummaryMemory
from haystack.nodes import PromptNode


@pytest.fixture
def prompt_node():
    return PromptNode("gpt-3.5-turbo", api_key="som_fake_key", max_length=256)


@pytest.fixture
def tools_manager():
    example_tool_node = MagicMock()  # Replace with an actual tool node or a MagicMock of a tool node
    return ToolsManager([Tool(name="ExampleTool", pipeline_or_node=example_tool_node, description="Example tool")])


@pytest.mark.unit
def test_conversational_agent_init(prompt_node):
    agent = ConversationalAgent(prompt_node)
    assert isinstance(agent.memory, ConversationSummaryMemory)
    assert agent.prompt_node == prompt_node


@pytest.mark.unit
def test_conversational_agent_run(prompt_node):
    agent = ConversationalAgent(prompt_node)
    prompt_node.prompt = MagicMock(return_value=["Test response"])
    user_input = "Test input"
    response = agent.run(user_input)
    assert response["answers"][0].answer == "Test response"


@pytest.mark.unit
def test_conversational_agent_with_tools_init(prompt_node, tools_manager):
    agent = ConversationalAgentWithTools(prompt_node, tools_manager=tools_manager)
    assert isinstance(agent.memory, ConversationSummaryMemory)
    assert agent.prompt_node == prompt_node
    assert agent.tm == tools_manager


@pytest.mark.unit
def test_conversational_agent_with_tools_run(prompt_node, tools_manager):
    agent = ConversationalAgentWithTools(prompt_node, tools_manager=tools_manager)
    prompt_node.prompt = MagicMock(return_value=["Final Answer: Test response"])
    user_input = "Test input"
    response = agent.run(user_input)
    assert response["answers"][0].answer == "Test response"
