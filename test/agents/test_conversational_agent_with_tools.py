import pytest
from unittest.mock import MagicMock

from haystack.agents import Tool
from haystack.agents.conversational import ConversationalAgentWithTools
from haystack.agents.memory import ConversationSummaryMemory, NoMemory
from haystack.nodes import PromptNode


@pytest.fixture
def prompt_node():
    prompt_node = PromptNode("google/flan-t5-xxl", api_key="fake_key", max_length=256)
    return prompt_node


@pytest.mark.unit
def test_init(prompt_node):
    agent = ConversationalAgentWithTools(prompt_node)

    # Test normal case
    assert isinstance(agent.memory, ConversationSummaryMemory)
    assert callable(agent.prompt_parameters_resolver)
    assert agent.max_steps == 5
    assert agent.final_answer_pattern == r"Final Answer\s*:\s*(.*)"


@pytest.mark.unit
def test_tooling(prompt_node):
    agent = ConversationalAgentWithTools(prompt_node)
    # ConversationalAgentWithTools has no tools by default
    assert not agent.tm.tools

    # but can add tools
    agent.tm.add_tool(Tool("ExampleTool", lambda x: x, description="Example tool"))
    assert agent.tm.tools
    assert agent.tm.tools["ExampleTool"].name == "ExampleTool"


@pytest.mark.unit
def test_agent_with_memory(prompt_node):
    # Test with summary memory
    agent = ConversationalAgentWithTools(prompt_node, memory=ConversationSummaryMemory(prompt_node))
    assert isinstance(agent.memory, ConversationSummaryMemory)

    # Test with no memory
    agent = ConversationalAgentWithTools(prompt_node, memory=NoMemory())
    assert isinstance(agent.memory, NoMemory)


@pytest.mark.unit
def test_run(prompt_node):
    agent = ConversationalAgentWithTools(prompt_node)

    # Mock the Agent run method
    agent.run = MagicMock(return_value="Hello")
    assert agent.run("query") == "Hello"
    agent.run.assert_called_once_with("query")
