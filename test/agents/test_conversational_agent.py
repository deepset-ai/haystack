import pytest
from unittest.mock import MagicMock, patch

from haystack.errors import AgentError
from haystack.agents.base import Tool
from haystack.agents.conversational import ConversationalAgent
from haystack.agents.memory import ConversationSummaryMemory, ConversationMemory, NoMemory
from haystack.nodes import PromptNode


@pytest.fixture
@patch("haystack.nodes.prompt.prompt_node.PromptModel")
def prompt_node(mock_model):
    prompt_node = PromptNode()
    return prompt_node


@pytest.mark.unit
def test_init_without_tools(prompt_node):
    agent = ConversationalAgent(prompt_node)

    # Test normal case
    assert isinstance(agent.memory, ConversationMemory)
    assert callable(agent.prompt_parameters_resolver)
    assert agent.max_steps == 2
    assert agent.final_answer_pattern == r"^([\s\S]+)$"
    assert agent.prompt_template.name == "conversational-agent-without-tools"

    # ConversationalAgent doesn't have tools
    assert not agent.tm.tools


@pytest.mark.unit
def test_init_with_tools(prompt_node):
    agent = ConversationalAgent(prompt_node, tools=[Tool("ExampleTool", lambda x: x, description="Example tool")])

    # Test normal case
    assert isinstance(agent.memory, ConversationMemory)
    assert callable(agent.prompt_parameters_resolver)
    assert agent.max_steps == 5
    assert agent.final_answer_pattern == r"Final Answer\s*:\s*(.*)"
    assert agent.prompt_template.name == "conversational-agent"
    assert agent.has_tool("ExampleTool")


@pytest.mark.unit
def test_init_with_summary_memory(prompt_node):
    # Test with summary memory
    agent = ConversationalAgent(prompt_node, memory=ConversationSummaryMemory(prompt_node))
    assert isinstance(agent.memory, ConversationSummaryMemory)


@pytest.mark.unit
def test_init_with_no_memory(prompt_node):
    # Test with no memory
    agent = ConversationalAgent(prompt_node, memory=NoMemory())
    assert isinstance(agent.memory, NoMemory)


@pytest.mark.unit
def test_init_with_custom_max_steps(prompt_node):
    # Test with custom max step
    agent = ConversationalAgent(prompt_node, max_steps=8)
    assert agent.max_steps == 8


@pytest.mark.unit
def test_init_with_custom_prompt_template(prompt_node):
    # Test with custom prompt template
    agent = ConversationalAgent(prompt_node, prompt_template="translation")
    assert agent.prompt_template.name == "translation"


@pytest.mark.unit
def test_run(prompt_node):
    agent = ConversationalAgent(prompt_node)

    # Mock the Agent run method
    agent.run = MagicMock(return_value="Hello")
    assert agent.run("query") == "Hello"
    agent.run.assert_called_once_with("query")


@pytest.mark.unit
def test_add_tool(prompt_node):
    agent = ConversationalAgent(prompt_node, tools=[Tool("ExampleTool", lambda x: x, description="Example tool")])
    # ConversationalAgent has tools
    assert len(agent.tm.tools) == 1

    # and add more tools if ConversationalAgent is initialized with tools
    agent.add_tool(Tool("AnotherTool", lambda x: x, description="Example tool"))
    assert len(agent.tm.tools) == 2


@pytest.mark.unit
def test_add_tool_not_allowed(prompt_node):
    agent = ConversationalAgent(prompt_node)
    # ConversationalAgent has no tools
    assert not agent.tm.tools

    # and can't add tools when a ConversationalAgent is initialized without tools
    with pytest.raises(
        AgentError, match="You cannot add tools after initializing the ConversationalAgent without any tools."
    ):
        agent.add_tool(Tool("ExampleTool", lambda x: x, description="Example tool"))
