import pytest
from unittest.mock import MagicMock, patch, Mock

from haystack.agents import Agent, AgentStep
from haystack.agents.base import Tool
from haystack.agents.conversational import ConversationalAgent
from haystack.agents.memory import ConversationSummaryMemory, ConversationMemory, NoMemory
from haystack.nodes import PromptNode


@pytest.mark.unit
def test_init():
    with patch("haystack.nodes.prompt.prompt_template.fetch_from_prompthub") as mock_prompthub:
        mock_prompthub.side_effect = [("This is a test prompt. Use your knowledge to answer this question: {question}")]
        prompt_node = PromptNode()
        agent = ConversationalAgent(prompt_node)

    # Test normal case
    assert isinstance(agent.memory, ConversationMemory)
    assert callable(agent.prompt_parameters_resolver)
    assert agent.max_steps == 2
    assert agent.final_answer_pattern == r"^([\s\S]+)$"

    # ConversationalAgent doesn't have tools
    assert not agent.tm.tools


@pytest.mark.unit
def test_init_with_summary_memory():
    with patch("haystack.nodes.prompt.prompt_template.fetch_from_prompthub") as mock_prompthub:
        mock_prompthub.side_effect = [("This is a test prompt. Use your knowledge to answer this question: {question}")]
        prompt_node = PromptNode(default_prompt_template="this is a test")
        # Test with summary memory
        agent = ConversationalAgent(prompt_node, memory=ConversationSummaryMemory(prompt_node))
        assert isinstance(agent.memory, ConversationSummaryMemory)


@pytest.mark.unit
def test_init_with_no_memory():
    with patch("haystack.nodes.prompt.prompt_template.fetch_from_prompthub") as mock_prompthub:
        mock_prompthub.side_effect = [("This is a test prompt. Use your knowledge to answer this question: {question}")]
        prompt_node = PromptNode()
        # Test with no memory
        agent = ConversationalAgent(prompt_node, memory=NoMemory())
        assert isinstance(agent.memory, NoMemory)


@pytest.mark.unit
def test_run():
    with patch("haystack.nodes.prompt.prompt_template.fetch_from_prompthub") as mock_prompthub:
        mock_prompthub.side_effect = [("This is a test prompt. Use your knowledge to answer this question: {question}")]
        prompt_node = PromptNode()
        agent = ConversationalAgent(prompt_node)

        # Mock the Agent run method
        agent.run = MagicMock(return_value="Hello")
        assert agent.run("query") == "Hello"
        agent.run.assert_called_once_with("query")


@pytest.fixture
def prompt_node():
    prompt_node = PromptNode("google/flan-t5-xxl", api_key="fake_key", max_length=256)
    return prompt_node


@pytest.mark.unit
def test_init(prompt_node):
    agent = ConversationalAgent(prompt_node)

    # Test normal case
    assert isinstance(agent.memory, ConversationMemory)
    assert callable(agent.prompt_parameters_resolver)
    assert agent.prompt_template == "conversational-agent-with-tools"
    assert agent.max_steps == 5
    assert agent.final_answer_pattern == r"Final Answer\s*:\s*(.*)"


@pytest.mark.unit
def test_tooling(prompt_node):
    agent = ConversationalAgent(prompt_node)
    # ConversationalAgentWithTools has no tools by default
    assert not agent.tm.tools

    # but can add tools
    agent.tm.add_tool(Tool("ExampleTool", lambda x: x, description="Example tool"))
    assert agent.tm.tools
    assert agent.tm.tools["ExampleTool"].name == "ExampleTool"


@pytest.mark.unit
def test_agent_with_memory(prompt_node):
    # Test with summary memory
    agent = ConversationalAgent(prompt_node, memory=ConversationSummaryMemory(prompt_node))
    assert isinstance(agent.memory, ConversationSummaryMemory)

    # Test with no memory
    agent = ConversationalAgent(prompt_node, memory=NoMemory())
    assert isinstance(agent.memory, NoMemory)


@pytest.mark.unit
def test_run(prompt_node):
    """
    Test that the invocation of ConversationalAgent run method in turn invokes _step of the Agent superclass
    Make sure that the agent is starting from the correct step 1, and max_steps is 5
    """
    mock_step = Mock(spec=Agent._step)

    # Replace the original _step method with the mock
    Agent._step = mock_step

    # Initialize agent
    prompt_node = PromptNode()
    agent = ConversationalAgent(prompt_node)

    # Run agent
    agent.run(query="query")

    assert mock_step.call_count == 1

    # Check the parameters passed to _step method
    assert mock_step.call_args[0][0] == "query"
    agent_step = mock_step.call_args[0][1]
    expected_agent_step = AgentStep(
        current_step=1,
        max_steps=5,
        prompt_node_response="",
        final_answer_pattern=r"Final Answer\s*:\s*(.*)",
        transcript="",
    )
    # compare the string representation of the objects
    assert str(agent_step) == str(expected_agent_step)
