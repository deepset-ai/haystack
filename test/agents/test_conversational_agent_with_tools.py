import pytest
from unittest.mock import MagicMock, Mock, call

from haystack.agents import Tool, Agent, AgentStep
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
    """
    Test that the invocation of ConversationalAgentWithTools run method in turn invokes _step of the Agent superclass
    Make sure that the agent is starting from the correct step 1, and max_steps is 5
    """
    mock_step = Mock(spec=Agent._step)

    # Replace the original _step method with the mock
    Agent._step = mock_step

    # Initialize agent
    prompt_node = PromptNode()
    agent = ConversationalAgentWithTools(prompt_node)

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
