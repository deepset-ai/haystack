import pytest
from unittest.mock import MagicMock, Mock

from haystack.agents.conversational import ConversationalAgent
from haystack.agents.memory import ConversationSummaryMemory, ConversationMemory, NoMemory


@pytest.mark.unit
def test_init():
    prompt_node = Mock()
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
    # Test with summary memory
    prompt_node = Mock()
    agent = ConversationalAgent(prompt_node, memory=ConversationSummaryMemory(prompt_node))
    assert isinstance(agent.memory, ConversationSummaryMemory)


@pytest.mark.unit
def test_init_with_no_memory():
    prompt_node = Mock()
    # Test with no memory
    agent = ConversationalAgent(prompt_node, memory=NoMemory())
    assert isinstance(agent.memory, NoMemory)


@pytest.mark.unit
def test_run():
    prompt_node = Mock()
    agent = ConversationalAgent(prompt_node)

    # Mock the Agent run method
    agent.run = MagicMock(return_value="Hello")
    assert agent.run("query") == "Hello"
    agent.run.assert_called_once_with("query")
