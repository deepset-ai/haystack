from unittest.mock import MagicMock

import pytest
from typing import Dict, Any
from haystack.nodes import PromptNode, PromptTemplate
from haystack.agents.memory import NoMemory, ConversationMemory, ConversationSummaryMemory


@pytest.fixture
def mocked_prompt_node():
    mock_prompt_node = MagicMock(spec=PromptNode)
    mock_prompt_node.default_prompt_template = PromptTemplate(
        "conversational-summary", "Summarize the conversation: {chat_transcript}"
    )
    mock_prompt_node.prompt.return_value = ["This is a summary."]
    return mock_prompt_node


@pytest.fixture
def mocked_prompt_template():
    return PromptTemplate("conversational-summary", "Summarize the conversation: {chat_transcript}")


@pytest.mark.unit
def test_no_memory():
    no_mem = NoMemory()
    assert no_mem.load() == {}
    no_mem.save({"key": "value"})
    no_mem.clear()


@pytest.mark.unit
def test_conversation_memory():
    conv_mem = ConversationMemory()
    assert conv_mem.load() == ""
    data: Dict[str, Any] = {"input": "Hello", "output": "Hi there"}
    conv_mem.save(data)
    assert conv_mem.load() == "Human: Hello\nAI: Hi there\n"
    conv_mem.clear()
    assert conv_mem.load() == ""


@pytest.mark.unit
def test_conversation_summary_memory(mocked_prompt_node):
    mocked_prompt_node.prompt.return_value = ["This is a fake summary definitely."]
    summary_mem = ConversationSummaryMemory(mocked_prompt_node)

    # Test saving and loading without summaries
    data1: Dict[str, Any] = {"input": "Hello", "output": "Hi there"}
    summary_mem.save(data1)
    assert summary_mem.load() == "\n Human: Hello\nAI: Hi there\n"

    data2: Dict[str, Any] = {"input": "How are you?", "output": "I'm doing well, thanks."}
    summary_mem.save(data2)
    assert summary_mem.load() == "\n Human: Hello\nAI: Hi there\nHuman: How are you?\nAI: I'm doing well, thanks.\n"

    # Test summarization
    data3: Dict[str, Any] = {"input": "What's the weather like?", "output": "It's sunny outside."}
    summary_mem.save(data3)
    assert summary_mem.load() == "This is a fake summary definitely.\n "

    summary_mem.clear()
    assert summary_mem.load() == "\n "


@pytest.mark.unit
def test_conversation_summary_memory_with_template(mocked_prompt_node, mocked_prompt_template):
    summary_mem = ConversationSummaryMemory(mocked_prompt_node, prompt_template=mocked_prompt_template)

    data1: Dict[str, Any] = {"input": "Hello", "output": "Hi there"}
    summary_mem.save(data1)
    assert summary_mem.load() == "\n Human: Hello\nAI: Hi there\n"

    data2: Dict[str, Any] = {"input": "How are you?", "output": "I'm doing well, thanks."}
    summary_mem.save(data2)
    assert summary_mem.load() == "\n Human: Hello\nAI: Hi there\nHuman: How are you?\nAI: I'm doing well, thanks.\n"

    data3: Dict[str, Any] = {"input": "What's the weather like?", "output": "It's sunny outside."}
    summary_mem.save(data3)
    assert summary_mem.load() == "This is a summary.\n "

    summary_mem.clear()
    assert summary_mem.load() == "\n "
