from unittest.mock import MagicMock
from haystack.nodes import PromptNode, PromptTemplate
import pytest
from typing import Dict, Any

from haystack.agents.memory import ConversationSummaryMemory


@pytest.fixture
def mocked_prompt_node():
    mock_prompt_node = MagicMock(spec=PromptNode)
    mock_prompt_node.default_prompt_template = PromptTemplate("Summarize the conversation: {chat_transcript}")
    mock_prompt_node.prompt.return_value = ["This is a summary."]
    return mock_prompt_node


@pytest.mark.unit
def test_conversation_summary_memory(mocked_prompt_node):
    summary = "This is a fake summary definitely."
    mocked_prompt_node.prompt.return_value = [summary]
    summary_mem = ConversationSummaryMemory(mocked_prompt_node)

    # Test saving and loading without summaries
    data1: Dict[str, Any] = {"input": "Hello", "output": "Hi there"}
    summary_mem.save(data1)
    assert summary_mem.load() == "\nHuman: Hello\nAI: Hi there\n"
    assert summary_mem.has_unsummarized_snippets()
    assert summary_mem.unsummarized_snippets() == 1

    data2: Dict[str, Any] = {"input": "How are you?", "output": "I'm doing well, thanks."}
    summary_mem.save(data2)
    assert summary_mem.load() == "\nHuman: Hello\nAI: Hi there\nHuman: How are you?\nAI: I'm doing well, thanks.\n"
    assert summary_mem.has_unsummarized_snippets()
    assert summary_mem.unsummarized_snippets() == 2

    # Test summarization
    data3: Dict[str, Any] = {"input": "What's the weather like?", "output": "It's sunny outside."}
    summary_mem.save(data3)
    assert summary_mem.load() == summary
    assert not summary_mem.has_unsummarized_snippets()
    assert summary_mem.unsummarized_snippets() == 0

    summary_mem.clear()
    assert summary_mem.load() == ""


@pytest.mark.unit
def test_conversation_summary_memory_lower_summary_frequency(mocked_prompt_node):
    summary = "This is a fake summary definitely."
    mocked_prompt_node.prompt.return_value = [summary]
    summary_mem = ConversationSummaryMemory(mocked_prompt_node, summary_frequency=2)

    data1: Dict[str, Any] = {"input": "Hello", "output": "Hi there"}
    summary_mem.save(data1)
    assert summary_mem.load() == "\nHuman: Hello\nAI: Hi there\n"
    assert summary_mem.has_unsummarized_snippets()
    assert summary_mem.unsummarized_snippets() == 1

    # Test summarization
    data2: Dict[str, Any] = {"input": "How are you?", "output": "I'm doing well, thanks."}
    summary_mem.save(data2)
    assert summary_mem.load() == summary
    assert not summary_mem.has_unsummarized_snippets()
    assert summary_mem.unsummarized_snippets() == 0

    data3: Dict[str, Any] = {"input": "What's the weather like?", "output": "It's sunny outside."}
    summary_mem.save(data3)
    assert summary_mem.load() == summary + "\nHuman: What's the weather like?\nAI: It's sunny outside.\n"
    assert summary_mem.has_unsummarized_snippets()
    assert summary_mem.unsummarized_snippets() == 1

    summary_mem.clear()
    assert summary_mem.load() == ""

    # start over
    summary_mem.save(data1)
    assert summary_mem.load() == "\nHuman: Hello\nAI: Hi there\n"
    assert summary_mem.has_unsummarized_snippets()
    assert summary_mem.unsummarized_snippets() == 1

    # Test summarization
    data2: Dict[str, Any] = {"input": "How are you?", "output": "I'm doing well, thanks."}
    summary_mem.save(data2)
    assert summary_mem.load() == summary
    assert not summary_mem.has_unsummarized_snippets()
    assert summary_mem.unsummarized_snippets() == 0


@pytest.mark.unit
def test_conversation_summary_is_accumulating(mocked_prompt_node):
    # ensure that the summary memory works after being triggered twice
    summary = "This is a fake summary definitely."
    mocked_prompt_node.prompt.return_value = [summary]
    summary_mem = ConversationSummaryMemory(mocked_prompt_node, summary_frequency=2)

    data1: Dict[str, Any] = {"input": "Hello", "output": "Hi there"}
    summary_mem.save(data1)
    assert summary_mem.load() == "\nHuman: Hello\nAI: Hi there\n"
    assert summary_mem.has_unsummarized_snippets()
    assert summary_mem.unsummarized_snippets() == 1

    # Test summarization
    data2: Dict[str, Any] = {"input": "How are you?", "output": "I'm doing well, thanks."}
    summary_mem.save(data2)
    assert summary_mem.load() == summary
    assert not summary_mem.has_unsummarized_snippets()
    assert summary_mem.unsummarized_snippets() == 0

    # Add more snippets
    new_snippet = "\nHuman: What's the weather like?\nAI: It's sunny outside.\n"
    data3: Dict[str, Any] = {"input": "What's the weather like?", "output": "It's sunny outside."}
    summary_mem.save(data3)
    assert summary_mem.load() == summary + new_snippet
    assert summary_mem.has_unsummarized_snippets()
    assert summary_mem.unsummarized_snippets() == 1

    # Trigger summarization again
    data3: Dict[str, Any] = {"input": "What's the weather tomorrow?", "output": "It will be sunny."}
    summary_mem.save(data3)

    # Ensure that the summary is accumulating
    assert summary_mem.load() == summary + summary
    assert not summary_mem.has_unsummarized_snippets()


@pytest.mark.unit
def test_conversation_summary_memory_with_template(mocked_prompt_node):
    pt = PromptTemplate("Summarize the conversation: {chat_transcript}")
    summary_mem = ConversationSummaryMemory(mocked_prompt_node, prompt_template=pt)

    data1: Dict[str, Any] = {"input": "Hello", "output": "Hi there"}
    summary_mem.save(data1)
    assert summary_mem.load() == "\nHuman: Hello\nAI: Hi there\n"

    data2: Dict[str, Any] = {"input": "How are you?", "output": "I'm doing well, thanks."}
    summary_mem.save(data2)
    assert summary_mem.load() == "\nHuman: Hello\nAI: Hi there\nHuman: How are you?\nAI: I'm doing well, thanks.\n"

    data3: Dict[str, Any] = {"input": "What's the weather like?", "output": "It's sunny outside."}
    summary_mem.save(data3)
    assert summary_mem.load() == "This is a summary."

    summary_mem.clear()
    assert summary_mem.load() == ""
