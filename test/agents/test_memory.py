import pytest
from typing import Dict, Any
from haystack.agents.memory import NoMemory, ConversationMemory


@pytest.mark.unit
def test_no_memory():
    no_mem = NoMemory()
    assert no_mem.load() == ""
    no_mem.save({"key": "value"})
    no_mem.clear()


@pytest.mark.unit
def test_conversation_memory():
    conv_mem = ConversationMemory()
    assert conv_mem.load() == ""
    data: Dict[str, Any] = {"input": "Hello", "output": "Hi there"}
    conv_mem.save(data)
    assert conv_mem.load() == "Human: Hello\nAI: Hi there\n"

    data: Dict[str, Any] = {"input": "How are you?", "output": "I'm doing well, thanks."}
    conv_mem.save(data)
    assert conv_mem.load() == "Human: Hello\nAI: Hi there\nHuman: How are you?\nAI: I'm doing well, thanks.\n"
    assert conv_mem.load(window_size=1) == "Human: How are you?\nAI: I'm doing well, thanks.\n"

    conv_mem.clear()
    assert conv_mem.load() == ""


@pytest.mark.unit
def test_conversation_memory_window_size():
    conv_mem = ConversationMemory()
    assert conv_mem.load() == ""
    data: Dict[str, Any] = {"input": "Hello", "output": "Hi there"}
    conv_mem.save(data)
    data: Dict[str, Any] = {"input": "How are you?", "output": "I'm doing well, thanks."}
    conv_mem.save(data)
    assert conv_mem.load() == "Human: Hello\nAI: Hi there\nHuman: How are you?\nAI: I'm doing well, thanks.\n"
    assert conv_mem.load(window_size=1) == "Human: How are you?\nAI: I'm doing well, thanks.\n"

    # clear the memory
    conv_mem.clear()
    assert conv_mem.load() == ""
    assert conv_mem.load(window_size=1) == ""
