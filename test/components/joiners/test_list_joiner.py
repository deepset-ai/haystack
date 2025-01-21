import pytest

from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack.components.joiners import ListJoiner


def test_list_joiner_empty():
    """
    Test that ListJoiner correctly handles an empty list of inputs
    """
    joiner = ListJoiner()
    result = joiner.run([])
    assert result["joined_list"] == []


def test_list_joiner_single_list():
    """
    Test that ListJoiner correctly handles a single list input
    """
    input_list = [1, 2, 3]
    joiner = ListJoiner()
    result = joiner.run([input_list])
    assert result["joined_list"] == input_list


def test_list_joiner_multiple_lists():
    """
    Test that ListJoiner correctly merges multiple lists maintaining order
    """
    list1 = [1, 2]
    list2 = [3, 4]
    list3 = [5]
    
    joiner = ListJoiner()
    result = joiner.run([list1, list2, list3])
    
    # Input order should be preserved
    assert result["joined_list"] == [1, 2, 3, 4, 5]


def test_list_joiner_chat_messages():
    """
    Test that ListJoiner correctly handles ChatMessage lists
    """
    messages1 = [
        ChatMessage.from_user("Hello"),
        ChatMessage.from_assistant("Hi there")
    ]
    messages2 = [ChatMessage.from_user("How are you?")]
    
    joiner = ListJoiner()
    result = joiner.run([messages1, messages2])
    
    assert len(result["joined_list"]) == 3
    assert all(isinstance(msg, ChatMessage) for msg in result["joined_list"])
    # Order should be preserved
    assert result["joined_list"][0].text == "Hello"
    assert result["joined_list"][1].text == "Hi there"
    assert result["joined_list"][2].text == "How are you?"


def test_list_joiner_in_pipeline():
    """
    Test that ListJoiner works correctly in a Pipeline
    """
    messages1 = [
        ChatMessage.from_user("Hello"),
        ChatMessage.from_assistant("Hi there")
    ]
    messages2 = [ChatMessage.from_user("How are you?")]
    
    p = Pipeline()
    p.add_component("joiner", ListJoiner())
    
    result = p.run(data={"lists": [messages1, messages2]})
    
    assert len(result["joiner"]["joined_list"]) == 3
    assert all(isinstance(msg, ChatMessage) for msg in result["joiner"]["joined_list"])