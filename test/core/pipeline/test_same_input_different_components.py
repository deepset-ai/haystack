from typing import List

from haystack.components.builders import DynamicChatPromptBuilder
from haystack import Pipeline
from haystack.dataclasses import ChatMessage
from haystack import component
from unittest.mock import MagicMock


class MethodTracker:
    # This class is used to track the number of times a method is called and with which arguments
    def __init__(self, method):
        self.method = method
        self.call_count = 0
        self.called_with = None

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self.called_with = (args, kwargs)
        return self.method(*args, **kwargs)


@component
class MessageMerger:
    @component.output_types(merged_message=str)
    def run(self, messages: List[ChatMessage], metadata: dict = None):
        return {"merged_message": "\n".join(t.content for t in messages)}


@component
class FakeGenerator:
    # This component is a fake generator that always returns the same message
    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage]):
        return {"replies": [ChatMessage.from_assistant("Fake message")]}


def test_same_input_different_components():
    """
    Test that passing the same input reference to different components
    does not alter the correct Pipeline run logic.
    """

    prompt_builder = DynamicChatPromptBuilder()
    llm = FakeGenerator()
    mm1 = MessageMerger()
    mm2 = MessageMerger()

    mm1_tracked_run = MethodTracker(mm1.run)
    mm1.run = mm1_tracked_run

    mm2_tracked_run = MethodTracker(mm2.run)
    mm2.run = mm2_tracked_run

    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.add_component("mm1", mm1)
    pipe.add_component("mm2", mm2)

    pipe.connect("prompt_builder.prompt", "llm.messages")
    pipe.connect("prompt_builder.prompt", "mm1")
    pipe.connect("llm.replies", "mm2")

    location = "Berlin"
    messages = [
        ChatMessage.from_system("Always respond in English even if some input data is in other languages."),
        ChatMessage.from_user("Tell me about {{location}}"),
    ]
    params = {"metadata": {"metadata_key": "metadata_value", "meta2": "value2"}}

    pipe.run(
        data={
            "prompt_builder": {"template_variables": {"location": location}, "prompt_source": messages},
            "mm1": params,
            "mm2": params,
        }
    )

    assert mm1_tracked_run.call_count == 1
    assert mm1_tracked_run.called_with[1]["metadata"] == params["metadata"]

    assert mm2_tracked_run.call_count == 1
    assert mm2_tracked_run.called_with[1]["metadata"] == params["metadata"]
