# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from itertools import chain
from typing import Any, Dict, List, Optional, Type

from haystack import component, default_from_dict, default_to_dict
from haystack.core.component.types import Variadic
from haystack.utils import deserialize_type, serialize_type


@component
class ListJoiner:
    """
    A component that joins multiple lists into a single flat list.

    The ListJoiner receives multiple lists of the same type and concatenates them into a single flat list.
    The output order respects the pipeline's execution sequence, with earlier inputs being added first.

    Usage example:
    ```python
    from haystack.components.builders import ChatPromptBuilder
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack import Pipeline
    from haystack.components.joiners import ListJoiner
    from typing import List


    user_message = [ChatMessage.from_user("Give a brief answer the following question: {{query}}")]

    feedback_prompt = \"""
        You are given a question and an answer.
        Your task is to provide a score and a brief feedback on the answer.
        Question: {{query}}
        Answer: {{response}}
        \"""
    feedback_message = [ChatMessage.from_system(feedback_prompt)]

    prompt_builder = ChatPromptBuilder(template=user_message)
    feedback_prompt_builder = ChatPromptBuilder(template=feedback_message)
    llm = OpenAIChatGenerator(model="gpt-4o-mini")
    feedback_llm = OpenAIChatGenerator(model="gpt-4o-mini")

    pipe = Pipeline()
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", llm)
    pipe.add_component("feedback_prompt_builder", feedback_prompt_builder)
    pipe.add_component("feedback_llm", feedback_llm)
    pipe.add_component("list_joiner", ListJoiner(List[ChatMessage]))

    pipe.connect("prompt_builder.prompt", "llm.messages")
    pipe.connect("prompt_builder.prompt", "list_joiner")
    pipe.connect("llm.replies", "list_joiner")
    pipe.connect("llm.replies", "feedback_prompt_builder.response")
    pipe.connect("feedback_prompt_builder.prompt", "feedback_llm.messages")
    pipe.connect("feedback_llm.replies", "list_joiner")

    query = "What is nuclear physics?"
    ans = pipe.run(data={"prompt_builder": {"template_variables":{"query": query}},
        "feedback_prompt_builder": {"template_variables":{"query": query}}})

    print(ans["list_joiner"]["values"])
    ```
    """

    def __init__(self, list_type_: Optional[Type] = None):
        """
        Creates a ListJoiner component.

        :param list_type_: The expected type of the lists this component will join (e.g., List[ChatMessage]).
            If specified, all input lists must conform to this type. If None, the component defaults to handling
            lists of any type including mixed types.
        """
        self.list_type_ = list_type_
        if list_type_ is not None:
            component.set_output_types(self, values=list_type_)
        else:
            component.set_output_types(self, values=List[Any])

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns: Dictionary with serialized data.
        """
        return default_to_dict(
            self, list_type_=serialize_type(self.list_type_) if self.list_type_ is not None else None
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ListJoiner":
        """
        Deserializes the component from a dictionary.

        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        init_parameters = data.get("init_parameters")
        if init_parameters is not None and init_parameters.get("list_type_") is not None:
            data["init_parameters"]["list_type_"] = deserialize_type(data["init_parameters"]["list_type_"])
        return default_from_dict(cls, data)

    def run(self, values: Variadic[List[Any]]) -> Dict[str, List[Any]]:
        """
        Joins multiple lists into a single flat list.

        :param values: The list to be joined.
        :returns: Dictionary with 'values' key containing the joined list.
        """
        result = list(chain(*values))
        return {"values": result}
