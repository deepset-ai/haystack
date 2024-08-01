# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from enum import Enum
from math import inf
from typing import Any, Callable, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.core.component.types import Variadic
from haystack.dataclasses.answer import ExtractedAnswer, ExtractedTableAnswer, GeneratedAnswer

AnswerType = Union[GeneratedAnswer, ExtractedTableAnswer, ExtractedAnswer]

logger = logging.getLogger(__name__)


class JoinMode(Enum):
    """
    Enum for AnswerJoiner join modes.
    """

    CONCATENATE = "concatenate"

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(string: str) -> "JoinMode":
        """
        Convert a string to a JoinMode enum.
        """
        enum_map = {e.value: e for e in JoinMode}
        mode = enum_map.get(string)
        if mode is None:
            msg = f"Unknown join mode '{string}'. Supported modes in AnswerJoiner are: {list(enum_map.keys())}"
            raise ValueError(msg)
        return mode


@component
class AnswerJoiner:
    """
    Merges multiple lists of `Answer` objects into a single list.

    Use this component to combine answers from different Generators into a single list.
    Currently, the component supports only one join mode: `CONCATENATE`.
    This mode concatenates multiple lists of answers into a single list.

    ### Usage example

    In this example, AnswerJoiner merges answers from two different Generators:

    ```python
    from haystack.components.builders import AnswerBuilder
    from haystack.components.joiners import AnswerJoiner

    from haystack.core.pipeline import Pipeline

    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage


    query = "What's Natural Language Processing?"
    messages = [ChatMessage.from_system("You are a helpful, respectful and honest assistant. Be super concise."),
                ChatMessage.from_user(query)]

    pipe = Pipeline()
    pipe.add_component("gpt-4o", OpenAIChatGenerator(model="gpt-4o"))
    pipe.add_component("llama", OpenAIChatGenerator(model="gpt-3.5-turbo"))
    pipe.add_component("aba", AnswerBuilder())
    pipe.add_component("abb", AnswerBuilder())
    pipe.add_component("joiner", AnswerJoiner())

    pipe.connect("gpt-4o.replies", "aba")
    pipe.connect("llama.replies", "abb")
    pipe.connect("aba.answers", "joiner")
    pipe.connect("abb.answers", "joiner")

    results = pipe.run(data={"gpt-4o": {"messages": messages},
                                "llama": {"messages": messages},
                                "aba": {"query": query},
                                "abb": {"query": query}})
    ```
    """

    def __init__(
        self,
        join_mode: Union[str, JoinMode] = JoinMode.CONCATENATE,
        top_k: Optional[int] = None,
        sort_by_score: bool = False,
    ):
        """
        Creates an AnswerJoiner component.

        :param join_mode:
            Specifies the join mode to use. Available modes:
            - `concatenate`: Concatenates multiple lists of Answers into a single list.
        :param top_k:
            The maximum number of Answers to return.
        :param sort_by_score:
            If `True`, sorts the documents by score in descending order.
            If a document has no score, it is handled as if its score is -infinity.
        """
        if isinstance(join_mode, str):
            join_mode = JoinMode.from_str(join_mode)
        join_mode_functions: Dict[JoinMode, Callable[[List[List[AnswerType]]], List[AnswerType]]] = {
            JoinMode.CONCATENATE: self._concatenate
        }
        self.join_mode_function: Callable[[List[List[AnswerType]]], List[AnswerType]] = join_mode_functions[join_mode]
        self.join_mode = join_mode
        self.top_k = top_k
        self.sort_by_score = sort_by_score

    @component.output_types(answers=List[AnswerType])
    def run(self, answers: Variadic[List[AnswerType]], top_k: Optional[int] = None):
        """
        Joins multiple lists of Answers into a single list depending on the `join_mode` parameter.

        :param answers:
            Nested list of Answers to be merged.

        :param top_k:
            The maximum number of Answers to return. Overrides the instance's `top_k` if provided.

        :returns:
            A dictionary with the following keys:
            - `answers`: Merged list of Answers
        """
        answers_list = list(answers)
        join_function = self.join_mode_function
        output_answers: List[AnswerType] = join_function(answers_list)

        if self.sort_by_score:
            output_answers = sorted(
                output_answers, key=lambda answer: answer.score if hasattr(answer, "score") else -inf, reverse=True
            )

        top_k = top_k or self.top_k
        if top_k:
            output_answers = output_answers[:top_k]
        return {"answers": output_answers}

    def _concatenate(self, answer_lists: List[List[AnswerType]]) -> List[AnswerType]:
        """
        Concatenate multiple lists of Answers, flattening them into a single list and sorting by score.

        :param answer_lists: List of lists of Answers to be flattened.
        """
        return list(itertools.chain.from_iterable(answer_lists))

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, join_mode=str(self.join_mode), top_k=self.top_k, sort_by_score=self.sort_by_score)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnswerJoiner":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)
