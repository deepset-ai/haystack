# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from enum import Enum
from math import inf
from typing import Any, Callable, Dict, List, Optional, Union

from haystack import Answer, component, default_from_dict, default_to_dict, logging
from haystack.core.component.types import Variadic

logger = logging.getLogger(__name__)


class JoinMode(Enum):
    """
    Enum for join mode.
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
    A component that joins multiple list of Answers into a single list.

    It supports different joins modes:
    - concatenate: Keeps the highest scored Answer in case of duplicates.

    """

    def __init__(self, join_mode: Union[str, JoinMode] = JoinMode.CONCATENATE, top_k: Optional[int] = None):
        """
        Create an AnswerJoiner component.

        :param join_mode:
            Specifies the join mode to use. Available modes:
            - `concatenate`
            Defaults to `concatenate`.
        :param top_k:
            The maximum number of Answers to return. Defaults to 10 if not specified.
        """
        if isinstance(join_mode, str):
            join_mode = JoinMode.from_str(join_mode)
        join_mode_functions: Dict[JoinMode, Callable[[List[List[Answer]]], List[Answer]]] = {
            JoinMode.CONCATENATE: self._concatenate
        }
        self.join_mode_function = join_mode_functions.get(join_mode)
        if not self.join_mode_function:
            raise ValueError(f"Join mode '{join_mode}' is not supported.")

        self.join_mode = join_mode
        self.top_k = top_k or 10

    @component.output_types(answers=List[Answer])
    def run(self, answers: Variadic[List[Answer]], top_k: Optional[int] = None):
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
        answers = list(answers)
        output_answers: List[Answer] = self.join_mode_function(answers)

        effective_top_k = top_k if top_k is not None else self.top_k
        output_answers = output_answers[:effective_top_k]

        return {"answers": output_answers}

    def _concatenate(self, answer_lists: List[List[Answer]]) -> List[Answer]:
        """
        Concatenate multiple lists of Answers, flattening them into a single list and sorting by score.

        :param answer_lists: List of lists of Answers to be concatenated.
        :returns: A list of Answers, sorted by score.
        """
        # flatten
        flattened_answers = list(itertools.chain.from_iterable(answer_lists))

        # sort in descending order
        sorted_answers = sorted(
            flattened_answers, key=lambda answer: answer.score if hasattr(answer, "score") else -inf, reverse=True
        )

        return sorted_answers

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(self, join_mode=str(self.join_mode), top_k=self.top_k)

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
