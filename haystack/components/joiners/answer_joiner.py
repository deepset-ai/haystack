# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from enum import Enum
from math import inf
from typing import Any, Callable, Dict, List, Optional, Union

from haystack import Answer, component, default_from_dict, default_to_dict, logging
from haystack.core.component.types import Variadic
from haystack.utils import deserialize_callable, serialize_callable

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
    The `AnswerJoiner` is useful for merging of multiple lists of `Answer` objects into a single unified list.

    This component is useful in scenarios where you have answers from different generators and need to combine
    them into a single list. One option of merging is to use predefined join modes, such as `CONCATENATE`, which
    keeps the highest scored answer in case of duplicates. Another option is to provide a custom join function
    that takes a list of lists of `Answer` objects and returns a single list of `Answer` objects.
    """

    def __init__(
        self,
        join_mode: Union[str, JoinMode] = JoinMode.CONCATENATE,
        custom_join_function: Optional[Callable[[List[List[Answer]]], List[Answer]]] = None,
        top_k: Optional[int] = None,
    ):
        """
        Create an AnswerJoiner component.

        :param join_mode:
            Specifies the join mode to use. Available modes:
            - `concatenate`
            Defaults to `concatenate`.
        :param custom_join_function:
            A custom function to join multiple lists of Answers into a single list. If provided, it
            takes precedence over the `join_mode` parameter.
        :param top_k:
            The maximum number of Answers to return. Defaults to 10 if not specified.
        """
        if isinstance(join_mode, str):
            join_mode = JoinMode.from_str(join_mode)
        join_mode_functions: Dict[JoinMode, Callable[[List[List[Answer]]], List[Answer]]] = {
            JoinMode.CONCATENATE: self._concatenate
        }
        if join_mode not in join_mode_functions:
            raise ValueError(f"Join mode '{join_mode}' is not supported.")

        if custom_join_function and not callable(custom_join_function):
            raise ValueError("The provided custom_join_function is not callable.")

        # we'll need to serialize the custom_join_function
        self.custom_join_function = custom_join_function

        # Assign the join function: prioritize custom function if provided
        self.join_mode_function: Callable[[List[List[Answer]]], List[Answer]] = (
            custom_join_function if custom_join_function else join_mode_functions[join_mode]
        )

        self.join_mode = join_mode
        self.top_k = top_k or 10

    @component.output_types(answers=List[Answer])
    def run(
        self,
        answers: Variadic[List[Answer]],
        custom_join_function: Optional[Callable[[List[List[Answer]]], List[Answer]]] = None,
        top_k: Optional[int] = None,
    ):
        """
        Joins multiple lists of Answers into a single list depending on the `join_mode` parameter.

        :param answers:
            Nested list of Answers to be merged.
        :param custom_join_function:
            A custom function to join lists of Answers. Overrides the default behavior if provided.
            The function should accept a list of lists of Answers and return a single list of Answers.

        :param top_k:
            The maximum number of Answers to return. Overrides the instance's `top_k` if provided.

        :returns:
            A dictionary with the following keys:
            - `answers`: Merged list of Answers
        """
        answers_list = list(answers)

        if custom_join_function and not callable(custom_join_function):
            raise ValueError("The provided custom_join_function is not callable.")

        # Use custom join function if provided at runtime, else use the init join function
        join_function = custom_join_function or self.join_mode_function

        output_answers: List[Answer] = join_function(answers_list)

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

        # sort in descending order by score if available
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
        return default_to_dict(
            self,
            join_mode=str(self.join_mode),
            custom_join_function=serialize_callable(self.custom_join_function) if self.custom_join_function else None,
            top_k=self.top_k,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnswerJoiner":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        init_params = data.get("init_parameters", {})
        custom_join_function = init_params.get("custom_join_function")
        if custom_join_function:
            data["init_parameters"]["custom_join_function"] = deserialize_callable(custom_join_function)
        return default_from_dict(cls, data)
