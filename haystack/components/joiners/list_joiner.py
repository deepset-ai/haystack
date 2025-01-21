# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Generic, TypeVar, Optional

from haystack import component, logging
from haystack.core.component.types import Variadic

logger = logging.getLogger(__name__)

T = TypeVar('T')

@component
class ListJoiner(Generic[T]):
    """
    Joins multiple lists into a single flat list, preserving the order of inputs.

    Use this component when you need to merge multiple lists (e.g., `List[ChatMessage]`)
    into a single list while maintaining the sequence of inputs.

    ### Usage example:

    ```python
    from haystack import Pipeline
    from haystack.dataclasses import ChatMessage
    from haystack.components.joiners import ListJoiner

    messages1 = [ChatMessage.from_user("Hello"), ChatMessage.from_assistant("Hi there")]
    messages2 = [ChatMessage.from_user("How are you?")]

    p = Pipeline()
    p.add_component("joiner", ListJoiner())
    result = p.run(data={"lists": [messages1, messages2]})
    # result.messages will be [messages1[0], messages1[1], messages2[0]]
    ```
    """

    def __init__(self):
        """
        Creates a new ListJoiner instance.
        The joiner preserves the order of inputs, adding earlier inputs first.
        """

    @component.output_types(joined_list=List[Any])
    def run(self, lists: Variadic[List[T]]) -> Dict[str, List[T]]:
        """
        Join multiple input lists into a single flat list.

        :param lists: Multiple lists to be joined.
        
        :returns:
        A dictionary with the following key:
            - `joined_list`: A single list containing all elements from input lists in order.
        """
        output: List[T] = []
        for input_list in lists:
            output.extend(input_list)
        return {"joined_list": output}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return {}