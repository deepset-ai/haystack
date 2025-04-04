# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Union

from numpy import exp

from haystack.tools.tool import Tool
from haystack.tools.toolset import Toolset


def expand_page_range(page_range: List[Union[str, int]]) -> List[int]:
    """
    Takes a list of page numbers and ranges and expands them into a list of page numbers.

    For example, given a page_range=['1-3', '5', '8', '10-12'] the function will return [1, 2, 3, 5, 8, 10, 11, 12]

    :param page_range: List of page numbers and ranges
    :returns:
        An expanded list of page integers

    """
    expanded_page_range = []

    for page in page_range:
        if isinstance(page, int):
            # check if it's a range wrongly passed as an integer expression
            if "-" in str(page):
                msg = "range must be a string in the format 'start-end'"
                raise ValueError(f"Invalid page range: {page} - {msg}")
            expanded_page_range.append(page)

        elif isinstance(page, str) and page.isdigit():
            expanded_page_range.append(int(page))

        elif isinstance(page, str) and "-" in page:
            start, end = page.split("-")
            expanded_page_range.extend(range(int(start), int(end) + 1))

        else:
            msg = "range must be a string in the format 'start-end' or an integer"
            raise ValueError(f"Invalid page range: {page} - {msg}")

    if not expanded_page_range:
        raise ValueError("No valid page numbers or ranges found in the input list")

    return expanded_page_range


def expit(x) -> float:
    """
    Compute logistic sigmoid function. Maps input values to a range between 0 and 1

    :param x: input value. Can be a scalar or a numpy array.
    """
    return 1 / (1 + exp(-x))


def serialize_tools_or_toolset(
    tools: Union[Toolset, List[Tool], None],
) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
    """
    Serialize a Toolset or a list of Tools to a dictionary or a list of tool dictionaries.

    :param tools: A Toolset, a list of Tools, or None
    :returns: A dictionary, a list of tool dictionaries, or None if tools is None
    """
    if tools is None:
        return None
    if isinstance(tools, Toolset):
        return tools.to_dict()
    else:
        return [tool.to_dict() for tool in tools] if tools else []
