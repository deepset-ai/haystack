# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import mimetypes
from pathlib import Path
from typing import Any, Optional, Union, overload

from numpy import exp, ndarray

CUSTOM_MIMETYPES = {
    # we add markdown because it is not added by the mimetypes module
    # see https://github.com/python/cpython/pull/17995
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    # we add msg because it is not added by the mimetypes module
    ".msg": "application/vnd.ms-outlook",
}


def expand_page_range(page_range: list[Union[str, int]]) -> list[int]:
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


@overload
def expit(x: float) -> float: ...
@overload
def expit(x: ndarray[Any, Any]) -> ndarray[Any, Any]: ...
def expit(x: Union[float, ndarray[Any, Any]]) -> Union[float, ndarray[Any, Any]]:
    """
    Compute logistic sigmoid function. Maps input values to a range between 0 and 1

    :param x: input value. Can be a scalar or a numpy array.
    """
    return 1 / (1 + exp(-x))


def _guess_mime_type(path: Path) -> Optional[str]:
    """
    Guess the MIME type of the provided file path.

    :param path: The file path to get the MIME type for.

    :returns: The MIME type of the provided file path, or `None` if the MIME type cannot be determined.
    """
    extension = path.suffix.lower()
    mime_type = mimetypes.guess_type(path.as_posix())[0]
    # lookup custom mappings if the mime type is not found
    return CUSTOM_MIMETYPES.get(extension, mime_type)
