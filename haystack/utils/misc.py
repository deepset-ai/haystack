# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import mimetypes
import tempfile
from math import inf
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

from numpy import exp, ndarray

from haystack import logging

if TYPE_CHECKING:
    from haystack.dataclasses import Document

CUSTOM_MIMETYPES = {
    # we add markdown because it is not added by the mimetypes module
    # see https://github.com/python/cpython/pull/17995
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    # we add msg because it is not added by the mimetypes module
    ".msg": "application/vnd.ms-outlook",
}

logger = logging.getLogger(__name__)


def expand_page_range(page_range: list[str | int]) -> list[int]:
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
def expit(x: float | ndarray[Any, Any]) -> float | ndarray[Any, Any]:
    """
    Compute logistic sigmoid function. Maps input values to a range between 0 and 1

    :param x: input value. Can be a scalar or a numpy array.
    """
    return 1 / (1 + exp(-x))


def _guess_mime_type(path: Path) -> str | None:
    """
    Guess the MIME type of the provided file path.

    :param path: The file path to get the MIME type for.

    :returns: The MIME type of the provided file path, or `None` if the MIME type cannot be determined.
    """
    extension = path.suffix.lower()
    mime_type = mimetypes.guess_type(path.as_posix())[0]
    # lookup custom mappings if the mime type is not found
    return CUSTOM_MIMETYPES.get(extension, mime_type)


def _get_output_dir(out_dir: str) -> str:
    """
    Find or create a writable directory for saving status files.

    Tries in the following order:

        1. ~/.haystack/{out_dir}
        2. {tempdir}/haystack/{out_dir}
        3. ./.haystack/{out_dir}

    :raises RuntimeError: If no directory could be created.
    :returns:
        The path to the created directory.
    """

    candidates = [
        Path.home() / ".haystack" / out_dir,
        Path(tempfile.gettempdir()) / "haystack" / out_dir,
        Path.cwd() / ".haystack" / out_dir,
    ]

    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return str(candidate)
        except Exception:
            continue

    raise RuntimeError(
        f"Could not create a writable directory for output files in any of the following locations: {candidates}"
    )


def _deduplicate_documents(documents: list["Document"]) -> list["Document"]:
    """
    Deduplicate a list of documents by their id keeping the duplicate with the highest score if a score is present.

    :param documents: List of documents to deduplicate.
    :returns: List of deduplicated documents.
    """
    # Keep for each Document id the one with the highest score
    highest_scoring_docs: dict[str, "Document"] = {}
    for doc in documents:
        score = doc.score if doc.score is not None else -inf
        best = highest_scoring_docs.get(doc.id)

        if best is None or score > (best.score if best.score is not None else -inf):
            highest_scoring_docs[doc.id] = doc

    return list(highest_scoring_docs.values())


@overload
def _parse_dict_from_json(
    text: str, expected_keys: list[str] | None = ..., raise_on_failure: Literal[True] = ...
) -> dict[str, Any]: ...
@overload
def _parse_dict_from_json(
    text: str, expected_keys: list[str] | None = ..., raise_on_failure: Literal[False] = ...
) -> dict[str, Any] | None: ...
@overload
def _parse_dict_from_json(
    text: str, expected_keys: list[str] | None = ..., raise_on_failure: bool = ...
) -> dict[str, Any] | None: ...
def _parse_dict_from_json(
    text: str, expected_keys: list[str] | None = None, raise_on_failure: bool = True
) -> dict[str, Any] | None:
    """
    Parses a JSON string containing a dictionary.

    :param text: The string to parse.
    :param expected_keys: A list of keys that must be present in the parsed dictionary.
    :param raise_on_failure: If True, raises an exception on failure. If False, logs a warning and returns None.

    :return: The parsed dictionary, or None if parsing fails and raise_on_failure is False.
    :raises json.JSONDecodeError: If the text is not valid JSON and raise_on_failure is True.
    :raises ValueError: If the parsed object is not a dictionary or has missing expected keys,
        and `raise_on_failure` is True.
    """
    cleaned_text = text.strip()

    try:
        parsed_json = json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        if raise_on_failure:
            raise e
        logger.warning("Failed to parse JSON from text: {text}. Error: {error}", text=text, error=e)
        return None

    if not isinstance(parsed_json, dict):
        if raise_on_failure:
            raise ValueError(f"Expected a JSON object containing a dictionary but got {type(parsed_json).__name__}")
        logger.warning(
            "Expected a JSON object containing a dictionary but got {type}. Returning None",
            type=type(parsed_json).__name__,
        )
        return None

    if not expected_keys:
        return parsed_json

    missing_keys = [key for key in expected_keys if key not in parsed_json]
    if missing_keys:
        if raise_on_failure:
            raise ValueError(f"Missing expected keys in JSON: {missing_keys}. Got keys: {list(parsed_json.keys())}")
        logger.warning(
            "Missing expected keys in JSON: {missing_keys}. Got keys: {keys}",
            missing_keys=missing_keys,
            keys=list(parsed_json.keys()),
        )
        return None

    return parsed_json
