# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from haystack.dataclasses import ByteStream

LinkFormat = Literal["markdown", "plain", "none"]
SUPPORTED_LINK_FORMATS = ("markdown", "plain", "none")


@dataclass(frozen=True)
class LinkAnnotation:
    """
    A hyperlink annotation rectangle and its target URI.
    """

    rect: tuple[float, float, float, float]
    uri: str


def validate_link_format(link_format: LinkFormat) -> LinkFormat:
    """
    Validate a converter hyperlink output format.

    :param link_format:
        The requested link output format.
    :returns:
        The validated link output format.
    """
    if link_format not in SUPPORTED_LINK_FORMATS:
        msg = f"Unknown link format '{link_format}'. Supported formats are: 'markdown', 'plain', 'none'"
        raise ValueError(msg)
    return link_format


def normalize_rect(rect: Any) -> tuple[float, float, float, float]:
    """
    Normalize a PDF rectangle to ``(left, bottom, right, top)``.
    """
    left, bottom, right, top = (float(value) for value in rect)
    return min(left, right), min(bottom, top), max(left, right), max(bottom, top)


def format_link_text(text: str, uri: str, link_format: LinkFormat) -> str:
    """
    Format a text fragment with a hyperlink while preserving surrounding whitespace.
    """
    if link_format == "none":
        return text

    prefix_length = len(text) - len(text.lstrip())
    suffix_start = len(text.rstrip())
    prefix = text[:prefix_length]
    suffix = text[suffix_start:]
    label = text[prefix_length:suffix_start]
    if not label:
        return text

    if link_format == "markdown":
        return f"{prefix}[{label}]({uri}){suffix}"
    return f"{prefix}{label} ({uri}){suffix}"


def link_for_bbox(bbox: tuple[float, float, float, float], links: list[LinkAnnotation]) -> LinkAnnotation | None:
    """
    Return the hyperlink annotation whose rectangle best matches the given bounding box.
    """
    if not links:
        return None

    left, bottom, right, top = normalize_rect(bbox)
    center_x = (left + right) / 2
    center_y = (bottom + top) / 2

    best_link = None
    best_score = 0.0
    for link in links:
        link_left, link_bottom, link_right, link_top = link.rect
        center_inside = link_left <= center_x <= link_right and link_bottom <= center_y <= link_top

        overlap_width = max(0.0, min(right, link_right) - max(left, link_left))
        overlap_height = max(0.0, min(top, link_top) - max(bottom, link_bottom))
        overlap_area = overlap_width * overlap_height
        if not center_inside and overlap_area == 0:
            continue

        score = overlap_area
        if center_inside:
            score += 1.0
        if score > best_score:
            best_score = score
            best_link = link

    return best_link


def get_bytestream_from_source(source: str | Path | ByteStream, guess_mime_type: bool = False) -> ByteStream:
    """
    Creates a ByteStream object from a source.

    :param source:
        A source to convert to a ByteStream. Can be a string (path to a file), a Path object, or a ByteStream.
    :param guess_mime_type:
        Whether to guess the mime type from the file.
    :return:
        A ByteStream object.
    """

    if isinstance(source, ByteStream):
        return source
    if isinstance(source, (str, Path)):
        bs = ByteStream.from_file_path(Path(source), guess_mime_type=guess_mime_type)
        bs.meta["file_path"] = str(source)
        return bs
    raise ValueError(f"Unsupported source type {type(source)}")


def normalize_metadata(meta: dict[str, Any] | list[dict[str, Any]] | None, sources_count: int) -> list[dict[str, Any]]:
    """
    Normalize the metadata input for a converter.

    Given all the possible value of the meta input for a converter (None, dictionary or list of dicts),
    makes sure to return a list of dictionaries of the correct length for the converter to use.

    :param meta: the meta input of the converter, as-is
    :param sources_count: the number of sources the converter received
    :returns: a list of dictionaries of the make length as the sources list
    """
    if meta is None:
        return [{}] * sources_count
    if isinstance(meta, dict):
        return [meta] * sources_count
    if isinstance(meta, list):
        if sources_count != len(meta):
            raise ValueError("The length of the metadata list must match the number of sources.")
        return meta
    raise ValueError("meta must be either None, a dictionary or a list of dictionaries.")
