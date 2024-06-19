# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack.dataclasses import ByteStream


def get_bytestream_from_source(source: Union[str, Path, ByteStream]) -> ByteStream:
    """
    Creates a ByteStream object from a source.

    :param source:
        A source to convert to a ByteStream. Can be a string (path to a file), a Path object, or a ByteStream.
    :return:
        A ByteStream object.
    """

    if isinstance(source, ByteStream):
        return source
    if isinstance(source, (str, Path)):
        bs = ByteStream.from_file_path(Path(source))
        bs.meta["file_path"] = str(source)
        return bs
    raise ValueError(f"Unsupported source type {type(source)}")


def normalize_metadata(
    meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]], sources_count: int
) -> List[Dict[str, Any]]:
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
