from pathlib import Path
from typing import Union

from haystack.dataclasses import ByteStream


def get_bytestream_from_source(source: Union[str, Path, ByteStream]) -> ByteStream:
    """
    Creates a ByteStream object from a source.
    :param source: A source to convert to a ByteStream. Can be a string (path to a file), a Path object, or a ByteStream.
    :return: A ByteStream object.
    """

    if isinstance(source, ByteStream):
        return source
    if isinstance(source, (str, Path)):
        bs = ByteStream.from_file_path(Path(source))
        bs.meta["file_path"] = str(source)
        return bs
    raise ValueError(f"Unsupported source type {type(source)}")
