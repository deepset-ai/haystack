# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.components.converters.utils import get_bytestream_from_source, normalize_metadata
from haystack.dataclasses import ByteStream


def test_normalize_metadata_None():
    assert normalize_metadata(None, sources_count=1) == [{}]
    assert normalize_metadata(None, sources_count=3) == [{}, {}, {}]


def test_normalize_metadata_single_dict():
    assert normalize_metadata({"a": 1}, sources_count=1) == [{"a": 1}]
    assert normalize_metadata({"a": 1}, sources_count=3) == [{"a": 1}, {"a": 1}, {"a": 1}]


def test_normalize_metadata_list_of_right_size():
    assert normalize_metadata([{"a": 1}], sources_count=1) == [{"a": 1}]
    assert normalize_metadata([{"a": 1}, {"b": 2}, {"c": 3}], sources_count=3) == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_normalize_metadata_list_of_wrong_size():
    with pytest.raises(ValueError, match="The length of the metadata list must match the number of sources."):
        normalize_metadata([{"a": 1}], sources_count=3)
    with pytest.raises(ValueError, match="The length of the metadata list must match the number of sources."):
        assert normalize_metadata([{"a": 1}, {"b": 2}, {"c": 3}], sources_count=1)


def test_normalize_metadata_other_type():
    with pytest.raises(ValueError, match="meta must be either None, a dictionary or a list of dictionaries."):
        normalize_metadata(({"a": 1},), sources_count=1)


def test_get_bytestream_from_path_object(tmp_path):
    bytes_ = b"hello world"
    source = tmp_path / "test.txt"
    source.write_bytes(bytes_)

    bs = get_bytestream_from_source(source, guess_mime_type=True)

    assert isinstance(bs, ByteStream)
    assert bs.data == bytes_
    assert bs.mime_type == "text/plain"
    assert bs.meta["file_path"].endswith("test.txt")


def test_get_bytestream_from_string_path(tmp_path):
    bytes_ = b"hello world"
    source = tmp_path / "test.txt"
    source.write_bytes(bytes_)

    bs = get_bytestream_from_source(str(source), guess_mime_type=True)

    assert isinstance(bs, ByteStream)
    assert bs.data == bytes_
    assert bs.mime_type == "text/plain"
    assert bs.meta["file_path"].endswith("test.txt")


def test_get_bytestream_from_source_invalid_type():
    with pytest.raises(ValueError, match="Unsupported source type"):
        get_bytestream_from_source(123)


def test_get_bytestream_from_source_bytestream_passthrough():
    bs = ByteStream(data=b"spam", mime_type="text/custom", meta={"spam": "eggs"})
    result = get_bytestream_from_source(bs)
    assert result is bs
