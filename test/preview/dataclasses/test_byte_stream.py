import io

from haystack.preview.dataclasses import ByteStream

import pytest


@pytest.mark.unit
def test_from_file_path(tmp_path, request):
    test_bytes = "Hello, world!\n".encode()
    test_path = tmp_path / request.node.name
    with open(test_path, "wb") as fd:
        assert fd.write(test_bytes)

    b = ByteStream.from_file_path(test_path)
    assert b.data == test_bytes


@pytest.mark.unit
def test_from_string():
    test_string = "Hello, world!"
    b = ByteStream.from_string(test_string)
    assert b.data.decode() == test_string


@pytest.mark.unit
def test_to_file(tmp_path, request):
    test_str = "Hello, world!\n"
    test_path = tmp_path / request.node.name

    ByteStream(test_str.encode()).to_file(test_path)
    with open(test_path, "rb") as fd:
        assert fd.read().decode() == test_str
