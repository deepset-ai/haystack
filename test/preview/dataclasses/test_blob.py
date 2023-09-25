from haystack.preview.dataclasses import Blob

import pytest


@pytest.mark.unit
def test_from_file_path(tmp_path, request):
    test_bytes = "Hello, world!\n".encode()
    test_path = tmp_path / request.node.name
    with open(test_path, "wb") as fd:
        assert fd.write(test_bytes)

    f = Blob.from_file_path(test_path)
    assert f.data == test_bytes


def test_save(tmp_path, request):
    test_str = "Hello, world!\n"
    test_path = tmp_path / request.node.name

    Blob(test_str.encode()).save(test_path)
    with open(test_path, "rb") as fd:
        assert fd.read().decode() == test_str
