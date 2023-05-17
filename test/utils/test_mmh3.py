import pytest

from haystack.mmh3 import hash128


@pytest.mark.unit
def test_mmh3():
    content = "This is the document text" * 100
    hashed_content = hash128(content)
    assert hashed_content == 305042678480070366459393623793278501577
