from typing import Optional

from canals.component import InputSocket


def test_is_not_optional():
    s = InputSocket("test_name", int)
    assert s.is_optional is False


def test_is_optional():
    s = InputSocket("test_name", Optional[int])
    assert s.is_optional
