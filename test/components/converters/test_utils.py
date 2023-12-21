import pytest
from haystack.components.converters.utils import normalize_metadata


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
