import pytest

from haystack.preview import Document, DeserializationError
from haystack.preview.utils.marshalling import marshal_type, unmarshal_type


TYPE_STRING_PAIRS = [(int, "int"), (Document, "haystack.preview.dataclasses.document.Document")]


@pytest.mark.unit
@pytest.mark.parametrize("type_,string", TYPE_STRING_PAIRS)
def test_marshal_type(type_, string):
    assert marshal_type(type_) == string


@pytest.mark.unit
@pytest.mark.parametrize("type_,string", TYPE_STRING_PAIRS)
def test_unmarshal_type(type_, string):
    assert unmarshal_type(string) == type_


@pytest.mark.unit
def test_unmarshal_type_missing_builtin():
    with pytest.raises(DeserializationError, match="Could not locate builtin called 'something'"):
        unmarshal_type("something")


@pytest.mark.unit
def test_unmarshal_type_missing_module():
    with pytest.raises(DeserializationError, match="Could not locate the module 'something'"):
        unmarshal_type("something.int")


@pytest.mark.unit
def test_unmarshal_type_missing_type():
    with pytest.raises(DeserializationError, match="Could not locate the type 'Documentttt'"):
        unmarshal_type("haystack.preview.dataclasses.document.Documentttt")
