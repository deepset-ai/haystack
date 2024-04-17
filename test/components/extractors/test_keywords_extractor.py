import pytest

from haystack import ComponentError, DeserializationError
from haystack.components.extractors import KeywordsExtractor, KeywordsExtractorBackend


@pytest.mark.unit
def test_keyword_extractor_initiate():
    _ = KeywordsExtractor(backend=KeywordsExtractorBackend.YAKE)

    _ = KeywordsExtractor(backend="yake")

    with pytest.raises(ComponentError, match=r"not ready yet"):
        KeywordsExtractor(backend=KeywordsExtractorBackend.KEYBERT)
        KeywordsExtractor(backend="keybert")

    with pytest.raises(ComponentError, match=r"Invalid backend"):
        KeywordsExtractor(backend="random_backend")


@pytest.mark.unit
def test_keyword_extractor_init_with_backend_kwargs():
    backend_kwargs = {"param1": "value1", "param2": "value2"}
    extractor = KeywordsExtractor(backend="yake", top_n=5, backend_kwargs=backend_kwargs)
    assert extractor.type == KeywordsExtractorBackend.YAKE
    assert extractor.top_n == 5
    assert extractor.initialized == True


@pytest.mark.unit
def test_keyword_extractor_init_with_invalid_backend_kwargs():
    backend_kwargs = "invalid_backend_kwargs"

    extractor = KeywordsExtractor(backend="yake", top_n=3, backend_kwargs=backend_kwargs)
    assert extractor._backend._backend_kwargs == {}


@pytest.mark.unit
def test_keywords_extractor_methods():
    extractor = KeywordsExtractor(backend=KeywordsExtractorBackend.YAKE)

    serde_data = extractor.to_dict()
    serde_data_2 = dict(extractor)
    assert serde_data == serde_data_2
    new_extractor = KeywordsExtractor.from_dict(serde_data)

    assert type(new_extractor._backend) == type(extractor._backend)

    with pytest.raises(DeserializationError, match=r"Couldn't deserialize"):
        serde_data["init_parameters"].pop("backend")
        _ = KeywordsExtractor.from_dict(serde_data)
