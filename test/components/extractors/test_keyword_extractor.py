import pytest

from haystack import ComponentError, DeserializationError
from haystack.components.extractors import KeywordExtractor, KeywordExtractorBackend


@pytest.mark.unit
def test_keyword_extractor_backend():
    _ = KeywordExtractor(backend=KeywordExtractorBackend.YAKE)

    _ = KeywordExtractor(backend="yake", n=3)

    _ = KeywordExtractor(backend=KeywordExtractorBackend.KEYBERT, model="all-mpnet-base-v2")

    _ = KeywordExtractor(backend=KeywordExtractorBackend.KEYBERT)

    _ = KeywordExtractor(backend="keybert", model="all-MiniLM-L6-v2", use_maxsum=True)

    with pytest.raises(ComponentError, match=r"Invalid backend"):
        KeywordExtractor(backend="random_backend", model="all-MiniLM-L6-v2")

    with pytest.raises(ComponentError, match=r"Invalid parameter"):
        KeywordExtractor(backend="yake", wrong_param=3)

    with pytest.raises(ComponentError, match=r"Invalid parameter"):
        KeywordExtractor(backend="keybert", wrong_param="ABC")


@pytest.mark.unit
def test_keyword_extractor_serde():
    extractor = KeywordExtractor(backend=KeywordExtractorBackend.KEYBERT, model="all-MiniLM-L6-v2")
    serde_data = extractor.to_dict()
    new_extractor = KeywordExtractor.from_dict(serde_data)

    assert new_extractor._backend._type == extractor._backend._type
    assert new_extractor._backend.config == extractor._backend.config

    with pytest.raises(DeserializationError, match=r"Couldn't deserialize"):
        serde_data["init_parameters"].pop("backend")
        _ = KeywordExtractor.from_dict(serde_data)
