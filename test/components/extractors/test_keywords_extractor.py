import pytest

from haystack import ComponentError, DeserializationError, Document
from haystack.components.extractors import KeywordsExtractor, KeywordsExtractorBackend


@pytest.mark.unit
def test_keyword_extractor_initiate():
    _ = KeywordsExtractor(backend=KeywordsExtractorBackend.YAKE)

    _ = KeywordsExtractor(backend="yake")

    _ = KeywordsExtractor(backend=KeywordsExtractorBackend.KEYBERT)
    _ = KeywordsExtractor(backend="keybert")

    with pytest.raises(ComponentError, match=r"Invalid backend"):
        KeywordsExtractor(backend="random_backend")


@pytest.mark.unit
@pytest.mark.parametrize(
    "backend,type,initialized",
    [("yake", KeywordsExtractorBackend.YAKE, True), ("keybert", KeywordsExtractorBackend.KEYBERT, False)],
)
def test_keyword_extractor_init_with_backend_kwargs(backend, type, initialized):
    backend_kwargs = {"param1": "value1", "param2": "value2"}
    extractor = KeywordsExtractor(backend=backend, top_n=5, backend_kwargs=backend_kwargs)
    assert extractor.type == type
    assert extractor.top_n == 5
    assert extractor.initialized == initialized
    assert extractor._backend._backend_kwargs == backend_kwargs


@pytest.mark.unit
@pytest.mark.parametrize("backend", ["yake", "keybert"])
def test_keyword_extractor_init_with_invalid_backend_kwargs(backend):
    backend_kwargs = "invalid_backend_kwargs"

    extractor = KeywordsExtractor(backend=backend, top_n=3, backend_kwargs=backend_kwargs)
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


@pytest.mark.unit
def test_run_without_warm_up():
    """
    Tests that run method raises ComponentError if model is not warmed up
    """
    extractor = KeywordsExtractor(backend="keybert", top_n=5)

    documents = [Document(content="doc1"), Document(content="doc2")]

    error_msg = "KeywordsExtractorBackend.KEYBERT was not initialized - Did you call `warm_up()`?"
    with pytest.raises(ComponentError, match=error_msg):
        extractor.run(documents)
