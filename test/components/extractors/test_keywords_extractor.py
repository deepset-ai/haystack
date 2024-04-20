import pytest

from haystack import ComponentError, DeserializationError, Document
from haystack.components.extractors import KeywordsExtractor, KeywordsExtractorBackend
from haystack.utils import ComponentDevice, Secret


@pytest.fixture
def backend_fixture_model_name_or_path():
    return {
        "keyphrase_ngram_range": (1, 2),
        "stop_words": None,
        "model_name_or_path": "a_sentence_transformer_model",
        "default_prompt_name": "a_prompt_name",
        "token": "1234",
    }


@pytest.fixture
def backend_fixture_model():
    return {
        "keyphrase_ngram_range": (1, 2),
        "stop_words": None,
        "model": "a_sentence_transformer_model",
        "default_prompt_name": "a_prompt_name",
        "token": "1234",
    }


@pytest.mark.unit
def test_keyword_extractor_initiate():
    _ = KeywordsExtractor(backend=KeywordsExtractorBackend.YAKE)

    _ = KeywordsExtractor(backend="yake")

    _ = KeywordsExtractor(backend=KeywordsExtractorBackend.SENTENCETRANSFORMER)
    _ = KeywordsExtractor(backend="sentence_transformer")

    with pytest.raises(ComponentError, match=r"Invalid backend"):
        KeywordsExtractor(backend="random_backend")


@pytest.mark.unit
@pytest.mark.parametrize(
    "backend,type,initialized,expected_backend_kwargs",
    [
        ("yake", KeywordsExtractorBackend.YAKE, True, {"candidates": "value1", "stop_words": "value2"}),
        (
            "sentence_transformer",
            KeywordsExtractorBackend.SENTENCETRANSFORMER,
            False,
            {"candidates": "value1", "stop_words": "value2", "keyphrase_ngram_range": (3, 3)},
        ),
    ],
)
def test_keyword_extractor_init_with_backend_kwargs(backend, type, initialized, expected_backend_kwargs):
    backend_kwargs = {"candidates": "value1", "stop_words": "value2"}
    extractor = KeywordsExtractor(backend=backend, top_n=5, backend_kwargs=backend_kwargs)
    assert extractor.type == type
    assert extractor.top_n == 5
    assert extractor.initialized == initialized
    assert extractor._backend._backend_kwargs == expected_backend_kwargs


@pytest.mark.unit
@pytest.mark.parametrize(
    "backend,expected", [("yake", {}), ("sentence_transformer", {"keyphrase_ngram_range": (3, 3)})]
)
def test_keyword_extractor_init_with_invalid_backend_kwargs(backend, expected):
    backend_kwargs = "invalid_backend_kwargs"

    extractor = KeywordsExtractor(backend=backend, top_n=3, backend_kwargs=backend_kwargs)
    assert extractor._backend._backend_kwargs == expected


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
@pytest.mark.parametrize("kwargs, expected", [({"keyphrase_ngram_range": (1, 2)}, (1, 2)), ({}, (3, 3))])
def test_keybert_model_keyphrase_init(kwargs, expected):
    "Tests that the keyphrase_ngram_range attrib are initialized correctly when _KeyBertBackend init is called"
    extractor = KeywordsExtractor(backend=KeywordsExtractorBackend.SENTENCETRANSFORMER, top_n=3, backend_kwargs=kwargs)

    assert extractor._backend._backend_kwargs == {"keyphrase_ngram_range": expected}
    assert extractor._backend._keybert_model._model_kwargs == {}
    assert extractor._backend._keybert_model._model_name == "all-MiniLM-L6-v2"
    assert extractor._backend._keybert_model._device == ComponentDevice.resolve_device(None)
    assert extractor._backend._keybert_model._token == Secret.from_env_var("HF_API_TOKEN", strict=False)


@pytest.mark.unit
@pytest.mark.parametrize("backend_kwargs", ["backend_fixture_model_name_or_path", "backend_fixture_model"])
def test_keybert_model_backend_init(backend_kwargs, request: pytest.FixtureRequest):
    "test that the _backend_kwargs and _sentence_transformer_kwargs attrib are initialized correctly when _KeyBertBackend init is called"
    backend_kwargs = request.getfixturevalue(backend_kwargs)
    extractor = KeywordsExtractor(
        backend=KeywordsExtractorBackend.SENTENCETRANSFORMER, top_n=3, backend_kwargs=backend_kwargs
    )

    assert extractor._backend._backend_kwargs == {"keyphrase_ngram_range": (1, 2), "stop_words": None}
    assert extractor._backend._keybert_model._model_kwargs == {"default_prompt_name": "a_prompt_name"}
    assert extractor._backend._keybert_model._model_name == "a_sentence_transformer_model"
    assert extractor._backend._keybert_model._device == ComponentDevice.resolve_device(None)
    assert extractor._backend._keybert_model._token == "1234"


@pytest.mark.unit
def test_run_without_warm_up():
    """
    Tests that run method raises ComponentError if model is not warmed up
    """
    extractor = KeywordsExtractor(backend="sentence_transformer", top_n=5)

    documents = [Document(content="doc1"), Document(content="doc2")]

    error_msg = "Did you call `warm_up()`?"
    with pytest.raises(ComponentError) as exc_info:
        extractor.run(documents)

    # Check if the error message contains a specific string
    assert error_msg in str(exc_info.value)
