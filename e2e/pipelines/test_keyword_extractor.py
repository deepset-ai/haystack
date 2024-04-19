import pytest

from haystack import ComponentError, Document, Pipeline
from haystack.components.extractors import KeywordAnnotation, KeywordExtractor, KeywordExtractorBackend


@pytest.fixture
def raw_texts():
    return [
        "My name is Clara and I live in Berkeley, California.",
        "I'm Merlin, the happy pig!",
        "New York State declared a state of emergency after the announcement of the end of the world.",
        "",  # Intentionally empty.
    ]


@pytest.fixture
def yake_annotations():
    return [
        [
            KeywordAnnotation(keyword="California", positions=[41], score=0.030396371632413578),
            KeywordAnnotation(keyword="Berkeley", positions=[31], score=0.08596317751626563),
            KeywordAnnotation(keyword="Clara", positions=[11], score=0.1447773057422032),
            KeywordAnnotation(keyword="live", positions=[23], score=0.29736558256021506),
        ],
        [
            KeywordAnnotation(keyword="Merlin", positions=[4], score=0.08596317751626563),
            KeywordAnnotation(keyword="pig", positions=[22], score=0.15831692877998726),
            KeywordAnnotation(keyword="happy", positions=[16], score=0.29736558256021506),
        ],
        [
            KeywordAnnotation(keyword="State", positions=[9], score=0.06052574043360847),
            KeywordAnnotation(keyword="York", positions=[4], score=0.07261214632111582),
            KeywordAnnotation(keyword="world", positions=[86], score=0.09101163530720666),
            KeywordAnnotation(keyword="declared", positions=[15], score=0.13528014248445303),
            KeywordAnnotation(keyword="emergency", positions=[35], score=0.13528014248445303),
            KeywordAnnotation(keyword="announcement", positions=[55], score=0.13528014248445303),
            KeywordAnnotation(keyword="end", positions=[75], score=0.13528014248445303),
        ],
        [],
    ]


@pytest.fixture
def keybert_annotations():
    return [
        [
            KeywordAnnotation(keyword="clara", positions=[11], score=0.6184),
            KeywordAnnotation(keyword="berkeley", positions=[31], score=0.4935),
            KeywordAnnotation(keyword="california", positions=[41], score=0.4804),
            KeywordAnnotation(keyword="live", positions=[23], score=0.1577),
        ],
        [
            KeywordAnnotation(keyword="merlin", positions=[4], score=0.6313),
            KeywordAnnotation(keyword="pig", positions=[22], score=0.5261),
            KeywordAnnotation(keyword="happy", positions=[16], score=0.3234),
        ],
        [
            KeywordAnnotation(keyword="emergency", positions=[35], score=0.4204),
            KeywordAnnotation(keyword="york", positions=[4], score=0.4204),
            KeywordAnnotation(keyword="state", positions=[9, 26], score=0.3825),
            KeywordAnnotation(keyword="announcement", positions=[55], score=0.3227),
            KeywordAnnotation(keyword="declared", positions=[15], score=0.2564),
            KeywordAnnotation(keyword="world", positions=[86], score=0.1959),
            KeywordAnnotation(keyword="end", positions=[75], score=0.1808),
            KeywordAnnotation(keyword="new", positions=[0], score=0.0479),
        ],
        [],
    ]


def test_keyword_extractor_init():
    extractor = KeywordExtractor(backend=KeywordExtractorBackend.YAKE)

    with pytest.raises(ComponentError, match=r"not initialized"):
        extractor.run(documents=[])

    assert not extractor.initialized
    extractor.warm_up()
    assert extractor.initialized


@pytest.mark.parametrize("n", [1])
def test_yake_keyword_extractor_backend(raw_texts, yake_annotations, n):
    extractor = KeywordExtractor(backend=KeywordExtractorBackend.YAKE, n=n)
    extractor.warm_up()
    _extract_and_check_predictions(extractor, raw_texts, yake_annotations)


@pytest.mark.parametrize("n", [1])
def test_yake_keyword_extractor_in_pipeline(raw_texts, yake_annotations, n):
    pipeline = Pipeline()
    pipeline.add_component(
        name="keyword_extractor", instance=KeywordExtractor(backend=KeywordExtractorBackend.YAKE, n=n)
    )

    outputs = pipeline.run({"keyword_extractor": {"documents": [Document(content=text) for text in raw_texts]}})[
        "keyword_extractor"
    ]["documents"]
    predicted = [KeywordExtractor.get_stored_annotations(doc) for doc in outputs]
    _check_predictions(predicted, yake_annotations)


@pytest.mark.parametrize("keyphrase_ngram_range", [(1, 1)])
def test_keybert_keyword_extractor_in_pipeline(raw_texts, keybert_annotations, keyphrase_ngram_range):
    pipeline = Pipeline()
    pipeline.add_component(
        name="keyword_extractor",
        instance=KeywordExtractor(backend=KeywordExtractorBackend.KEYBERT, keyphrase_ngram_range=keyphrase_ngram_range),
    )

    outputs = pipeline.run({"keyword_extractor": {"documents": [Document(content=text) for text in raw_texts]}})[
        "keyword_extractor"
    ]["documents"]
    predicted = [KeywordExtractor.get_stored_annotations(doc) for doc in outputs]
    _check_predictions(predicted, keybert_annotations)


def _extract_and_check_predictions(extractor, texts, expected):
    docs = [Document(content=text) for text in texts]
    outputs = extractor.run(documents=docs)["documents"]
    assert all(id(a) == id(b) for a, b in zip(docs, outputs))
    predicted = [KeywordExtractor.get_stored_annotations(doc) for doc in outputs]
    _check_predictions(predicted, expected)


def _check_predictions(predicted, expected):
    assert len(predicted) == len(expected)
    for pred, exp in zip(predicted, expected):
        assert len(pred) == len(exp)

        for a, b in zip(pred, exp):
            assert a.keyword == b.keyword
            assert a.positions == b.positions
            assert a.score == b.score
