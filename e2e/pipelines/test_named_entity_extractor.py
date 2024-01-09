import pytest

from haystack import Document, Pipeline, ComponentError
from haystack.components.extractors import NamedEntityAnnotation, NamedEntityExtractor, NamedEntityExtractorBackend


@pytest.fixture
def raw_texts():
    return [
        "My name is Clara and I live in Berkeley, California.",
        "I'm Merlin, the happy pig!",
        "New York State declared a state of emergency after the announcement of the end of the world.",
        "",  # Intentionally empty.
    ]


@pytest.fixture
def hf_annotations():
    return [
        [
            NamedEntityAnnotation(entity="PER", start=11, end=16),
            NamedEntityAnnotation(entity="LOC", start=31, end=39),
            NamedEntityAnnotation(entity="LOC", start=41, end=51),
        ],
        [NamedEntityAnnotation(entity="PER", start=4, end=10)],
        [NamedEntityAnnotation(entity="LOC", start=0, end=14)],
        [],
    ]


@pytest.fixture
def spacy_annotations():
    return [
        [
            NamedEntityAnnotation(entity="PERSON", start=11, end=16),
            NamedEntityAnnotation(entity="GPE", start=31, end=39),
            NamedEntityAnnotation(entity="GPE", start=41, end=51),
        ],
        [NamedEntityAnnotation(entity="PERSON", start=4, end=10)],
        [NamedEntityAnnotation(entity="GPE", start=0, end=14)],
        [],
    ]


def test_ner_extractor_init():
    extractor = NamedEntityExtractor(
        backend=NamedEntityExtractorBackend.HUGGING_FACE, model_name_or_path="dslim/bert-base-NER", device_id=-1
    )

    with pytest.raises(ComponentError, match=r"not initialized"):
        extractor.run(documents=[])

    assert not extractor.initialized
    extractor.warm_up()
    assert extractor.initialized


@pytest.mark.parametrize("batch_size", [1, 3])
def test_ner_extractor_hf_backend(raw_texts, hf_annotations, batch_size):
    extractor = NamedEntityExtractor(
        backend=NamedEntityExtractorBackend.HUGGING_FACE, model_name_or_path="dslim/bert-base-NER"
    )
    extractor.warm_up()

    _extract_and_check_predictions(extractor, raw_texts, hf_annotations, batch_size)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_ner_extractor_spacy_backend(raw_texts, spacy_annotations, batch_size):
    extractor = NamedEntityExtractor(backend=NamedEntityExtractorBackend.SPACY, model_name_or_path="en_core_web_trf")
    extractor.warm_up()

    _extract_and_check_predictions(extractor, raw_texts, spacy_annotations, batch_size)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_ner_extractor_in_pipeline(raw_texts, hf_annotations, batch_size):
    pipeline = Pipeline()
    pipeline.add_component(
        name="ner_extractor",
        instance=NamedEntityExtractor(
            backend=NamedEntityExtractorBackend.HUGGING_FACE, model_name_or_path="dslim/bert-base-NER"
        ),
    )

    outputs = pipeline.run(
        {"ner_extractor": {"documents": [Document(content=text) for text in raw_texts], "batch_size": batch_size}}
    )["ner_extractor"]["documents"]
    predicted = [NamedEntityExtractor.get_stored_annotations(doc) for doc in outputs]
    _check_predictions(predicted, hf_annotations)


def _extract_and_check_predictions(extractor, texts, expected, batch_size):
    docs = [Document(content=text) for text in texts]
    outputs = extractor.run(documents=docs, batch_size=batch_size)["documents"]
    assert all(id(a) == id(b) for a, b in zip(docs, outputs))
    predicted = [NamedEntityExtractor.get_stored_annotations(doc) for doc in outputs]

    _check_predictions(predicted, expected)


def _check_predictions(predicted, expected):
    assert len(predicted) == len(expected)
    for pred, exp in zip(predicted, expected):
        assert len(pred) == len(exp)

        for a, b in zip(pred, exp):
            assert a.entity == b.entity
            assert a.start == b.start
            assert a.end == b.end
