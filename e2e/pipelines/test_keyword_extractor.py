import pytest

from haystack import Document, Pipeline
from haystack.components.extractors import (
    HighlightedText,
    KeywordsExtractor,
    KeywordsExtractorBackend,
    KeyWordsSelection,
)


@pytest.fixture
def raw_texts():
    return [
        (
            "Supervised learning is the machine learning task of learning a"
            "function that maps an input to an output based on example input-output pairs. "
            "It infers a function from labeled training data consisting of a set of training examples. "
            "In supervised learning, each example is a pair consisting of an input object "
            "(typically a vector) and a desired output value (also called the supervisory signal). "
            "A supervised learning algorithm analyzes the training data and produces an inferred "
            "function, which can be used for mapping new examples. An optimal scenario will allow for "
            "the algorithm to correctly determine the class labels for unseen instances. "
            "This requires the learning algorithm to generalize from the training data to unseen "
            "situations in a 'reasonable' way (see inductive bias)."
        ),
        (
            "Haystack is an open-source framework for building production-ready LLM applications "
            "retrieval-augmented generative pipelines and state-of-the-art search systems "
            "that work intelligently over large document collections. "
            "Learn more about Haystack and how it works."
        ),
        "",  # Intentionally empty.
    ]


@pytest.fixture
def yake_keywords_five():
    return [
        {
            "keywords": [
                KeyWordsSelection(entity="machine learning task", score=0.026288458458349206),
                KeyWordsSelection(entity="afunction that maps", score=0.047289948689021914),
                KeyWordsSelection(entity="Supervised learning", score=0.07732438104871593),
                KeyWordsSelection(entity="learning", score=0.0775411459456495),
                KeyWordsSelection(entity="maps an input", score=0.08445317698651879),
            ],
            "highlight": HighlightedText(
                text=(
                    "<kw>Supervised learning</kw> is the <kw>machine learning task</kw> of <kw>learning</kw> "
                    "<kw>afunction that maps</kw> an input to an output based on example input-output pairs. "
                    "It infers a function from labeled training data consisting of a set of training examples. "
                    "In <kw>supervised learning</kw>, each example is a pair consisting of an input object "
                    "(typically a vector) and a desired output value (also called the supervisory signal). "
                    "A <kw>supervised learning</kw> algorithm analyzes the training data and produces an "
                    "inferred function, which can be used for mapping new examples. An optimal scenario will "
                    "allow for the algorithm to correctly determine the class labels for unseen instances. "
                    "This requires the <kw>learning</kw> algorithm to generalize from the training data to "
                    "unseen situations in a 'reasonable' way (see inductive bias)."
                )
            ),
        },
        {
            "keywords": [
                KeyWordsSelection(entity="building production-ready LLM", score=0.006534094663438077),
                KeyWordsSelection(entity="production-ready LLM applications", score=0.006534094663438078),
                KeyWordsSelection(entity="LLM applications retrieval-augmented", score=0.006534094663438078),
                KeyWordsSelection(entity="large document collections", score=0.00944562772813063),
                KeyWordsSelection(entity="applications retrieval-augmented generative", score=0.01560948497642163),
            ],
            "highlight": HighlightedText(
                text=(
                    "Haystack is an open-source framework for <kw>building production-ready LLM</kw> "
                    "<kw>applications retrieval-augmented generative</kw> pipelines and state-of-the-art "
                    "search systems that work intelligently over <kw>large document collections</kw>. "
                    "Learn more about Haystack and how it works."
                )
            ),
        },
        {"keywords": [], "highlight": HighlightedText(text="")},
    ]


@pytest.fixture
def yake_keywords_one():
    return [
        {
            "keywords": [KeyWordsSelection(entity="machine learning task", score=0.026288458458349206)],
            "highlight": HighlightedText(
                text=(
                    "Supervised learning is the <kw>machine learning task</kw> of learning afunction that maps an "
                    "input to an output based on example input-output pairs. It infers a function from labeled "
                    "training data consisting of a set of training examples. In supervised learning, each example "
                    "is a pair consisting of an input object (typically a vector) and a desired output value "
                    "(also called the supervisory signal). A supervised learning algorithm analyzes the training "
                    "data and produces an inferred function, which can be used for mapping new examples. "
                    "An optimal scenario will allow for the algorithm to correctly determine the class labels "
                    "for unseen instances. This requires the learning algorithm to generalize from the training "
                    "data to unseen situations in a 'reasonable' way (see inductive bias)."
                )
            ),
        },
        {
            "keywords": [KeyWordsSelection(entity="building production-ready LLM", score=0.006534094663438077)],
            "highlight": HighlightedText(
                text=(
                    "Haystack is an open-source framework for <kw>building production-ready LLM</kw> "
                    "applications retrieval-augmented generative pipelines and state-of-the-art search systems "
                    "that work intelligently over large document collections. Learn more about Haystack and how it works."
                )
            ),
        },
        {"keywords": [], "highlight": HighlightedText(text="")},
    ]


@pytest.mark.parametrize("top_n, expected", [(5, "yake_keywords_five"), (1, "yake_keywords_one")])
def test_keywords_extractor_yake_backend(raw_texts, top_n, expected, request: pytest.FixtureRequest):
    expected = request.getfixturevalue(expected)
    extractor = KeywordsExtractor(backend=KeywordsExtractorBackend.YAKE, top_n=top_n)
    extractor.warm_up()
    _extract_and_check_predictions(extractor, raw_texts, expected)


@pytest.mark.parametrize("top_n, expected", [(5, "yake_keywords_five"), (1, "yake_keywords_one")])
def test_keyword_extractor_extractor_in_pipeline(raw_texts, top_n, expected, request: pytest.FixtureRequest):
    expected = request.getfixturevalue(expected)
    pipeline = Pipeline()
    pipeline.add_component(
        name="keyword_extractor", instance=KeywordsExtractor(backend=KeywordsExtractorBackend.YAKE, top_n=top_n)
    )

    outputs = pipeline.run({"keyword_extractor": {"documents": [Document(content=text) for text in raw_texts]}})[
        "keyword_extractor"
    ]["documents"]
    predicted = [KeywordsExtractor.get_stored_annotations(doc) for doc in outputs]

    assert predicted == expected


def _extract_and_check_predictions(extractor, texts, expected):
    docs = [Document(content=text) for text in texts]
    outputs = extractor.run(documents=docs)["documents"]
    assert all(id(a) == id(b) for a, b in zip(docs, outputs))
    predicted = [KeywordsExtractor.get_stored_annotations(doc) for doc in outputs]

    assert predicted == expected
