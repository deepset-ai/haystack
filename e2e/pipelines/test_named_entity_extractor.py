# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Note: We only test the Spacy backend in this module, which is executed in the e2e environment.
# We don't test Spacy in test/components/extractors/test_named_entity_extractor.py, which is executed in the
# test environment. Spacy is not installed in the test environment to keep the CI fast.

import os
import sys

import pytest
from thinc.api import NumpyOps, get_current_ops, set_current_ops

from haystack import Document, Pipeline
from haystack.components.extractors import (
    NamedEntityAnnotation,
    NamedEntityExtractor,
    NamedEntityExtractorBackend,
)


@pytest.fixture
def raw_texts() -> list:
    return [
        "My name is Clara and I live in Berkeley, California.",
        "I'm Merlin, the happy pig!",
        "New York State declared a state of emergency after the announcement of the end of the world.",
        "",  # Intentionally empty.
    ]


@pytest.fixture
def hf_annotations() -> list:
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
def spacy_annotations() -> list:
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


def test_ner_extractor_init(del_hf_env_vars) -> None:
    extractor = NamedEntityExtractor(backend=NamedEntityExtractorBackend.HUGGING_FACE, model="dslim/bert-base-NER")
    extractor.warm_up()
    assert extractor.initialized


@pytest.mark.parametrize("batch_size", [1, 3])
def test_ner_extractor_hf_backend(raw_texts, hf_annotations, batch_size, del_hf_env_vars) -> None:
    extractor = NamedEntityExtractor(backend=NamedEntityExtractorBackend.HUGGING_FACE, model="dslim/bert-base-NER")
    extractor.warm_up()

    _extract_and_check_predictions(extractor, raw_texts, hf_annotations, batch_size)


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.skipif(
    not os.environ.get("HF_API_TOKEN", None) and not os.environ.get("HF_TOKEN", None),
    reason="Export an env var called HF_API_TOKEN or HF_TOKEN containing the Hugging Face token to run this test.",
)
def test_ner_extractor_hf_backend_private_models(raw_texts, hf_annotations, batch_size) -> None:
    extractor = NamedEntityExtractor(backend=NamedEntityExtractorBackend.HUGGING_FACE, model="deepset/bert-base-NER")
    extractor.warm_up()

    _extract_and_check_predictions(extractor, raw_texts, hf_annotations, batch_size)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_ner_extractor_spacy_backend(raw_texts, spacy_annotations, batch_size) -> None:
    extractor = NamedEntityExtractor(backend=NamedEntityExtractorBackend.SPACY, model="en_core_web_trf")
    extractor.warm_up()

    _extract_and_check_predictions(extractor, raw_texts, spacy_annotations, batch_size)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_ner_extractor_in_pipeline(raw_texts, hf_annotations, batch_size, del_hf_env_vars) -> None:
    pipeline = Pipeline()
    pipeline.add_component(
        name="ner_extractor",
        instance=NamedEntityExtractor(backend=NamedEntityExtractorBackend.HUGGING_FACE, model="dslim/bert-base-NER"),
    )

    outputs = pipeline.run(
        {"ner_extractor": {"documents": [Document(content=text) for text in raw_texts], "batch_size": batch_size}}
    )["ner_extractor"]["documents"]
    predicted = [NamedEntityExtractor.get_stored_annotations(doc) for doc in outputs]
    _check_predictions(predicted, hf_annotations)


def _extract_and_check_predictions(extractor, texts, expected, batch_size) -> None:
    docs = [Document(content=text) for text in texts]
    outputs = extractor.run(documents=docs, batch_size=batch_size)["documents"]
    for original_doc, output_doc in zip(docs, outputs):
        # we don't modify documents in place
        assert original_doc is not output_doc

        # apart from meta, the documents should be identical
        output_doc_dict = output_doc.to_dict(flatten=False)
        output_doc_dict.pop("meta", None)
        assert original_doc.to_dict() == output_doc_dict

    predicted = [NamedEntityExtractor.get_stored_annotations(doc) for doc in outputs]

    _check_predictions(predicted, expected)


def _check_predictions(predicted, expected) -> None:
    assert len(predicted) == len(expected)
    for pred, exp in zip(predicted, expected):
        assert len(pred) == len(exp)

        for a, b in zip(pred, exp):
            assert a.entity == b.entity
            assert a.start == b.start
            assert a.end == b.end

class TestNamedEntityExtractorDeviceRestoration:
    def test_spacy_backend_restores_device_state(self):
        """
        Verify that NamedEntityExtractor (spaCy) restores the previous Thinc Ops state
        after the component runs.
        """
        # 1. Setup a custom state
        custom_ops = NumpyOps()
        setattr(custom_ops, "owner", "test_marker")
        set_current_ops(custom_ops)

        try:
            # 2. Initialize and run (triggering the context manager)
            extractor = NamedEntityExtractor(backend="spacy", model="en_core_web_sm")

            # Since _SpacyBackend is private, we access it via getattr to avoid IDE warnings
            backend = getattr(extractor, "_backend")
            select_device_method = getattr(backend, "_select_device")

            with select_device_method():
                # Inside the context, the state might change
                pass

            # 3. Verify state is restored
            final_ops = get_current_ops()
            assert getattr(final_ops, "owner", None) == "test_marker"

        finally:
            # Clean up global state
            set_current_ops(NumpyOps())
