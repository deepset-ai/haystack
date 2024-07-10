# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from haystack import ComponentError, DeserializationError, Pipeline
from haystack.components.extractors import NamedEntityExtractor, NamedEntityExtractorBackend
from haystack.utils.device import ComponentDevice


def test_named_entity_extractor_backend():
    _ = NamedEntityExtractor(backend=NamedEntityExtractorBackend.HUGGING_FACE, model="dslim/bert-base-NER")

    _ = NamedEntityExtractor(backend="hugging_face", model="dslim/bert-base-NER")

    _ = NamedEntityExtractor(backend=NamedEntityExtractorBackend.SPACY, model="en_core_web_sm")

    _ = NamedEntityExtractor(backend="spacy", model="en_core_web_sm")

    with pytest.raises(ComponentError, match=r"Invalid backend"):
        NamedEntityExtractor(backend="random_backend", model="dslim/bert-base-NER")


def test_named_entity_extractor_serde():
    extractor = NamedEntityExtractor(
        backend=NamedEntityExtractorBackend.HUGGING_FACE,
        model="dslim/bert-base-NER",
        device=ComponentDevice.from_str("cuda:1"),
    )

    serde_data = extractor.to_dict()
    new_extractor = NamedEntityExtractor.from_dict(serde_data)

    assert type(new_extractor._backend) == type(extractor._backend)
    assert new_extractor._backend.model_name == extractor._backend.model_name
    assert new_extractor._backend.device == extractor._backend.device

    with pytest.raises(DeserializationError, match=r"Couldn't deserialize"):
        serde_data["init_parameters"].pop("backend")
        _ = NamedEntityExtractor.from_dict(serde_data)


def test_named_entity_extractor_from_dict_no_default_parameters_hf():
    data = {
        "type": "haystack.components.extractors.named_entity_extractor.NamedEntityExtractor",
        "init_parameters": {"backend": "HUGGING_FACE", "model": "dslim/bert-base-NER"},
    }
    extractor = NamedEntityExtractor.from_dict(data)

    assert extractor._backend.model_name == "dslim/bert-base-NER"
    assert extractor._backend.device == ComponentDevice.resolve_device(None)


# tests for NamedEntityExtractor serialization/deserialization in a pipeline
def test_named_entity_extractor_pipeline_serde(tmp_path):
    extractor = NamedEntityExtractor(backend=NamedEntityExtractorBackend.HUGGING_FACE, model="dslim/bert-base-NER")
    p = Pipeline()
    p.add_component(instance=extractor, name="extractor")

    with open(tmp_path / "test_pipeline.yaml", "w") as f:
        p.dump(f)
    with open(tmp_path / "test_pipeline.yaml", "r") as f:
        q = Pipeline.load(f)

    assert p.to_dict() == q.to_dict(), "Pipeline serialization/deserialization with NamedEntityExtractor failed."


def test_named_entity_extractor_serde_none_device():
    extractor = NamedEntityExtractor(
        backend=NamedEntityExtractorBackend.HUGGING_FACE, model="dslim/bert-base-NER", device=None
    )

    serde_data = extractor.to_dict()
    new_extractor = NamedEntityExtractor.from_dict(serde_data)

    assert type(new_extractor._backend) == type(extractor._backend)
    assert new_extractor._backend.model_name == extractor._backend.model_name
    assert new_extractor._backend.device == extractor._backend.device
