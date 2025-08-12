# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# Note: We do not test the Spacy backend in this module.
# Spacy is not installed in the test environment to keep the CI fast.
# We test the Spacy backend in e2e/pipelines/test_named_entity_extractor.py.

from unittest.mock import patch

import pytest

from haystack import ComponentError, DeserializationError, Document, Pipeline
from haystack.components.extractors import NamedEntityAnnotation, NamedEntityExtractor, NamedEntityExtractorBackend
from haystack.utils.auth import Secret
from haystack.utils.device import ComponentDevice


def test_named_entity_extractor_backend():
    _ = NamedEntityExtractor(backend=NamedEntityExtractorBackend.HUGGING_FACE, model="dslim/bert-base-NER")

    # private model
    _ = NamedEntityExtractor(backend=NamedEntityExtractorBackend.HUGGING_FACE, model="deepset/bert-base-NER")

    _ = NamedEntityExtractor(backend="hugging_face", model="dslim/bert-base-NER")

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


def test_to_dict_default(monkeypatch):
    monkeypatch.delenv("HF_API_TOKEN", raising=False)

    component = NamedEntityExtractor(
        backend=NamedEntityExtractorBackend.HUGGING_FACE,
        model="dslim/bert-base-NER",
        device=ComponentDevice.from_str("mps"),
    )
    data = component.to_dict()

    assert data == {
        "type": "haystack.components.extractors.named_entity_extractor.NamedEntityExtractor",
        "init_parameters": {
            "backend": "HUGGING_FACE",
            "model": "dslim/bert-base-NER",
            "device": {"type": "single", "device": "mps"},
            "pipeline_kwargs": {"model": "dslim/bert-base-NER", "device": "mps", "task": "ner"},
            "token": {"type": "env_var", "env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False},
        },
    }


def test_to_dict_with_parameters():
    component = NamedEntityExtractor(
        backend=NamedEntityExtractorBackend.HUGGING_FACE,
        model="dslim/bert-base-NER",
        device=ComponentDevice.from_str("mps"),
        pipeline_kwargs={"model_kwargs": {"load_in_4bit": True}},
        token=Secret.from_env_var("ENV_VAR", strict=False),
    )
    data = component.to_dict()

    assert data == {
        "type": "haystack.components.extractors.named_entity_extractor.NamedEntityExtractor",
        "init_parameters": {
            "backend": "HUGGING_FACE",
            "model": "dslim/bert-base-NER",
            "device": {"type": "single", "device": "mps"},
            "pipeline_kwargs": {
                "model": "dslim/bert-base-NER",
                "device": "mps",
                "task": "ner",
                "model_kwargs": {"load_in_4bit": True},
            },
            "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
        },
    }


def test_named_entity_extractor_from_dict_no_default_parameters_hf(monkeypatch):
    monkeypatch.delenv("HF_API_TOKEN", raising=False)

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


def test_named_entity_extractor_run():
    """Test the NamedEntityExtractor.run method with mocked model interaction."""
    documents = [Document(content="My name is Clara and I live in Berkeley, California.")]

    expected_annotations = [
        [
            NamedEntityAnnotation(entity="PER", start=11, end=16, score=0.95),
            NamedEntityAnnotation(entity="LOC", start=31, end=39, score=0.88),
            NamedEntityAnnotation(entity="LOC", start=41, end=51, score=0.92),
        ]
    ]

    extractor = NamedEntityExtractor(backend=NamedEntityExtractorBackend.HUGGING_FACE, model="dslim/bert-base-NER")

    with patch.object(extractor._backend, "annotate", return_value=expected_annotations) as mock_annotate:
        extractor._backend.pipeline = "mocked_pipeline"
        extractor._warmed_up = True

        result = extractor.run(documents=documents, batch_size=2)

        mock_annotate.assert_called_once_with(["My name is Clara and I live in Berkeley, California."], batch_size=2)

        assert "documents" in result
        assert len(result["documents"]) == 1

        assert isinstance(result["documents"][0], Document)
        assert result["documents"][0].content == documents[0].content
        assert "named_entities" in result["documents"][0].meta
        assert result["documents"][0].meta["named_entities"] == expected_annotations[0]
        assert "named_entities" not in documents[0].meta


def test_named_entity_extractor_run_not_warmed_up():
    """Test that run method raises error when not warmed up."""
    extractor = NamedEntityExtractor(backend=NamedEntityExtractorBackend.HUGGING_FACE, model="dslim/bert-base-NER")

    documents = [Document(content="Test document")]

    with pytest.raises(RuntimeError, match="The component NamedEntityExtractor was not warmed up"):
        extractor.run(documents=documents)
