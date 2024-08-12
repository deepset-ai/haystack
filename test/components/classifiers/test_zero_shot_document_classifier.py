# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from unittest.mock import patch

from haystack import Document
from haystack.components.classifiers import TransformersZeroShotDocumentClassifier
from haystack.utils import ComponentDevice, Secret


class TestTransformersZeroShotDocumentClassifier:
    def test_to_dict(self):
        component = TransformersZeroShotDocumentClassifier(
            model="cross-encoder/nli-distilroberta-base", labels=["positive", "negative"]
        )
        component_dict = component.to_dict()
        assert component_dict == {
            "type": "haystack.components.classifiers.zero_shot_document_classifier.TransformersZeroShotDocumentClassifier",
            "init_parameters": {
                "model": "cross-encoder/nli-distilroberta-base",
                "labels": ["positive", "negative"],
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "huggingface_pipeline_kwargs": {
                    "model": "cross-encoder/nli-distilroberta-base",
                    "device": ComponentDevice.resolve_device(None).to_hf(),
                    "task": "zero-shot-classification",
                },
            },
        }

    def test_from_dict(self, monkeypatch):
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        data = {
            "type": "haystack.components.classifiers.zero_shot_document_classifier.TransformersZeroShotDocumentClassifier",
            "init_parameters": {
                "model": "cross-encoder/nli-distilroberta-base",
                "labels": ["positive", "negative"],
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "huggingface_pipeline_kwargs": {
                    "model": "cross-encoder/nli-distilroberta-base",
                    "device": ComponentDevice.resolve_device(None).to_hf(),
                    "task": "zero-shot-classification",
                },
            },
        }

        component = TransformersZeroShotDocumentClassifier.from_dict(data)
        assert component.labels == ["positive", "negative"]
        assert component.pipeline is None
        assert component.token == Secret.from_dict(
            {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"}
        )
        assert component.huggingface_pipeline_kwargs == {
            "model": "cross-encoder/nli-distilroberta-base",
            "device": ComponentDevice.resolve_device(None).to_hf(),
            "task": "zero-shot-classification",
            "token": None,
        }

    def test_from_dict_no_default_parameters(self, monkeypatch):
        monkeypatch.delenv("HF_API_TOKEN", raising=False)
        data = {
            "type": "haystack.components.classifiers.zero_shot_document_classifier.TransformersZeroShotDocumentClassifier",
            "init_parameters": {"model": "cross-encoder/nli-distilroberta-base", "labels": ["positive", "negative"]},
        }
        component = TransformersZeroShotDocumentClassifier.from_dict(data)
        assert component.labels == ["positive", "negative"]
        assert component.pipeline is None
        assert component.token == Secret.from_dict(
            {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"}
        )
        assert component.huggingface_pipeline_kwargs == {
            "model": "cross-encoder/nli-distilroberta-base",
            "device": ComponentDevice.resolve_device(None).to_hf(),
            "task": "zero-shot-classification",
            "token": None,
        }

    @patch("haystack.components.classifiers.zero_shot_document_classifier.pipeline")
    def test_warm_up(self, hf_pipeline_mock):
        component = TransformersZeroShotDocumentClassifier(
            model="cross-encoder/nli-distilroberta-base", labels=["positive", "negative"]
        )
        component.warm_up()
        assert component.pipeline is not None

    def test_run_fails_without_warm_up(self):
        component = TransformersZeroShotDocumentClassifier(
            model="cross-encoder/nli-distilroberta-base", labels=["positive", "negative"]
        )
        positive_documents = [Document(content="That's good. I like it.")]
        with pytest.raises(RuntimeError):
            component.run(documents=positive_documents)

    @patch("haystack.components.classifiers.zero_shot_document_classifier.pipeline")
    def test_run_fails_with_non_document_input(self, hf_pipeline_mock):
        hf_pipeline_mock.return_value = " "
        component = TransformersZeroShotDocumentClassifier(
            model="cross-encoder/nli-distilroberta-base", labels=["positive", "negative"]
        )
        component.warm_up()
        text_list = ["That's good. I like it.", "That's bad. I don't like it."]
        with pytest.raises(TypeError):
            component.run(documents=text_list)

    @patch("haystack.components.classifiers.zero_shot_document_classifier.pipeline")
    def test_run_unit(self, hf_pipeline_mock):
        hf_pipeline_mock.return_value = [
            {"sequence": "That's good. I like it.", "labels": ["positive", "negative"], "scores": [0.99, 0.01]},
            {"sequence": "That's bad. I don't like it.", "labels": ["negative", "positive"], "scores": [0.99, 0.01]},
        ]
        component = TransformersZeroShotDocumentClassifier(
            model="cross-encoder/nli-distilroberta-base", labels=["positive", "negative"]
        )
        component.pipeline = hf_pipeline_mock
        positive_document = Document(content="That's good. I like it.")
        negative_document = Document(content="That's bad. I don't like it.")
        result = component.run(documents=[positive_document, negative_document])
        assert component.pipeline is not None
        assert result["documents"][0].to_dict()["classification"]["label"] == "positive"
        assert result["documents"][1].to_dict()["classification"]["label"] == "negative"

    @pytest.mark.integration
    def test_run(self):
        component = TransformersZeroShotDocumentClassifier(
            model="cross-encoder/nli-distilroberta-base", labels=["positive", "negative"]
        )
        component.warm_up()
        positive_document = Document(content="That's good. I like it. " * 1000)
        negative_document = Document(content="That's bad. I don't like it.")
        result = component.run(documents=[positive_document, negative_document])
        assert component.pipeline is not None
        assert result["documents"][0].to_dict()["classification"]["label"] == "positive"
        assert result["documents"][1].to_dict()["classification"]["label"] == "negative"
