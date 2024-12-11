# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
import random
from typing import List
from unittest.mock import Mock, patch

import pytest
from openai import APIError

from haystack import Document
from haystack.components.embedders.openai_document_embedder import OpenAIDocumentEmbedder
from haystack.utils.auth import Secret


def mock_openai_response(input: List[str], model: str = "text-embedding-ada-002", **kwargs) -> dict:
    dict_response = {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": [random.random() for _ in range(1536)]}
            for i in range(len(input))
        ],
        "model": model,
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }

    return dict_response


class TestOpenAIDocumentEmbedder:
    def test_init_default(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        embedder = OpenAIDocumentEmbedder()
        assert embedder.api_key.resolve_value() == "fake-api-key"
        assert embedder.model == "text-embedding-ada-002"
        assert embedder.organization is None
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"
        assert embedder.client.max_retries == 5
        assert embedder.client.timeout == 30.0

    def test_init_with_parameters(self, monkeypatch):
        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        embedder = OpenAIDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key-2"),
            model="model",
            organization="my-org",
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
            timeout=40.0,
            max_retries=1,
        )
        assert embedder.api_key.resolve_value() == "fake-api-key-2"
        assert embedder.organization == "my-org"
        assert embedder.model == "model"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "
        assert embedder.client.max_retries == 1
        assert embedder.client.timeout == 40.0

    def test_init_with_parameters_and_env_vars(self, monkeypatch):
        monkeypatch.setenv("OPENAI_TIMEOUT", "100")
        monkeypatch.setenv("OPENAI_MAX_RETRIES", "10")
        embedder = OpenAIDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key-2"),
            model="model",
            organization="my-org",
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        assert embedder.api_key.resolve_value() == "fake-api-key-2"
        assert embedder.organization == "my-org"
        assert embedder.model == "model"
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.meta_fields_to_embed == ["test_field"]
        assert embedder.embedding_separator == " | "
        assert embedder.client.max_retries == 10
        assert embedder.client.timeout == 100.0

    def test_init_fail_wo_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="None of the .* environment variables are set"):
            OpenAIDocumentEmbedder()

    def test_to_dict(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake-api-key")
        component = OpenAIDocumentEmbedder()
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.openai_document_embedder.OpenAIDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["OPENAI_API_KEY"], "strict": True, "type": "env_var"},
                "api_base_url": None,
                "model": "text-embedding-ada-002",
                "dimensions": None,
                "organization": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "meta_fields_to_embed": [],
                "embedding_separator": "\n",
            },
        }

    def test_to_dict_with_custom_init_parameters(self, monkeypatch):
        monkeypatch.setenv("ENV_VAR", "fake-api-key")
        component = OpenAIDocumentEmbedder(
            api_key=Secret.from_env_var("ENV_VAR", strict=False),
            model="model",
            organization="my-org",
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            meta_fields_to_embed=["test_field"],
            embedding_separator=" | ",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.openai_document_embedder.OpenAIDocumentEmbedder",
            "init_parameters": {
                "api_key": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "api_base_url": None,
                "model": "model",
                "dimensions": None,
                "organization": "my-org",
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "meta_fields_to_embed": ["test_field"],
                "embedding_separator": " | ",
            },
        }

    def test_prepare_texts_to_embed_w_metadata(self):
        documents = [
            Document(id=f"{i}", content=f"document number {i}:\ncontent", meta={"meta_field": f"meta_value {i}"})
            for i in range(5)
        ]

        embedder = OpenAIDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key"), meta_fields_to_embed=["meta_field"], embedding_separator=" | "
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        # note that newline is replaced by space
        assert prepared_texts == {
            "0": "meta_value 0 | document number 0: content",
            "1": "meta_value 1 | document number 1: content",
            "2": "meta_value 2 | document number 2: content",
            "3": "meta_value 3 | document number 3: content",
            "4": "meta_value 4 | document number 4: content",
        }

    def test_prepare_texts_to_embed_w_suffix(self):
        documents = [Document(id=f"{i}", content=f"document number {i}") for i in range(5)]

        embedder = OpenAIDocumentEmbedder(
            api_key=Secret.from_token("fake-api-key"), prefix="my_prefix ", suffix=" my_suffix"
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == {
            "0": "my_prefix document number 0 my_suffix",
            "1": "my_prefix document number 1 my_suffix",
            "2": "my_prefix document number 2 my_suffix",
            "3": "my_prefix document number 3 my_suffix",
            "4": "my_prefix document number 4 my_suffix",
        }

    def test_run_wrong_input_format(self):
        embedder = OpenAIDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))

        # wrong formats
        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="OpenAIDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=string_input)

        with pytest.raises(TypeError, match="OpenAIDocumentEmbedder expects a list of Documents as input"):
            embedder.run(documents=list_integers_input)

    def test_run_on_empty_list(self):
        embedder = OpenAIDocumentEmbedder(api_key=Secret.from_token("fake-api-key"))

        empty_list_input = []
        result = embedder.run(documents=empty_list_input)

        assert result["documents"] is not None
        assert not result["documents"]  # empty list

    def test_embed_batch_handles_exceptions_gracefully(self, caplog):
        embedder = OpenAIDocumentEmbedder(api_key=Secret.from_token("fake_api_key"))
        fake_texts_to_embed = {"1": "text1", "2": "text2"}
        with patch.object(
            embedder.client.embeddings,
            "create",
            side_effect=APIError(message="Mocked error", request=Mock(), body=None),
        ):
            embedder._embed_batch(texts_to_embed=fake_texts_to_embed, batch_size=2)

        assert len(caplog.records) == 1
        assert "Failed embedding of documents 1, 2 caused by Mocked error" in caplog.records[0].msg

    @pytest.mark.skipif(os.environ.get("OPENAI_API_KEY", "") == "", reason="OPENAI_API_KEY is not set")
    @pytest.mark.integration
    def test_run(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        model = "text-embedding-ada-002"

        embedder = OpenAIDocumentEmbedder(model=model, meta_fields_to_embed=["topic"], embedding_separator=" | ")

        result = embedder.run(documents=docs)
        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 1536
            assert all(isinstance(x, float) for x in doc.embedding)

        assert (
            "text" in result["meta"]["model"] and "ada" in result["meta"]["model"]
        ), "The model name does not contain 'text' and 'ada'"

        assert result["meta"]["usage"] == {"prompt_tokens": 15, "total_tokens": 15}, "Usage information does not match"
