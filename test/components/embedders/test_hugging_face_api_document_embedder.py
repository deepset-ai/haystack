# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import MagicMock, patch

import random
import pytest
from huggingface_hub.utils import RepositoryNotFoundError

from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
from haystack.dataclasses import Document
from haystack.utils.auth import Secret
from haystack.utils.hf import HFEmbeddingAPIType


@pytest.fixture
def mock_check_valid_model():
    with patch(
        "haystack.components.embedders.hugging_face_api_document_embedder.check_valid_model",
        MagicMock(return_value=None),
    ) as mock:
        yield mock


def mock_embedding_generation(json, **kwargs):
    response = str([[random.random() for _ in range(384)] for _ in range(len(json["inputs"]))]).encode()
    return response


class TestHuggingFaceAPIDocumentEmbedder:
    def test_init_invalid_api_type(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIDocumentEmbedder(api_type="invalid_api_type", api_params={})

    def test_init_serverless(self, mock_check_valid_model):
        model = "BAAI/bge-small-en-v1.5"
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": model}
        )

        assert embedder.api_type == HFEmbeddingAPIType.SERVERLESS_INFERENCE_API
        assert embedder.api_params == {"model": model}
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.truncate
        assert not embedder.normalize
        assert embedder.batch_size == 32
        assert embedder.progress_bar
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_serverless_invalid_model(self, mock_check_valid_model):
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id")
        with pytest.raises(RepositoryNotFoundError):
            HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "invalid_model_id"}
            )

    def test_init_serverless_no_model(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"param": "irrelevant"}
            )

    def test_init_tei(self):
        url = "https://some_model.com"

        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE, api_params={"url": url}
        )

        assert embedder.api_type == HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE
        assert embedder.api_params == {"url": url}
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.truncate
        assert not embedder.normalize
        assert embedder.batch_size == 32
        assert embedder.progress_bar
        assert embedder.meta_fields_to_embed == []
        assert embedder.embedding_separator == "\n"

    def test_init_tei_invalid_url(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE, api_params={"url": "invalid_url"}
            )

    def test_init_tei_no_url(self):
        with pytest.raises(ValueError):
            HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE, api_params={"param": "irrelevant"}
            )

    def test_to_dict(self, mock_check_valid_model):
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "BAAI/bge-small-en-v1.5"},
            prefix="prefix",
            suffix="suffix",
            truncate=False,
            normalize=True,
            batch_size=128,
            progress_bar=False,
            meta_fields_to_embed=["meta_field"],
            embedding_separator=" ",
        )

        data = embedder.to_dict()

        assert data == {
            "type": "haystack.components.embedders.hugging_face_api_document_embedder.HuggingFaceAPIDocumentEmbedder",
            "init_parameters": {
                "api_type": "serverless_inference_api",
                "api_params": {"model": "BAAI/bge-small-en-v1.5"},
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "prefix": "prefix",
                "suffix": "suffix",
                "truncate": False,
                "normalize": True,
                "batch_size": 128,
                "progress_bar": False,
                "meta_fields_to_embed": ["meta_field"],
                "embedding_separator": " ",
            },
        }

    def test_from_dict(self, mock_check_valid_model):
        data = {
            "type": "haystack.components.embedders.hugging_face_api_document_embedder.HuggingFaceAPIDocumentEmbedder",
            "init_parameters": {
                "api_type": HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                "api_params": {"model": "BAAI/bge-small-en-v1.5"},
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "prefix": "prefix",
                "suffix": "suffix",
                "truncate": False,
                "normalize": True,
                "batch_size": 128,
                "progress_bar": False,
                "meta_fields_to_embed": ["meta_field"],
                "embedding_separator": " ",
            },
        }

        embedder = HuggingFaceAPIDocumentEmbedder.from_dict(data)

        assert embedder.api_type == HFEmbeddingAPIType.SERVERLESS_INFERENCE_API
        assert embedder.api_params == {"model": "BAAI/bge-small-en-v1.5"}
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert not embedder.truncate
        assert embedder.normalize
        assert embedder.batch_size == 128
        assert not embedder.progress_bar
        assert embedder.meta_fields_to_embed == ["meta_field"]
        assert embedder.embedding_separator == " "

    def test_prepare_texts_to_embed_w_metadata(self):
        documents = [
            Document(content=f"document number {i}: content", meta={"meta_field": f"meta_value {i}"}) for i in range(5)
        ]

        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "https://some_model.com"},
            token=Secret.from_token("fake-api-token"),
            meta_fields_to_embed=["meta_field"],
            embedding_separator=" | ",
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "meta_value 0 | document number 0: content",
            "meta_value 1 | document number 1: content",
            "meta_value 2 | document number 2: content",
            "meta_value 3 | document number 3: content",
            "meta_value 4 | document number 4: content",
        ]

    def test_prepare_texts_to_embed_w_suffix(self, mock_check_valid_model):
        documents = [Document(content=f"document number {i}") for i in range(5)]

        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE,
            api_params={"url": "https://some_model.com"},
            token=Secret.from_token("fake-api-token"),
            prefix="my_prefix ",
            suffix=" my_suffix",
        )

        prepared_texts = embedder._prepare_texts_to_embed(documents)

        assert prepared_texts == [
            "my_prefix document number 0 my_suffix",
            "my_prefix document number 1 my_suffix",
            "my_prefix document number 2 my_suffix",
            "my_prefix document number 3 my_suffix",
            "my_prefix document number 4 my_suffix",
        ]

    def test_embed_batch(self, mock_check_valid_model):
        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]

        with patch("huggingface_hub.InferenceClient.post") as mock_embedding_patch:
            mock_embedding_patch.side_effect = mock_embedding_generation

            embedder = HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "BAAI/bge-small-en-v1.5"},
                token=Secret.from_token("fake-api-token"),
            )
            embeddings = embedder._embed_batch(texts_to_embed=texts, batch_size=2)

            assert mock_embedding_patch.call_count == 3

        assert isinstance(embeddings, list)
        assert len(embeddings) == len(texts)
        for embedding in embeddings:
            assert isinstance(embedding, list)
            assert len(embedding) == 384
            assert all(isinstance(x, float) for x in embedding)

    def test_run_wrong_input_format(self, mock_check_valid_model):
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API, api_params={"model": "BAAI/bge-small-en-v1.5"}
        )

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError):
            embedder.run(text=list_integers_input)

    def test_run_on_empty_list(self, mock_check_valid_model):
        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "BAAI/bge-small-en-v1.5"},
            token=Secret.from_token("fake-api-token"),
        )

        empty_list_input = []
        result = embedder.run(documents=empty_list_input)

        assert result["documents"] is not None
        assert not result["documents"]  # empty list

    def test_run(self, mock_check_valid_model):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        with patch("huggingface_hub.InferenceClient.post") as mock_embedding_patch:
            mock_embedding_patch.side_effect = mock_embedding_generation

            embedder = HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "BAAI/bge-small-en-v1.5"},
                token=Secret.from_token("fake-api-token"),
                prefix="prefix ",
                suffix=" suffix",
                meta_fields_to_embed=["topic"],
                embedding_separator=" | ",
            )

            result = embedder.run(documents=docs)

            mock_embedding_patch.assert_called_once_with(
                json={
                    "inputs": [
                        "prefix Cuisine | I love cheese suffix",
                        "prefix ML | A transformer is a deep learning architecture suffix",
                    ],
                    "truncate": True,
                    "normalize": False,
                },
                task="feature-extraction",
            )
        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 384
            assert all(isinstance(x, float) for x in doc.embedding)

    def test_run_custom_batch_size(self, mock_check_valid_model):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        with patch("huggingface_hub.InferenceClient.post") as mock_embedding_patch:
            mock_embedding_patch.side_effect = mock_embedding_generation

            embedder = HuggingFaceAPIDocumentEmbedder(
                api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
                api_params={"model": "BAAI/bge-small-en-v1.5"},
                token=Secret.from_token("fake-api-token"),
                prefix="prefix ",
                suffix=" suffix",
                meta_fields_to_embed=["topic"],
                embedding_separator=" | ",
                batch_size=1,
            )

            result = embedder.run(documents=docs)

            assert mock_embedding_patch.call_count == 2

        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 384
            assert all(isinstance(x, float) for x in doc.embedding)

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("HF_API_TOKEN", None),
        reason="Export an env var called HF_API_TOKEN containing the Hugging Face token to run this test.",
    )
    def test_live_run_serverless(self):
        docs = [
            Document(content="I love cheese", meta={"topic": "Cuisine"}),
            Document(content="A transformer is a deep learning architecture", meta={"topic": "ML"}),
        ]

        embedder = HuggingFaceAPIDocumentEmbedder(
            api_type=HFEmbeddingAPIType.SERVERLESS_INFERENCE_API,
            api_params={"model": "sentence-transformers/all-MiniLM-L6-v2"},
            meta_fields_to_embed=["topic"],
            embedding_separator=" | ",
        )
        result = embedder.run(documents=docs)
        documents_with_embeddings = result["documents"]

        assert isinstance(documents_with_embeddings, list)
        assert len(documents_with_embeddings) == len(docs)
        for doc in documents_with_embeddings:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 384
            assert all(isinstance(x, float) for x in doc.embedding)
