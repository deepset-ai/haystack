# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from huggingface_hub.utils import RepositoryNotFoundError

from haystack.components.embedders.hugging_face_tei_text_embedder import HuggingFaceTEITextEmbedder
from haystack.utils.auth import Secret


@pytest.fixture
def mock_check_valid_model():
    with patch(
        "haystack.components.embedders.hugging_face_tei_text_embedder.check_valid_model", MagicMock(return_value=None)
    ) as mock:
        yield mock


def mock_embedding_generation(json, **kwargs):
    response = str(np.array([np.random.rand(384) for i in range(len(json["inputs"]))]).tolist()).encode()
    return response


class TestHuggingFaceTEITextEmbedder:
    def test_init_default(self, monkeypatch, mock_check_valid_model):
        monkeypatch.setenv("HF_API_TOKEN", "fake-api-token")
        embedder = HuggingFaceTEITextEmbedder()

        assert embedder.model == "BAAI/bge-small-en-v1.5"
        assert embedder.url is None
        assert embedder.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.truncate is True
        assert embedder.normalize is False

    def test_init_with_parameters(self, mock_check_valid_model):
        embedder = HuggingFaceTEITextEmbedder(
            model="sentence-transformers/all-mpnet-base-v2",
            url="https://some_embedding_model.com",
            token=Secret.from_token("fake-api-token"),
            prefix="prefix",
            suffix="suffix",
            truncate=False,
            normalize=True,
        )

        assert embedder.model == "sentence-transformers/all-mpnet-base-v2"
        assert embedder.url == "https://some_embedding_model.com"
        assert embedder.token == Secret.from_token("fake-api-token")
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.truncate is False
        assert embedder.normalize is True

    def test_initialize_with_invalid_url(self, mock_check_valid_model):
        with pytest.raises(ValueError):
            HuggingFaceTEITextEmbedder(model="sentence-transformers/all-mpnet-base-v2", url="invalid_url")

    def test_initialize_with_url_but_invalid_model(self, mock_check_valid_model):
        # When custom TEI endpoint is used via URL, model must be provided and valid HuggingFace Hub model id
        mock_check_valid_model.side_effect = RepositoryNotFoundError("Invalid model id")
        with pytest.raises(RepositoryNotFoundError):
            HuggingFaceTEITextEmbedder(model="invalid_model_id", url="https://some_embedding_model.com")

    def test_to_dict(self, mock_check_valid_model):
        component = HuggingFaceTEITextEmbedder()
        data = component.to_dict()

        assert data == {
            "type": "haystack.components.embedders.hugging_face_tei_text_embedder.HuggingFaceTEITextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"},
                "model": "BAAI/bge-small-en-v1.5",
                "url": None,
                "prefix": "",
                "suffix": "",
                "truncate": True,
                "normalize": False,
            },
        }

    def test_from_dict(self, mock_check_valid_model):
        data = {
            "type": "haystack.components.embedders.hugging_face_tei_text_embedder.HuggingFaceTEITextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN"], "strict": False, "type": "env_var"},
                "model": "BAAI/bge-small-en-v1.5",
                "url": None,
                "prefix": "",
                "suffix": "",
                "truncate": True,
                "normalize": False,
            },
        }

        embedder = HuggingFaceTEITextEmbedder.from_dict(data)

        assert embedder.model == "BAAI/bge-small-en-v1.5"
        assert embedder.url is None
        assert embedder.token == Secret.from_env_var("HF_API_TOKEN", strict=False)
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.truncate is True
        assert embedder.normalize is False

    def test_to_dict_with_custom_init_parameters(self, mock_check_valid_model):
        component = HuggingFaceTEITextEmbedder(
            model="sentence-transformers/all-mpnet-base-v2",
            url="https://some_embedding_model.com",
            token=Secret.from_env_var("ENV_VAR", strict=False),
            prefix="prefix",
            suffix="suffix",
            truncate=False,
            normalize=True,
        )

        data = component.to_dict()

        assert data == {
            "type": "haystack.components.embedders.hugging_face_tei_text_embedder.HuggingFaceTEITextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "sentence-transformers/all-mpnet-base-v2",
                "url": "https://some_embedding_model.com",
                "prefix": "prefix",
                "suffix": "suffix",
                "truncate": False,
                "normalize": True,
            },
        }

    def test_from_dict_with_custom_init_parameters(self, mock_check_valid_model):
        data = {
            "type": "haystack.components.embedders.hugging_face_tei_text_embedder.HuggingFaceTEITextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "sentence-transformers/all-mpnet-base-v2",
                "url": "https://some_embedding_model.com",
                "prefix": "prefix",
                "suffix": "suffix",
                "truncate": False,
                "normalize": True,
            },
        }

        embedder = HuggingFaceTEITextEmbedder.from_dict(data)

        assert embedder.model == "sentence-transformers/all-mpnet-base-v2"
        assert embedder.url == "https://some_embedding_model.com"
        assert embedder.token == Secret.from_env_var("ENV_VAR", strict=False)
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.truncate is False
        assert embedder.normalize is True

    def test_run(self, mock_check_valid_model):
        with patch("huggingface_hub.InferenceClient.post") as mock_embedding_patch:
            mock_embedding_patch.side_effect = mock_embedding_generation

            embedder = HuggingFaceTEITextEmbedder(
                model="BAAI/bge-small-en-v1.5",
                token=Secret.from_token("fake-api-token"),
                prefix="prefix ",
                suffix=" suffix",
            )

            result = embedder.run(text="The food was delicious")

            mock_embedding_patch.assert_called_once_with(
                json={"inputs": ["prefix The food was delicious suffix"], "truncate": True, "normalize": False},
                task="feature-extraction",
            )

        assert len(result["embedding"]) == 384
        assert all(isinstance(x, float) for x in result["embedding"])

    @pytest.mark.flaky(reruns=5, reruns_delay=5)
    @pytest.mark.integration
    @pytest.mark.skipif(
        not os.environ.get("HF_API_TOKEN", None),
        reason="Export an env var called HF_API_TOKEN containing the Hugging Face token to run this test.",
    )
    def test_run_inference_api_endpoint(self):
        embedder = HuggingFaceTEITextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        result = embedder.run(text="The food was delicious")

        assert len(result["embedding"]) == 384
        assert all(isinstance(x, float) for x in result["embedding"])

    def test_run_wrong_input_format(self, mock_check_valid_model):
        embedder = HuggingFaceTEITextEmbedder(
            model="BAAI/bge-small-en-v1.5",
            url="https://some_embedding_model.com",
            token=Secret.from_token("fake-api-token"),
        )

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="HuggingFaceTEITextEmbedder expects a string as an input"):
            embedder.run(text=list_integers_input)
