# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import MagicMock, patch

import torch
import numpy as np
import pytest

from haystack.components.embedders.sentence_transformers_text_embedder import SentenceTransformersTextEmbedder
from haystack.utils import ComponentDevice, Secret


class TestSentenceTransformersTextEmbedder:
    def test_init_default(self):
        embedder = SentenceTransformersTextEmbedder(model="model")
        assert embedder.model == "model"
        assert embedder.device == ComponentDevice.resolve_device(None)
        assert embedder.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.normalize_embeddings is False
        assert embedder.trust_remote_code is False
        assert embedder.truncate_dim is None
        assert embedder.precision == "float32"

    def test_init_with_parameters(self):
        embedder = SentenceTransformersTextEmbedder(
            model="model",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_token("fake-api-token"),
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
            trust_remote_code=True,
            truncate_dim=256,
            precision="int8",
        )
        assert embedder.model == "model"
        assert embedder.device == ComponentDevice.from_str("cuda:0")
        assert embedder.token == Secret.from_token("fake-api-token")
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.normalize_embeddings is True
        assert embedder.trust_remote_code is True
        assert embedder.truncate_dim == 256
        assert embedder.precision == "int8"

    def test_to_dict(self):
        component = SentenceTransformersTextEmbedder(model="model", device=ComponentDevice.from_str("cpu"))
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
                "trust_remote_code": False,
                "truncate_dim": None,
                "model_kwargs": None,
                "tokenizer_kwargs": None,
                "precision": "float32",
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = SentenceTransformersTextEmbedder(
            model="model",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_env_var("ENV_VAR", strict=False),
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            normalize_embeddings=True,
            trust_remote_code=True,
            truncate_dim=256,
            model_kwargs={"torch_dtype": torch.float32},
            tokenizer_kwargs={"model_max_length": 512},
            precision="int8",
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": ComponentDevice.from_str("cuda:0").to_dict(),
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "normalize_embeddings": True,
                "trust_remote_code": True,
                "truncate_dim": 256,
                "model_kwargs": {"torch_dtype": "torch.float32"},
                "tokenizer_kwargs": {"model_max_length": 512},
                "precision": "int8",
            },
        }

    def test_to_dict_not_serialize_token(self):
        component = SentenceTransformersTextEmbedder(model="model", token=Secret.from_token("fake-api-token"))
        with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
            component.to_dict()

    def test_from_dict(self):
        data = {
            "type": "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
                "trust_remote_code": False,
                "truncate_dim": None,
                "model_kwargs": {"torch_dtype": "torch.float32"},
                "tokenizer_kwargs": {"model_max_length": 512},
                "precision": "float32",
            },
        }
        component = SentenceTransformersTextEmbedder.from_dict(data)
        assert component.model == "model"
        assert component.device == ComponentDevice.from_str("cpu")
        assert component.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.batch_size == 32
        assert component.progress_bar is True
        assert component.normalize_embeddings is False
        assert component.trust_remote_code is False
        assert component.truncate_dim is None
        assert component.model_kwargs == {"torch_dtype": torch.float32}
        assert component.tokenizer_kwargs == {"model_max_length": 512}
        assert component.precision == "float32"

    def test_from_dict_no_default_parameters(self):
        data = {
            "type": "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
            "init_parameters": {},
        }
        component = SentenceTransformersTextEmbedder.from_dict(data)
        assert component.model == "sentence-transformers/all-mpnet-base-v2"
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.batch_size == 32
        assert component.progress_bar is True
        assert component.normalize_embeddings is False
        assert component.trust_remote_code is False
        assert component.truncate_dim is None
        assert component.precision == "float32"

    def test_from_dict_none_device(self):
        data = {
            "type": "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
                "trust_remote_code": False,
                "truncate_dim": 256,
                "precision": "int8",
            },
        }
        component = SentenceTransformersTextEmbedder.from_dict(data)
        assert component.model == "model"
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.batch_size == 32
        assert component.progress_bar is True
        assert component.normalize_embeddings is False
        assert component.trust_remote_code is False
        assert component.truncate_dim == 256
        assert component.precision == "int8"

    @patch(
        "haystack.components.embedders.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        embedder = SentenceTransformersTextEmbedder(model="model", token=None, device=ComponentDevice.from_str("cpu"))
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="model",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            truncate_dim=None,
            model_kwargs=None,
            tokenizer_kwargs=None,
        )

    @patch(
        "haystack.components.embedders.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_warmup_doesnt_reload(self, mocked_factory):
        embedder = SentenceTransformersTextEmbedder(model="model")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()

    def test_run(self):
        embedder = SentenceTransformersTextEmbedder(model="model")
        embedder.embedding_backend = MagicMock()
        embedder.embedding_backend.embed = lambda x, **kwargs: np.random.rand(len(x), 16).tolist()

        text = "a nice text to embed"

        result = embedder.run(text=text)
        embedding = result["embedding"]

        assert isinstance(embedding, list)
        assert all(isinstance(el, float) for el in embedding)

    def test_run_wrong_input_format(self):
        embedder = SentenceTransformersTextEmbedder(model="model")
        embedder.embedding_backend = MagicMock()

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="SentenceTransformersTextEmbedder expects a string as input"):
            embedder.run(text=list_integers_input)

    @pytest.mark.integration
    def test_run_trunc(self):
        """
        sentence-transformers/paraphrase-albert-small-v2 maps sentences & paragraphs to a 768 dimensional dense vector space
        """
        checkpoint = "sentence-transformers/paraphrase-albert-small-v2"
        text = "a nice text to embed"

        embedder_def = SentenceTransformersTextEmbedder(model=checkpoint)
        embedder_def.warm_up()
        result_def = embedder_def.run(text=text)
        embedding_def = result_def["embedding"]

        embedder_trunc = SentenceTransformersTextEmbedder(model=checkpoint, truncate_dim=128)
        embedder_trunc.warm_up()
        result_trunc = embedder_trunc.run(text=text)
        embedding_trunc = result_trunc["embedding"]

        assert len(embedding_def) == 768
        assert len(embedding_trunc) == 128

    @pytest.mark.integration
    def test_run_quantization(self):
        """
        sentence-transformers/paraphrase-albert-small-v2 maps sentences & paragraphs to a 768 dimensional dense vector space
        """
        checkpoint = "sentence-transformers/paraphrase-albert-small-v2"
        text = "a nice text to embed"

        embedder_def = SentenceTransformersTextEmbedder(model=checkpoint, precision="int8")
        embedder_def.warm_up()
        result_def = embedder_def.run(text=text)
        embedding_def = result_def["embedding"]

        assert len(embedding_def) == 768
        assert all(isinstance(el, int) for el in embedding_def)
