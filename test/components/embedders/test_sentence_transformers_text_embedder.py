# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import random
from unittest.mock import MagicMock, patch

import pytest
import torch

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
        assert embedder.local_files_only is False
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
            local_files_only=True,
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
        assert embedder.local_files_only is True
        assert embedder.truncate_dim == 256
        assert embedder.precision == "int8"

    def test_to_dict(self):
        component = SentenceTransformersTextEmbedder(model="model", device=ComponentDevice.from_str("cpu"))
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",  # noqa: E501
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
                "local_files_only": False,
                "truncate_dim": None,
                "model_kwargs": None,
                "tokenizer_kwargs": None,
                "encode_kwargs": None,
                "config_kwargs": None,
                "precision": "float32",
                "backend": "torch",
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
            local_files_only=True,
            truncate_dim=256,
            model_kwargs={"torch_dtype": torch.float32},
            tokenizer_kwargs={"model_max_length": 512},
            config_kwargs={"use_memory_efficient_attention": False},
            precision="int8",
            encode_kwargs={"task": "clustering"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",  # noqa: E501
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
                "local_files_only": True,
                "truncate_dim": 256,
                "model_kwargs": {"torch_dtype": "torch.float32"},
                "tokenizer_kwargs": {"model_max_length": 512},
                "config_kwargs": {"use_memory_efficient_attention": False},
                "precision": "int8",
                "encode_kwargs": {"task": "clustering"},
                "backend": "torch",
            },
        }

    def test_to_dict_not_serialize_token(self):
        component = SentenceTransformersTextEmbedder(model="model", token=Secret.from_token("fake-api-token"))
        with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
            component.to_dict()

    def test_from_dict(self):
        data = {
            "type": "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",  # noqa: E501
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
                "local_files_only": False,
                "truncate_dim": None,
                "model_kwargs": {"torch_dtype": "torch.float32"},
                "tokenizer_kwargs": {"model_max_length": 512},
                "config_kwargs": {"use_memory_efficient_attention": False},
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
        assert component.local_files_only is False
        assert component.truncate_dim is None
        assert component.model_kwargs == {"torch_dtype": torch.float32}
        assert component.tokenizer_kwargs == {"model_max_length": 512}
        assert component.config_kwargs == {"use_memory_efficient_attention": False}
        assert component.precision == "float32"

    def test_from_dict_no_default_parameters(self):
        data = {
            "type": "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",  # noqa: E501
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
        assert component.local_files_only is False
        assert component.truncate_dim is None
        assert component.precision == "float32"

    def test_from_dict_none_device(self):
        data = {
            "type": "haystack.components.embedders.sentence_transformers_text_embedder.SentenceTransformersTextEmbedder",  # noqa: E501
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
                "local_files_only": False,
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
        assert component.local_files_only is False
        assert component.truncate_dim == 256
        assert component.precision == "int8"

    @patch(
        "haystack.components.embedders.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        embedder = SentenceTransformersTextEmbedder(
            model="model",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            tokenizer_kwargs={"model_max_length": 512},
        )
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.embedding_backend.model.max_seq_length = 512
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="model",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            local_files_only=False,
            truncate_dim=None,
            model_kwargs=None,
            tokenizer_kwargs={"model_max_length": 512},
            config_kwargs=None,
            backend="torch",
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
        embedder.embedding_backend.embed = lambda x, **kwargs: [
            [random.random() for _ in range(16)] for _ in range(len(x))
        ]

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
    @pytest.mark.slow
    def test_run_trunc(self, monkeypatch):
        """
        sentence-transformers/paraphrase-albert-small-v2 maps sentences & paragraphs to a 768 dimensional dense vector
        space
        """
        monkeypatch.delenv("HF_API_TOKEN", raising=False)  # https://github.com/deepset-ai/haystack/issues/8811
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
    @pytest.mark.slow
    def test_run_quantization(self):
        """
        sentence-transformers/paraphrase-albert-small-v2 maps sentences & paragraphs to a 768 dimensional dense vector
        space
        """
        checkpoint = "sentence-transformers/paraphrase-albert-small-v2"
        text = "a nice text to embed"

        embedder_def = SentenceTransformersTextEmbedder(model=checkpoint, precision="int8")
        embedder_def.warm_up()
        result_def = embedder_def.run(text=text)
        embedding_def = result_def["embedding"]

        assert len(embedding_def) == 768
        assert all(isinstance(el, int) for el in embedding_def)

    def test_embed_encode_kwargs(self):
        embedder = SentenceTransformersTextEmbedder(model="model", encode_kwargs={"task": "retrieval.query"})
        embedder.embedding_backend = MagicMock()
        text = "a nice text to embed"
        embedder.run(text=text)
        embedder.embedding_backend.embed.assert_called_once_with(
            [text],
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=False,
            precision="float32",
            task="retrieval.query",
        )

    @patch(
        "haystack.components.embedders.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_model_onnx_backend(self, mocked_factory):
        onnx_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            # setting the path isn't necessary if the repo contains a "onnx/model.onnx" file but this is to prevent
            # a HF warning
            model_kwargs={"file_name": "onnx/model.onnx"},
            backend="onnx",
        )
        onnx_embedder.warm_up()

        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            local_files_only=False,
            truncate_dim=None,
            model_kwargs={"file_name": "onnx/model.onnx"},
            tokenizer_kwargs=None,
            config_kwargs=None,
            backend="onnx",
        )

    @patch(
        "haystack.components.embedders.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_model_openvino_backend(self, mocked_factory):
        openvino_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            # setting the path isn't necessary if the repo contains a "openvino/openvino_model.xml" file but
            # this is to prevent a HF warning
            model_kwargs={"file_name": "openvino/openvino_model.xml"},
            backend="openvino",
        )
        openvino_embedder.warm_up()

        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            local_files_only=False,
            truncate_dim=None,
            model_kwargs={"file_name": "openvino/openvino_model.xml"},
            tokenizer_kwargs=None,
            config_kwargs=None,
            backend="openvino",
        )

    @patch(
        "haystack.components.embedders.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    @pytest.mark.parametrize("model_kwargs", [{"torch_dtype": "bfloat16"}, {"torch_dtype": "float16"}])
    def test_dtype_on_gpu(self, mocked_factory, model_kwargs):
        torch_dtype_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cuda:0"),
            model_kwargs=model_kwargs,
        )
        torch_dtype_embedder.warm_up()

        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cuda:0",
            auth_token=None,
            trust_remote_code=False,
            local_files_only=False,
            truncate_dim=None,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=None,
            config_kwargs=None,
            backend="torch",
        )
