# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import random
from unittest.mock import MagicMock, patch

import pytest
import torch

from haystack.components.embedders.sentence_transformers_sparse_text_embedder import (
    SentenceTransformersSparseTextEmbedder,
)
from haystack.utils import ComponentDevice, Secret


class TestSentenceTransformersSparseTextEmbedder:
    def test_init_default(self):
        embedder = SentenceTransformersSparseTextEmbedder(model="model")
        assert embedder.model == "model"
        assert embedder.device == ComponentDevice.resolve_device(None)
        assert embedder.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert embedder.prefix == ""
        assert embedder.suffix == ""
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.trust_remote_code is False
        assert embedder.local_files_only is False

    def test_init_with_parameters(self):
        embedder = SentenceTransformersSparseTextEmbedder(
            model="model",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_token("fake-api-token"),
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            trust_remote_code=True,
            local_files_only=True,
        )
        assert embedder.model == "model"
        assert embedder.device == ComponentDevice.from_str("cuda:0")
        assert embedder.token == Secret.from_token("fake-api-token")
        assert embedder.prefix == "prefix"
        assert embedder.suffix == "suffix"
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.trust_remote_code is True
        assert embedder.local_files_only is True

    def test_to_dict(self):
        component = SentenceTransformersSparseTextEmbedder(model="model", device=ComponentDevice.from_str("cpu"))
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.sentence_transformers_sparse_text_embedder.SentenceTransformersSparseTextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "trust_remote_code": False,
                "local_files_only": False,
                "model_kwargs": None,
                "tokenizer_kwargs": None,
                "encode_kwargs": None,
                "config_kwargs": None,
                "backend": "torch",
            },
        }

    def test_to_dict_with_custom_init_parameters(self):
        component = SentenceTransformersSparseTextEmbedder(
            model="model",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_env_var("ENV_VAR", strict=False),
            prefix="prefix",
            suffix="suffix",
            batch_size=64,
            progress_bar=False,
            trust_remote_code=True,
            local_files_only=True,
            model_kwargs={"torch_dtype": torch.float32},
            tokenizer_kwargs={"model_max_length": 512},
            config_kwargs={"use_memory_efficient_attention": False},
            encode_kwargs={"task": "clustering"},
        )
        data = component.to_dict()
        assert data == {
            "type": "haystack.components.embedders.sentence_transformers_sparse_text_embedder.SentenceTransformersSparseTextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": ComponentDevice.from_str("cuda:0").to_dict(),
                "prefix": "prefix",
                "suffix": "suffix",
                "batch_size": 64,
                "progress_bar": False,
                "trust_remote_code": True,
                "local_files_only": True,
                "model_kwargs": {"torch_dtype": "torch.float32"},
                "tokenizer_kwargs": {"model_max_length": 512},
                "config_kwargs": {"use_memory_efficient_attention": False},
                "encode_kwargs": {"task": "clustering"},
                "backend": "torch",
            },
        }

    def test_to_dict_not_serialize_token(self):
        component = SentenceTransformersSparseTextEmbedder(model="model", token=Secret.from_token("fake-api-token"))
        with pytest.raises(ValueError, match="Cannot serialize token-based secret"):
            component.to_dict()

    def test_from_dict(self):
        data = {
            "type": "haystack.components.embedders.sentence_transformers_sparse_text_embedder.SentenceTransformersSparseTextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "trust_remote_code": False,
                "local_files_only": False,
                "model_kwargs": {"torch_dtype": "torch.float32"},
                "tokenizer_kwargs": {"model_max_length": 512},
                "config_kwargs": {"use_memory_efficient_attention": False},
            },
        }
        component = SentenceTransformersSparseTextEmbedder.from_dict(data)
        assert component.model == "model"
        assert component.device == ComponentDevice.from_str("cpu")
        assert component.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.batch_size == 32
        assert component.progress_bar is True
        assert component.trust_remote_code is False
        assert component.local_files_only is False
        assert component.model_kwargs == {"torch_dtype": torch.float32}
        assert component.tokenizer_kwargs == {"model_max_length": 512}
        assert component.config_kwargs == {"use_memory_efficient_attention": False}

    def test_from_dict_no_default_parameters(self):
        data = {
            "type": "haystack.components.embedders.sentence_transformers_sparse_text_embedder.SentenceTransformersSparseTextEmbedder",
            "init_parameters": {},
        }
        component = SentenceTransformersSparseTextEmbedder.from_dict(data)
        assert component.model == "naver/splade-cocondenser-ensembledistil"
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.batch_size == 32
        assert component.progress_bar is True
        assert component.trust_remote_code is False
        assert component.local_files_only is False

    def test_from_dict_none_device(self):
        data = {
            "type": "haystack.components.embedders.sentence_transformers_sparse_text_embedder.SentenceTransformersSparseTextEmbedder",
            "init_parameters": {
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "model": "model",
                "device": None,
                "prefix": "",
                "suffix": "",
                "batch_size": 32,
                "progress_bar": True,
                "trust_remote_code": False,
                "local_files_only": False,
            },
        }
        component = SentenceTransformersSparseTextEmbedder.from_dict(data)
        assert component.model == "model"
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert component.prefix == ""
        assert component.suffix == ""
        assert component.batch_size == 32
        assert component.progress_bar is True
        assert component.trust_remote_code is False
        assert component.local_files_only is False

    @patch(
        "haystack.components.embedders.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_warmup(self, mocked_factory):
        embedder = SentenceTransformersSparseTextEmbedder(
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
            model_kwargs=None,
            tokenizer_kwargs={"model_max_length": 512},
            config_kwargs=None,
            backend="torch",
        )

    @patch(
        "haystack.components.embedders.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_warmup_doesnt_reload(self, mocked_factory):
        embedder = SentenceTransformersSparseTextEmbedder(model="model")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()

    def test_run(self):
        embedder = SentenceTransformersSparseTextEmbedder(model="model")
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
        embedder = SentenceTransformersSparseTextEmbedder(model="model")
        embedder.embedding_backend = MagicMock()

        list_integers_input = [1, 2, 3]

        with pytest.raises(TypeError, match="SentenceTransformersSparseTextEmbedder expects a string as input"):
            embedder.run(text=list_integers_input)

    @patch(
        "haystack.components.embedders.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_model_onnx_backend(self, mocked_factory):
        onnx_embedder = SentenceTransformersSparseTextEmbedder(
            model="naver/splade-cocondenser-ensembledistil",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            model_kwargs={
                "file_name": "onnx/model.onnx"
            },  # setting the path isn't necessary if the repo contains a "onnx/model.onnx" file but this is to prevent a HF warning
            backend="onnx",
        )
        onnx_embedder.warm_up()

        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="naver/splade-cocondenser-ensembledistil",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            local_files_only=False,
            model_kwargs={"file_name": "onnx/model.onnx"},
            tokenizer_kwargs=None,
            config_kwargs=None,
            backend="onnx",
        )

    @patch(
        "haystack.components.embedders.sentence_transformers_text_embedder._SentenceTransformersEmbeddingBackendFactory"
    )
    def test_model_openvino_backend(self, mocked_factory):
        openvino_embedder = SentenceTransformersSparseTextEmbedder(
            model="naver/splade-cocondenser-ensembledistil",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            model_kwargs={
                "file_name": "openvino/openvino_model.xml"
            },  # setting the path isn't necessary if the repo contains a "openvino/openvino_model.xml" file but this is to prevent a HF warning
            backend="openvino",
        )
        openvino_embedder.warm_up()

        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="naver/splade-cocondenser-ensembledistil",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            local_files_only=False,
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
        torch_dtype_embedder = SentenceTransformersSparseTextEmbedder(
            model="naver/splade-cocondenser-ensembledistil",
            token=None,
            device=ComponentDevice.from_str("cuda:0"),
            model_kwargs=model_kwargs,
        )
        torch_dtype_embedder.warm_up()

        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="naver/splade-cocondenser-ensembledistil",
            device="cuda:0",
            auth_token=None,
            trust_remote_code=False,
            local_files_only=False,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=None,
            config_kwargs=None,
            backend="torch",
        )
