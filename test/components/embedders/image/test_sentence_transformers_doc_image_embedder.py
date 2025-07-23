# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import glob
import random
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from haystack import Document
from haystack.components.embedders.image.sentence_transformers_doc_image_embedder import (
    SentenceTransformersDocumentImageEmbedder,
)
from haystack.utils.auth import Secret
from haystack.utils.device import ComponentDevice

IMPORT_PATH = "haystack.components.embedders.image.sentence_transformers_doc_image_embedder"


class TestSentenceTransformersDocumentImageEmbedder:
    def test_init_default(self):
        embedder = SentenceTransformersDocumentImageEmbedder(model="model")
        assert embedder.file_path_meta_field == "file_path"
        assert embedder.root_path == ""
        assert embedder.model == "model"
        assert embedder.device == ComponentDevice.resolve_device(None)
        assert embedder.token == Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False)
        assert embedder.batch_size == 32
        assert embedder.progress_bar is True
        assert embedder.trust_remote_code is False
        assert embedder.local_files_only is False
        assert embedder.precision == "float32"
        assert embedder._embedding_backend is None

    def test_init_with_parameters(self):
        embedder = SentenceTransformersDocumentImageEmbedder(
            file_path_meta_field="custom_file_path",
            root_path="root_path",
            model="model",
            device=ComponentDevice.from_str("cuda:0"),
            token=Secret.from_token("fake-api-token"),
            batch_size=64,
            progress_bar=False,
            trust_remote_code=True,
            local_files_only=True,
            precision="int8",
        )
        assert embedder.file_path_meta_field == "custom_file_path"
        assert embedder.root_path == "root_path"
        assert embedder.model == "model"
        assert embedder.device == ComponentDevice.from_str("cuda:0")
        assert embedder.token == Secret.from_token("fake-api-token")
        assert embedder.batch_size == 64
        assert embedder.progress_bar is False
        assert embedder.trust_remote_code
        assert embedder.local_files_only
        assert embedder.precision == "int8"
        assert embedder.backend == "torch"
        assert embedder.model_kwargs is None
        assert embedder.tokenizer_kwargs is None
        assert embedder.config_kwargs is None
        assert embedder.encode_kwargs is None
        assert embedder._embedding_backend is None

    def test_to_dict(self):
        component = SentenceTransformersDocumentImageEmbedder(
            model="model", device=ComponentDevice.from_str("cpu"), model_kwargs={"torch_dtype": "torch.float32"}
        )
        data = component.to_dict()
        assert data == {
            "type": f"{IMPORT_PATH}.SentenceTransformersDocumentImageEmbedder",
            "init_parameters": {
                "file_path_meta_field": "file_path",
                "root_path": "",
                "model": "model",
                "device": ComponentDevice.from_str("cpu").to_dict(),
                "token": {"env_vars": ["HF_API_TOKEN", "HF_TOKEN"], "strict": False, "type": "env_var"},
                "batch_size": 32,
                "progress_bar": True,
                "normalize_embeddings": False,
                "trust_remote_code": False,
                "local_files_only": False,
                "model_kwargs": {"torch_dtype": "torch.float32"},
                "tokenizer_kwargs": None,
                "encode_kwargs": None,
                "config_kwargs": None,
                "precision": "float32",
                "backend": "torch",
            },
        }

    def test_from_dict(self):
        init_parameters = {
            "file_path_meta_field": "custom_file_path",
            "root_path": "root_path",
            "model": "model",
            "device": ComponentDevice.from_str("cuda:0").to_dict(),
            "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
            "batch_size": 64,
            "progress_bar": False,
            "normalize_embeddings": True,
            "trust_remote_code": True,
            "local_files_only": True,
            "model_kwargs": {"torch_dtype": "torch.float32"},
            "tokenizer_kwargs": {"model_max_length": 512},
            "config_kwargs": {"use_memory_efficient_attention": True},
            "precision": "int8",
        }
        component = SentenceTransformersDocumentImageEmbedder.from_dict(
            {"type": f"{IMPORT_PATH}.SentenceTransformersDocumentImageEmbedder", "init_parameters": init_parameters}
        )
        assert component.file_path_meta_field == "custom_file_path"
        assert component.root_path == "root_path"
        assert component.model == "model"
        assert component.device == ComponentDevice.from_str("cuda:0")
        assert component.token == Secret.from_env_var("ENV_VAR", strict=False)
        assert component.batch_size == 64
        assert component.progress_bar is False
        assert component.normalize_embeddings is True
        assert component.trust_remote_code
        assert component.local_files_only
        assert component.model_kwargs == {"torch_dtype": torch.float32}
        assert component.tokenizer_kwargs == {"model_max_length": 512}
        assert component.config_kwargs == {"use_memory_efficient_attention": True}
        assert component.precision == "int8"

    def test_from_dict_none_device(self):
        init_parameters = {
            "file_path_meta_field": "custom_file_path",
            "root_path": "root_path",
            "model": "model",
            "device": None,
            "token": {"env_vars": ["ENV_VAR"], "strict": False, "type": "env_var"},
            "batch_size": 64,
            "progress_bar": False,
            "normalize_embeddings": False,
            "trust_remote_code": True,
            "local_files_only": False,
            "precision": "float32",
        }
        component = SentenceTransformersDocumentImageEmbedder.from_dict(
            {"type": f"{IMPORT_PATH}.SentenceTransformersDocumentImageEmbedder", "init_parameters": init_parameters}
        )
        assert component.file_path_meta_field == "custom_file_path"
        assert component.root_path == "root_path"
        assert component.model == "model"
        assert component.device == ComponentDevice.resolve_device(None)
        assert component.token == Secret.from_env_var("ENV_VAR", strict=False)
        assert component.batch_size == 64
        assert component.progress_bar is False
        assert component.trust_remote_code
        assert component.local_files_only is False
        assert component.precision == "float32"

    @patch(f"{IMPORT_PATH}._SentenceTransformersEmbeddingBackendFactory")
    def test_warmup(self, mocked_factory):
        embedder = SentenceTransformersDocumentImageEmbedder(
            model="model",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            tokenizer_kwargs={"model_max_length": 512},
            config_kwargs={"use_memory_efficient_attention": True},
        )
        mocked_factory.get_embedding_backend.assert_not_called()

        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="model",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            local_files_only=False,
            model_kwargs=None,
            tokenizer_kwargs={"model_max_length": 512},
            config_kwargs={"use_memory_efficient_attention": True},
            backend="torch",
        )

    @patch(f"{IMPORT_PATH}._SentenceTransformersEmbeddingBackendFactory")
    def test_warmup_doesnt_reload(self, mocked_factory):
        embedder = SentenceTransformersDocumentImageEmbedder(model="model")
        mocked_factory.get_embedding_backend.assert_not_called()
        embedder.warm_up()
        embedder.warm_up()
        mocked_factory.get_embedding_backend.assert_called_once()

    def test_run(self, test_files_path):
        embedder = SentenceTransformersDocumentImageEmbedder(model="model")
        embedder._embedding_backend = MagicMock()
        embedder._embedding_backend.embed = lambda x, **kwargs: [
            [random.random() for _ in range(16)] for _ in range(len(x))
        ]

        image_paths = glob.glob(str(test_files_path / "images" / "*.*")) + glob.glob(
            str(test_files_path / "pdf" / "*.pdf")
        )
        documents = []
        for i, path in enumerate(image_paths):
            document = Document(content=f"document number {i}", meta={"file_path": path})
            if path.endswith(".pdf"):
                document.meta["page_number"] = 1
            documents.append(document)

        result = embedder.run(documents=documents)

        assert isinstance(result["documents"], list)
        assert len(result["documents"]) == len(documents)
        for doc in result["documents"]:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert isinstance(doc.embedding[0], float)
            assert "embedding_source" in doc.meta
            assert doc.meta["embedding_source"]["type"] == "image"
            assert "file_path_meta_field" in doc.meta["embedding_source"]

    def test_run_no_warmup(self):
        embedder = SentenceTransformersDocumentImageEmbedder(model="model")

        with pytest.raises(RuntimeError, match="The embedding model has not been loaded."):
            embedder.run(documents=[Document(content="test")])

    def test_run_wrong_input_format(self):
        embedder = SentenceTransformersDocumentImageEmbedder(model="model")

        string_input = "text"
        list_integers_input = [1, 2, 3]

        with pytest.raises(
            TypeError, match="SentenceTransformersDocumentImageEmbedder expects a list of Documents as input"
        ):
            embedder.run(documents=string_input)

        with pytest.raises(
            TypeError, match="SentenceTransformersDocumentImageEmbedder expects a list of Documents as input"
        ):
            embedder.run(documents=list_integers_input)

    @patch(f"{IMPORT_PATH}._SentenceTransformersEmbeddingBackendFactory")
    def test_model_onnx_backend(self, mocked_factory):
        onnx_embedder = SentenceTransformersDocumentImageEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            model_kwargs={"file_name": "onnx/model.onnx"},
            # setting the path isn't necessary if the repo contains a "onnx/model.onnx" file
            # but this is to prevent a HF warning
            backend="onnx",
        )
        onnx_embedder.warm_up()

        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            local_files_only=False,
            model_kwargs={"file_name": "onnx/model.onnx"},
            tokenizer_kwargs=None,
            config_kwargs=None,
            backend="onnx",
        )

    @patch(f"{IMPORT_PATH}._SentenceTransformersEmbeddingBackendFactory")
    def test_model_openvino_backend(self, mocked_factory):
        openvino_embedder = SentenceTransformersDocumentImageEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2",
            token=None,
            device=ComponentDevice.from_str("cpu"),
            model_kwargs={"file_name": "openvino/openvino_model.xml"},
            # setting the path isn't necessary if the repo contains a "openvino/openvino_model.xml" file
            # but this is to prevent a HF warning
            backend="openvino",
        )
        openvino_embedder.warm_up()

        mocked_factory.get_embedding_backend.assert_called_once_with(
            model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            auth_token=None,
            trust_remote_code=False,
            local_files_only=False,
            model_kwargs={"file_name": "openvino/openvino_model.xml"},
            tokenizer_kwargs=None,
            config_kwargs=None,
            backend="openvino",
        )

    @patch(f"{IMPORT_PATH}._extract_image_sources_info")
    @patch(f"{IMPORT_PATH}._batch_convert_pdf_pages_to_images")
    @patch("PIL.Image.open")
    def test_run_none_images(
        self, mocked_pil_open, mocked_batch_convert_pdf_pages_to_images, mocked_extract_image_sources_info
    ):
        embedder = SentenceTransformersDocumentImageEmbedder(model="model")
        embedder._embedding_backend = MagicMock()

        mocked_extract_image_sources_info.return_value = [
            {"path": "doc1.pdf", "mime_type": "application/pdf", "page_number": 999},  # Page 999 doesn't exist
            {"path": "image1.jpg", "mime_type": "image/jpeg"},
        ]
        mocked_batch_convert_pdf_pages_to_images.return_value = {}  # Empty dict because page was skipped
        mocked_pil_open.return_value = Image.new("RGB", (100, 100))

        documents = [
            Document(content="PDF 1", meta={"file_path": "doc1.pdf", "page_number": 999}),
            Document(content="Image 1", meta={"file_path": "image1.jpg"}),
        ]

        with pytest.raises(RuntimeError, match="Conversion failed for some documents."):
            embedder.run(documents=documents)

    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.skipif(
        sys.platform == "darwin",
        reason=(
            "This model does not play well with GitHub macOS runners and"
            "we prefer to avoid altering PYTORCH_MPS_HIGH_WATERMARK_RATIO"
        ),
    )
    def test_live_run(self, test_files_path):
        embedder = SentenceTransformersDocumentImageEmbedder(model="sentence-transformers/clip-ViT-B-32")
        embedder.warm_up()

        documents = [
            Document(
                content="PDF document",
                meta={"file_path": str(test_files_path / "pdf" / "sample_pdf_1.pdf"), "page_number": 1},
            ),
            Document(content="Image document", meta={"file_path": str(test_files_path / "images" / "apple.jpg")}),
        ]

        result = embedder.run(documents=documents)
        assert len(result["documents"]) == len(documents)
        for doc in result["documents"]:
            assert isinstance(doc, Document)
            assert isinstance(doc.embedding, list)
            assert len(doc.embedding) == 512
            assert all(isinstance(x, float) for x in doc.embedding)
            assert "embedding_source" in doc.meta
            assert doc.meta["embedding_source"]["type"] == "image"
            assert "file_path_meta_field" in doc.meta["embedding_source"]
