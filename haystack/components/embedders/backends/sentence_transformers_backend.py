# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Literal

import numpy as np

from haystack.lazy_imports import LazyImport
from haystack.utils.auth import Secret

with LazyImport(message="Run 'pip install \"sentence-transformers>=5.0.0\"'") as sentence_transformers_import:
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util import quantize_embeddings

with LazyImport(message="Run 'pip install \"pillow\"'") as pillow_import:
    from PIL.Image import Image


class _SentenceTransformersEmbeddingBackendFactory:
    """
    Factory class to create instances of Sentence Transformers embedding backends.
    """

    _instances: dict[str, "_SentenceTransformersEmbeddingBackend"] = {}

    @staticmethod
    def get_embedding_backend(
        *,
        model: str,
        device: str | None = None,
        auth_token: Secret | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        truncate_dim: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ) -> "_SentenceTransformersEmbeddingBackend":
        cache_params = {
            "model": model,
            "device": device,
            "auth_token": auth_token,
            "trust_remote_code": trust_remote_code,
            "revision": revision,
            "local_files_only": local_files_only,
            "truncate_dim": truncate_dim,
            "model_kwargs": model_kwargs,
            "tokenizer_kwargs": tokenizer_kwargs,
            "config_kwargs": config_kwargs,
            "backend": backend,
        }

        embedding_backend_id = json.dumps(cache_params, sort_keys=True, default=str)

        if embedding_backend_id in _SentenceTransformersEmbeddingBackendFactory._instances:
            return _SentenceTransformersEmbeddingBackendFactory._instances[embedding_backend_id]

        embedding_backend = _SentenceTransformersEmbeddingBackend(
            model=model,
            device=device,
            auth_token=auth_token,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            truncate_dim=truncate_dim,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            backend=backend,
        )

        _SentenceTransformersEmbeddingBackendFactory._instances[embedding_backend_id] = embedding_backend
        return embedding_backend


class _SentenceTransformersEmbeddingBackend:
    """
    Class to manage Sentence Transformers embeddings.
    """

    def __init__(
        self,
        *,
        model: str,
        device: str | None = None,
        auth_token: Secret | None = None,
        trust_remote_code: bool = False,
        revision: str | None = None,
        local_files_only: bool = False,
        truncate_dim: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
        config_kwargs: dict[str, Any] | None = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ) -> None:
        sentence_transformers_import.check()

        self.model = SentenceTransformer(
            model_name_or_path=model,
            device=device,
            token=auth_token.resolve_value() if auth_token else None,
            trust_remote_code=trust_remote_code,
            revision=revision,
            local_files_only=local_files_only,
            truncate_dim=truncate_dim,
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
            config_kwargs=config_kwargs,
            backend=backend,
        )

    def embed(self, data: list[str] | list["Image"], **kwargs: Any) -> list[list[float]]:
        quantization_ranges = kwargs.pop("quantization_ranges", None)
        precision = kwargs.get("precision", "float32")
        if quantization_ranges is not None and precision in ("int8", "uint8"):
            # scalar quantization calibrates min/max ranges from the batch itself, which is degenerate for
            # small batches (a single text produces meaningless embeddings), so we quantize with explicit ranges
            kwargs["precision"] = "float32"
            embeddings = self.model.encode(data, **kwargs)
            return quantize_embeddings(embeddings, precision=precision, ranges=np.asarray(quantization_ranges)).tolist()
        return self.model.encode(data, **kwargs).tolist()
