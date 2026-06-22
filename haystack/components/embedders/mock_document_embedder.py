# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.components.embedders.mock_utils import (
    EmbeddingFn,
    _coerce_embedding,
    _deterministic_embedding,
    _estimate_usage,
    _validate_embedding,
)
from haystack.utils import deserialize_callable, serialize_callable


@component
class MockDocumentEmbedder:
    """
    A Document Embedder that returns deterministic embeddings without calling any API.

    It is a drop-in replacement for real Document Embedders (such as `OpenAIDocumentEmbedder`) in tests, smoke tests,
    and quick prototypes. It implements the same interface (`run`, `run_async`, serialization) but never contacts an
    external service, so it is fully deterministic and free to run.

    The embedding is selected based on how the component is configured:

    - **Deterministic (default)**: with no configuration, each document's embedding is derived from a hash of its
      (prepared) text. The same text always yields the same embedding, and different texts yield different
      embeddings, so the mock works in retrieval pipelines and is reproducible across runs and processes.
    - **Fixed embedding**: pass an `embedding` vector. The same vector is assigned to every document.
    - **Dynamic embedding**: pass an `embedding_fn` callable that receives the (prepared) text of a document and
      returns the embedding. This is useful when the embedding should depend on the input in a custom way.

    Like real Document Embedders, the metadata fields listed in `meta_fields_to_embed` are concatenated with the
    document content before embedding, so the deterministic embedding reflects the embedded metadata.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.embedders import MockDocumentEmbedder

    embedder = MockDocumentEmbedder(dimension=8)
    result = embedder.run([Document(content="I love pizza!")])
    print(result["documents"][0].embedding)  # a deterministic list of 8 floats
    ```
    """

    def __init__(
        self,
        embedding: list[float] | None = None,
        *,
        embedding_fn: EmbeddingFn | None = None,
        dimension: int = 768,
        model: str = "mock-model",
        meta: dict[str, Any] | None = None,
        prefix: str = "",
        suffix: str = "",
        meta_fields_to_embed: list[str] | None = None,
        embedding_separator: str = "\n",
        progress_bar: bool = False,
    ) -> None:
        """
        Creates an instance of MockDocumentEmbedder.

        :param embedding: An optional fixed embedding assigned to every document. Mutually exclusive with
            `embedding_fn`. If neither is provided, a deterministic embedding is derived from each document's text.
        :param embedding_fn: An optional callable that receives the prepared text of a document and returns the
            embedding as a list of floats. Mutually exclusive with `embedding`. To support serialization, pass a
            named function (lambdas and nested functions cannot be serialized).
        :param dimension: The number of dimensions of the deterministic embedding. Ignored when `embedding` or
            `embedding_fn` is provided, since their length is determined by the value or callable.
        :param model: The model name reported in the metadata. Purely cosmetic; no model is loaded.
        :param meta: Additional metadata merged into the output `meta`.
        :param prefix: A string to add at the beginning of each text before embedding.
        :param suffix: A string to add at the end of each text before embedding.
        :param meta_fields_to_embed: List of metadata fields to embed along with the document text.
        :param embedding_separator: Separator used to concatenate the metadata fields to the document text.
        :param progress_bar: Accepted for interface compatibility with real Document Embedders and ignored.
        :raises ValueError: If both `embedding` and `embedding_fn` are provided, or if `dimension` is not positive.
        """
        if embedding is not None and embedding_fn is not None:
            raise ValueError("Pass either 'embedding' or 'embedding_fn', not both.")
        if dimension <= 0:
            raise ValueError("'dimension' must be a positive integer.")

        self.embedding = _validate_embedding(embedding) if embedding is not None else None
        self.embedding_fn = embedding_fn
        self.dimension = dimension
        self.model = model
        self.meta = meta or {}
        self.prefix = prefix
        self.suffix = suffix
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.progress_bar = progress_bar

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        embedding_fn = serialize_callable(self.embedding_fn) if self.embedding_fn is not None else None
        return default_to_dict(
            self,
            embedding=self.embedding,
            embedding_fn=embedding_fn,
            dimension=self.dimension,
            model=self.model,
            meta=self.meta,
            prefix=self.prefix,
            suffix=self.suffix,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            progress_bar=self.progress_bar,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MockDocumentEmbedder:
        """Deserialize the component from a dictionary."""
        init_params = data.get("init_parameters", {})
        embedding_fn = init_params.get("embedding_fn")
        if embedding_fn:
            init_params["embedding_fn"] = deserialize_callable(embedding_fn)
        return default_from_dict(cls, data)

    def _prepare_text_to_embed(self, document: Document) -> str:
        """Concatenate the document content with the metadata fields to embed, mirroring real Document Embedders."""
        meta_values_to_embed = [
            str(document.meta[key])
            for key in self.meta_fields_to_embed
            if key in document.meta and document.meta[key] is not None
        ]
        return (
            self.prefix + self.embedding_separator.join([*meta_values_to_embed, document.content or ""]) + self.suffix
        )

    def _embed(self, text: str) -> list[float]:
        """Produce the embedding for the prepared text according to the configured mode."""
        if self.embedding_fn is not None:
            return _coerce_embedding(self.embedding_fn(text))
        if self.embedding is not None:
            return list(self.embedding)
        return _deterministic_embedding(text, self.dimension)

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    def run(self, documents: list[Document]) -> dict[str, Any]:
        """
        Return the input documents with deterministic embeddings added, without calling any API.

        :param documents: A list of documents to embed.
        :returns: A dictionary with the following keys:
            - `documents`: A list of documents with embeddings.
            - `meta`: Metadata about the (mock) model.
        :raises TypeError: If `documents` is not a list of `Document` objects.
        """
        if not isinstance(documents, list) or (documents and not isinstance(documents[0], Document)):
            raise TypeError(
                "MockDocumentEmbedder expects a list of Documents as input. "
                "In case you want to embed a string, please use the MockTextEmbedder."
            )

        texts_to_embed = [self._prepare_text_to_embed(document) for document in documents]
        new_documents = [
            replace(document, embedding=self._embed(text))
            for document, text in zip(documents, texts_to_embed, strict=True)
        ]

        meta: dict[str, Any] = {"model": self.model, "usage": _estimate_usage(texts_to_embed)}
        meta.update(self.meta)
        return {"documents": new_documents, "meta": meta}

    @component.output_types(documents=list[Document], meta=dict[str, Any])
    async def run_async(self, documents: list[Document]) -> dict[str, Any]:
        """
        Asynchronously return the input documents with deterministic embeddings added, without calling any API.

        :param documents: A list of documents to embed.
        :returns: A dictionary with the following keys:
            - `documents`: A list of documents with embeddings.
            - `meta`: Metadata about the (mock) model.
        :raises TypeError: If `documents` is not a list of `Document` objects.
        """
        return self.run(documents=documents)
