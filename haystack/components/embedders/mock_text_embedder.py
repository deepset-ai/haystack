# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from haystack import component, default_from_dict, default_to_dict
from haystack.components.embedders.mock_utils import (
    EmbeddingFn,
    _coerce_embedding,
    _deterministic_embedding,
    _estimate_usage,
    _validate_embedding,
)
from haystack.utils import deserialize_callable, serialize_callable


@component
class MockTextEmbedder:
    """
    A Text Embedder that returns deterministic embeddings without calling any API.

    It is a drop-in replacement for real Text Embedders (such as `OpenAITextEmbedder`) in tests, smoke tests, and
    quick prototypes. It implements the same interface (`run`, `run_async`, serialization) but never contacts an
    external service, so it is fully deterministic and free to run.

    The embedding is selected based on how the component is configured:

    - **Deterministic (default)**: with no configuration, the embedding is derived from a hash of the input text.
      The same text always yields the same embedding, and different texts yield different embeddings, so the mock
      works in retrieval pipelines and is reproducible across runs and processes.
    - **Fixed embedding**: pass an `embedding` vector. The same vector is returned for every input.
    - **Dynamic embedding**: pass an `embedding_fn` callable that receives the (prepared) text and returns the
      embedding. This is useful when the embedding should depend on the input in a custom way.

    ### Usage example

    ```python
    from haystack.components.embedders import MockTextEmbedder

    embedder = MockTextEmbedder(dimension=8)
    result = embedder.run("I love pizza!")
    print(result["embedding"])  # a deterministic list of 8 floats
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
    ) -> None:
        """
        Creates an instance of MockTextEmbedder.

        :param embedding: An optional fixed embedding returned for every input. Mutually exclusive with
            `embedding_fn`. If neither is provided, a deterministic embedding is derived from the input text.
        :param embedding_fn: An optional callable that receives the prepared text (after `prefix`/`suffix` are
            applied) and returns the embedding as a list of floats. Mutually exclusive with `embedding`. To support
            serialization, pass a named function (lambdas and nested functions cannot be serialized).
        :param dimension: The number of dimensions of the deterministic embedding. Ignored when `embedding` or
            `embedding_fn` is provided, since their length is determined by the value or callable.
        :param model: The model name reported in the metadata. Purely cosmetic; no model is loaded.
        :param meta: Additional metadata merged into the output `meta`.
        :param prefix: A string to add at the beginning of the text before embedding.
        :param suffix: A string to add at the end of the text before embedding.
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
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MockTextEmbedder:
        """Deserialize the component from a dictionary."""
        init_params = data.get("init_parameters", {})
        embedding_fn = init_params.get("embedding_fn")
        if embedding_fn:
            init_params["embedding_fn"] = deserialize_callable(embedding_fn)
        return default_from_dict(cls, data)

    def _embed(self, text: str) -> list[float]:
        """Produce the embedding for the prepared text according to the configured mode."""
        if self.embedding_fn is not None:
            return _coerce_embedding(self.embedding_fn(text))
        if self.embedding is not None:
            return list(self.embedding)
        return _deterministic_embedding(text, self.dimension)

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    def run(self, text: str) -> dict[str, Any]:
        """
        Return a deterministic embedding for the input text without calling any API.

        :param text: The text to embed.
        :returns: A dictionary with the following keys:
            - `embedding`: The embedding of the input text.
            - `meta`: Metadata about the (mock) model.
        :raises TypeError: If `text` is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(
                "MockTextEmbedder expects a string as an input. "
                "In case you want to embed a list of Documents, please use the MockDocumentEmbedder."
            )

        text_to_embed = self.prefix + text + self.suffix
        meta: dict[str, Any] = {"model": self.model, "usage": _estimate_usage([text_to_embed])}
        meta.update(self.meta)
        return {"embedding": self._embed(text_to_embed), "meta": meta}

    @component.output_types(embedding=list[float], meta=dict[str, Any])
    async def run_async(self, text: str) -> dict[str, Any]:
        """
        Asynchronously return a deterministic embedding for the input text without calling any API.

        :param text: The text to embed.
        :returns: A dictionary with the following keys:
            - `embedding`: The embedding of the input text.
            - `meta`: Metadata about the (mock) model.
        :raises TypeError: If `text` is not a string.
        """
        return self.run(text=text)
