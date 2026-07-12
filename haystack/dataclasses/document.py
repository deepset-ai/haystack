# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from numpy import ndarray

from haystack.dataclasses.byte_stream import ByteStream
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.utils.dataclasses import _warn_on_inplace_mutation

LEGACY_FIELDS = ["content_type", "id_hash_keys", "dataframe"]


class _BackwardCompatible(type):
    """
    Metaclass that handles Document backward compatibility.
    """

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Called before Document.__init__, handles legacy fields.

        Embedding was stored as NumPy arrays in 1.x, so we convert it to a list of floats.
        Other legacy fields are removed.
        """
        ### Conversion from 1.x Document ###
        content = kwargs.get("content")
        if content and not isinstance(content, str):
            raise ValueError("The `content` field must be a string or None.")

        # Embedding were stored as NumPy arrays in 1.x, so we convert it to the new type
        if isinstance(embedding := kwargs.get("embedding"), ndarray):
            kwargs["embedding"] = embedding.tolist()

        # Remove legacy fields
        for field_name in LEGACY_FIELDS:
            kwargs.pop(field_name, None)

        return super().__call__(*args, **kwargs)


@_warn_on_inplace_mutation
@dataclass
class Document(metaclass=_BackwardCompatible):  # noqa: PLW1641
    """
    Base data class containing some data to be queried.

    Can contain text snippets and file paths to images or audios. Documents can be sorted by score and saved
    to/from dictionary and JSON.

    :param id: Unique identifier for the document. When not set, it's generated based on the Document fields' values.
    :param content: Text of the document, if the document contains text.
    :param blob: Binary data associated with the document, if the document has any binary data associated with it.
    :param meta: Additional custom metadata for the document. Must be JSON-serializable.
    :param score: Score of the document. Used for ranking, usually assigned by retrievers.
    :param embedding: dense vector representation of the document.
    :param sparse_embedding: sparse vector representation of the document.
    """

    id: str = field(default="")
    content: str | None = field(default=None)
    blob: ByteStream | None = field(default=None)
    meta: dict[str, Any] = field(default_factory=dict)
    score: float | None = field(default=None)
    embedding: list[float] | None = field(default=None)
    sparse_embedding: SparseEmbedding | None = field(default=None)

    def __repr__(self) -> str:
        fields = []
        if self.content is not None:
            fields.append(
                f"content: '{self.content}'" if len(self.content) < 100 else f"content: '{self.content[:100]}...'"
            )
        if self.blob is not None:
            fields.append(f"blob: {len(self.blob.data)} bytes")
        if len(self.meta) > 0:
            fields.append(f"meta: {self.meta}")
        if self.score is not None:
            fields.append(f"score: {self.score}")
        if self.embedding is not None:
            fields.append(f"embedding: vector of size {len(self.embedding)}")
        if self.sparse_embedding is not None:
            fields.append(f"sparse_embedding: vector with {len(self.sparse_embedding.indices)} non-zero elements")
        fields_str = ", ".join(fields)
        return f"{self.__class__.__name__}(id={self.id}, {fields_str})"

    def __eq__(self, other: object) -> bool:
        """
        Compares Documents for equality.

        Two Documents are considered equals if their dictionary representation is identical.
        """
        if type(self) != type(other):
            return False
        self_dict = self.to_dict(flatten=False)
        other_dict = other.to_dict(flatten=False)
        return self_dict == other_dict