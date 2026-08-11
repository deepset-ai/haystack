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

_LEGACY_FIELDS = ["content_type", "id_hash_keys", "dataframe"]


class _RemoveLegacyFields(type):
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        """
        Called before Document.__init__, removes the legacy fields.
        """
        for field_name in _LEGACY_FIELDS:
            kwargs.pop(field_name, None)

        return super().__call__(*args, **kwargs)


@_warn_on_inplace_mutation
@dataclass
class Document(metaclass=_RemoveLegacyFields):  # noqa: PLW1641
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

    def __post_init__(self) -> None:
        """
        Checks content type, converts embedding from 1.x type and generates the ID based on the init parameters.
        """
        if self.content is not None and not isinstance(self.content, str):
            raise ValueError("The `content` field must be a string or None.")

        # Embeddings were stored as NumPy arrays in 1.x, so we convert them to the new type
        if isinstance(self.embedding, ndarray):
            self.embedding = self.embedding.tolist()

        # Generate an id only if not explicitly set
        self.id = self.id or self._create_id()

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
        return self.to_dict(flatten=False) == other.to_dict(flatten=False)

    @classmethod
    def _field_names(cls) -> set[str]:
        return set(_LEGACY_FIELDS) | {f.name for f in fields(cls)}

    def _create_id(self) -> str:
        """
        Creates a hash of the given content that acts as the document's ID.
        """
        text = self.content or None
        dataframe = None  # this allows the ID creation to remain unchanged even if the dataframe field has been removed
        blob = self.blob.data if self.blob is not None else None
        mime_type = self.blob.mime_type if self.blob is not None else None
        # Sort keys so meta order doesn't affect the ID. Keep "{}" for empty meta so existing IDs stay stable.
        meta = json.dumps(self.meta, sort_keys=True, default=str) if self.meta else "{}"
        embedding = self.embedding if self.embedding is not None else None
        sparse_embedding = self.sparse_embedding.to_dict() if self.sparse_embedding is not None else ""
        data = f"{text}{dataframe}{blob!r}{mime_type}{meta}{embedding}{sparse_embedding}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def to_dict(self, flatten: bool = True) -> dict[str, Any]:
        """
        Converts Document into a dictionary.

        `blob` field is converted to a JSON-serializable type.

        :param flatten:
            Whether to flatten the `meta` field. Defaults to `True` to be backward-compatible with Haystack 1.x.
            Meta keys that clash with document field names are kept in a nested `meta` dictionary.
        """
        data = asdict(self)

        # Use `ByteStream` and `SparseEmbedding`'s to_dict methods to convert them to JSON-serializable types.
        if self.blob is not None:
            data["blob"] = self.blob.to_dict()
        if self.sparse_embedding is not None:
            data["sparse_embedding"] = self.sparse_embedding.to_dict()

        if not flatten:
            return data

        field_names = self._field_names()
        flattened_meta: dict[str, Any] = {}
        colliding_meta: dict[str, Any] = {}
        for key, value in data.pop("meta").items():
            if key in field_names:
                colliding_meta[key] = value  # would clash with a document field: keep it nested
            else:
                flattened_meta[key] = value

        data = {**flattened_meta, **data}
        if colliding_meta:
            data["meta"] = colliding_meta
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        """
        Creates a new Document object from a dictionary.

        The `blob` field is converted to its original type.
        """
        field_names = cls._field_names()
        field_data: dict[str, Any] = {}
        flattened_meta: dict[str, Any] = {}
        for key, value in data.items():
            if key == "meta":
                continue  # merged into meta at the end
            if key in field_names:
                field_data[key] = value  # legacy fields included: _RemoveLegacyFields drops them before __init__
            else:
                flattened_meta[key] = value

        if blob := field_data.get("blob"):
            field_data["blob"] = ByteStream.from_dict(blob)
        if sparse_embedding := field_data.get("sparse_embedding"):
            field_data["sparse_embedding"] = SparseEmbedding.from_dict(sparse_embedding)

        nested_meta = data.get("meta") or {}
        return cls(**field_data, meta={**nested_meta, **flattened_meta})

    @property
    def content_type(self) -> str:
        """
        Returns the type of the content for the document.

        This is necessary to keep backward compatibility with 1.x.
        """
        if self.content is not None:
            return "text"
        raise ValueError("Content is not set.")
