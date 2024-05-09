# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import io
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

from numpy import ndarray
from pandas import DataFrame, read_json

from haystack import logging
from haystack.dataclasses.byte_stream import ByteStream
from haystack.dataclasses.sparse_embedding import SparseEmbedding

logger = logging.getLogger(__name__)


class _BackwardCompatible(type):
    """
    Metaclass that handles Document backward compatibility.
    """

    def __call__(cls, *args, **kwargs):
        """
        Called before Document.__init__, will remap legacy fields to new ones.

        Also handles building a Document from a flattened dictionary.
        """
        # Move `content` to new fields depending on the type
        content = kwargs.get("content")
        if isinstance(content, DataFrame):
            kwargs["dataframe"] = content
            del kwargs["content"]

        # Not used anymore
        if "content_type" in kwargs:
            del kwargs["content_type"]

        # Embedding were stored as NumPy arrays in 1.x, so we convert it to the new type
        if isinstance(embedding := kwargs.get("embedding"), ndarray):
            kwargs["embedding"] = embedding.tolist()

        # id_hash_keys is not used anymore
        if "id_hash_keys" in kwargs:
            del kwargs["id_hash_keys"]

        return super().__call__(*args, **kwargs)


@dataclass
class Document(metaclass=_BackwardCompatible):
    """
    Base data class containing some data to be queried.

    Can contain text snippets, tables, and file paths to images or audios. Documents can be sorted by score and saved
    to/from dictionary and JSON.

    :param id: Unique identifier for the document. When not set, it's generated based on the Document fields' values.
    :param content: Text of the document, if the document contains text.
    :param dataframe: Pandas dataframe with the document's content, if the document contains tabular data.
    :param blob: Binary data associated with the document, if the document has any binary data associated with it.
    :param meta: Additional custom metadata for the document. Must be JSON-serializable.
    :param score: Score of the document. Used for ranking, usually assigned by retrievers.
    :param embedding: dense vector representation of the document.
    :param sparse_embedding: sparse vector representation of the document.
    """

    id: str = field(default="")
    content: Optional[str] = field(default=None)
    dataframe: Optional[DataFrame] = field(default=None)
    blob: Optional[ByteStream] = field(default=None)
    meta: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = field(default=None)
    embedding: Optional[List[float]] = field(default=None)
    sparse_embedding: Optional[SparseEmbedding] = field(default=None)

    def __repr__(self):
        fields = []
        if self.content is not None:
            fields.append(
                f"content: '{self.content}'" if len(self.content) < 100 else f"content: '{self.content[:100]}...'"
            )
        if self.dataframe is not None:
            fields.append(f"dataframe: {self.dataframe.shape}")
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

    def __eq__(self, other):
        """
        Compares Documents for equality.

        Two Documents are considered equals if their dictionary representation is identical.
        """
        if type(self) != type(other):
            return False
        return self.to_dict() == other.to_dict()

    def __post_init__(self):
        """
        Generate the ID based on the init parameters.
        """
        # Generate an id only if not explicitly set
        self.id = self.id or self._create_id()

    def _create_id(self):
        """
        Creates a hash of the given content that acts as the document's ID.
        """
        text = self.content or None
        dataframe = self.dataframe.to_json() if self.dataframe is not None else None
        blob = self.blob.data if self.blob is not None else None
        mime_type = self.blob.mime_type if self.blob is not None else None
        meta = self.meta or {}
        embedding = self.embedding if self.embedding is not None else None
        sparse_embedding = self.sparse_embedding.to_dict() if self.sparse_embedding is not None else ""
        data = f"{text}{dataframe}{blob}{mime_type}{meta}{embedding}{sparse_embedding}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def to_dict(self, flatten=True) -> Dict[str, Any]:
        """
        Converts Document into a dictionary.

        `dataframe` and `blob` fields are converted to JSON-serializable types.

        :param flatten:
            Whether to flatten `meta` field or not. Defaults to `True` to be backward-compatible with Haystack 1.x.
        """
        data = asdict(self)
        if (dataframe := data.get("dataframe")) is not None:
            data["dataframe"] = dataframe.to_json()
        if (blob := data.get("blob")) is not None:
            data["blob"] = {"data": list(blob["data"]), "mime_type": blob["mime_type"]}

        if flatten:
            meta = data.pop("meta")
            return {**data, **meta}

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """
        Creates a new Document object from a dictionary.

        The `dataframe` and `blob` fields are converted to their original types.
        """
        if (dataframe := data.get("dataframe")) is not None:
            data["dataframe"] = read_json(io.StringIO(dataframe))
        if blob := data.get("blob"):
            data["blob"] = ByteStream(data=bytes(blob["data"]), mime_type=blob["mime_type"])
        if sparse_embedding := data.get("sparse_embedding"):
            data["sparse_embedding"] = SparseEmbedding.from_dict(sparse_embedding)

        # Store metadata for a moment while we try un-flattening allegedly flatten metadata.
        # We don't expect both a `meta=` keyword and flatten metadata keys so we'll raise a
        # ValueError later if this is the case.
        meta = data.pop("meta", {})
        # Unflatten metadata if it was flattened. We assume any keyword argument that's not
        # a document field is a metadata key. We treat legacy fields as document fields
        # for backward compatibility.
        flatten_meta = {}
        legacy_fields = ["content_type", "id_hash_keys"]
        document_fields = legacy_fields + [f.name for f in fields(cls)]
        for key in list(data.keys()):
            if key not in document_fields:
                flatten_meta[key] = data.pop(key)

        # We don't support passing both flatten keys and the `meta` keyword parameter
        if meta and flatten_meta:
            raise ValueError(
                "You can pass either the 'meta' parameter or flattened metadata keys as keyword arguments, "
                "but currently you're passing both. Pass either the 'meta' parameter or flattened metadata keys."
            )

        # Finally put back all the metadata
        return cls(**data, meta={**meta, **flatten_meta})

    @property
    def content_type(self):
        """
        Returns the type of the content for the document.

        This is necessary to keep backward compatibility with 1.x.

        :raises ValueError:
            If both `text` and `dataframe` fields are set or both are missing.
        """
        if self.content is not None and self.dataframe is not None:
            raise ValueError("Both text and dataframe are set.")

        if self.content is not None:
            return "text"
        elif self.dataframe is not None:
            return "table"
        raise ValueError("Neither text nor dataframe is set.")
