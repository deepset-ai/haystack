import hashlib
import logging
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional, Type, cast

import numpy
import pandas

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
        if isinstance(content, str):
            kwargs["text"] = content
        elif isinstance(content, pandas.DataFrame):
            kwargs["dataframe"] = content

        # We already moved `content` to `text` or `dataframe`, we can remove it
        if "content" in kwargs:
            del kwargs["content"]

        # Not used anymore
        if "content_type" in kwargs:
            del kwargs["content_type"]

        # Embedding were stored as NumPy arrays in 1.x, so we convert it to the new type
        if isinstance(embedding := kwargs.get("embedding"), numpy.ndarray):
            kwargs["embedding"] = embedding.tolist()

        # id_hash_keys is not used anymore
        if "id_hash_keys" in kwargs:
            del kwargs["id_hash_keys"]

        if kwargs.get("metadata") is None:
            # This must be a flattened Document, so we treat all keys that are not
            # Document fields as metadata.
            metadata = {}
            field_names = [f.name for f in fields(cast(Type[Document], cls))]
            keys = list(kwargs.keys())  # get a list of the keys as we'll modify the dict in the loop
            for key in keys:
                if key in field_names:
                    continue
                metadata[key] = kwargs.pop(key)
            kwargs["metadata"] = metadata

        return super().__call__(*args, **kwargs)


@dataclass
class Document(metaclass=_BackwardCompatible):
    """
    Base data class containing some data to be queried.
    Can contain text snippets, tables, and file paths to images or audios.
    Documents can be sorted by score and saved to/from dictionary and JSON.

    :param id: Unique identifier for the document. When not set, it's generated based on the Document fields' values.
    :param text: Text of the document, if the document contains text.
    :param dataframe: Pandas dataframe with the document's content, if the document contains tabular data.
    :param blob: Binary data associated with the document, if the document has any binary data associated with it.
    :param mime_type: MIME type of the document. Defaults to "text/plain".
    :param metadata: Additional custom metadata for the document. Must be JSON-serializable.
    :param score: Score of the document. Used for ranking, usually assigned by retrievers.
    :param embedding: Vector representation of the document.
    """

    id: str = field(default="")
    text: Optional[str] = field(default=None)
    dataframe: Optional[pandas.DataFrame] = field(default=None)
    blob: Optional[bytes] = field(default=None)
    mime_type: str = field(default="text/plain")
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: Optional[float] = field(default=None)
    embedding: Optional[List[float]] = field(default=None, repr=False)

    def __str__(self):
        fields = [f"mimetype: '{self.mime_type}'"]
        if self.text is not None:
            fields.append(f"text: '{self.text}'" if len(self.text) < 100 else f"text: '{self.text[:100]}...'")
        if self.dataframe is not None:
            fields.append(f"dataframe: {self.dataframe.shape}")
        if self.blob is not None:
            fields.append(f"blob: {len(self.blob)} bytes")
        fields_str = ", ".join(fields)
        return f"{self.__class__.__name__}(id={self.id}, {fields_str})"

    def __eq__(self, other):
        """
        Compares documents for equality. Uses the id to check whether the documents are supposed to be the same.
        """
        if type(self) == type(other):
            return self.id == other.id
        return False

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
        text = self.text or None
        dataframe = self.dataframe.to_json() if self.dataframe is not None else None
        blob = self.blob or None
        mime_type = self.mime_type or None
        metadata = self.metadata or {}
        embedding = self.embedding if self.embedding is not None else None
        data = f"{text}{dataframe}{blob}{mime_type}{metadata}{embedding}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def to_dict(self, flatten=True) -> Dict[str, Any]:
        """
        Converts Document into a dictionary.
        `dataframe` and `blob` fields are converted to JSON-serializable types.

        :param flatten: Whether to flatten `metadata` field or not. Defaults to `True` to be backward-compatible with Haystack 1.x.
        """
        data = asdict(self)
        if (dataframe := data.get("dataframe")) is not None:
            data["dataframe"] = dataframe.to_json()
        if (blob := data.get("blob")) is not None:
            data["blob"] = list(blob)

        if flatten:
            return {**data, **data.pop("metadata")}

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """
        Creates a new Document object from a dictionary.
        `dataframe` and `blob` fields are converted to their original types.
        """
        if (dataframe := data.get("dataframe")) is not None:
            data["dataframe"] = pandas.read_json(dataframe)
        if blob := data.get("blob"):
            data["blob"] = bytes(blob)
        return cls(**data)

    @property
    def content_type(self):
        """
        Returns the type of the content for the document.
        This is necessary to keep backward compatibility with 1.x.
        A ValueError will be raised if both `text` and `dataframe` fields are set
        or both are missing.
        """
        if self.text is not None and self.dataframe is not None:
            raise ValueError("Both text and dataframe are set.")

        if self.text is not None:
            return "text"
        elif self.dataframe is not None:
            return "table"
        raise ValueError("Neither text nor dataframe is set.")
