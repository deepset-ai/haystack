import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import pandas

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    Base data class containing some data to be queried.
    Can contain text snippets, tables, and file paths to images or audios.
    Documents can be sorted by score and saved to/from dictionary and JSON.

    :param id: Unique identifier for the document. When not set, it's generated based on the Document fields' values.
    :param text: Text of the document, if the document contains text.
    :param dataframe: Pandas dataframe with the document's content, if the document contains tabular data.
    :param blob: Binary data associated with the document, if the document has any binary data associated with it.
    :param mime_type: MIME type of the document. Defaults to "text/plain".
    :param metadata: Additional custom metadata for the document. Must be JSON serializable.
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
        `dataframe` and `blob` fields are converted JSON serialisable types.

        :param flatten: Whether to flatten `metadata` field or not. Defaults to True.
        """
        data = asdict(self)
        if (dataframe := data.get("dataframe", None)) is not None:
            data["dataframe"] = dataframe.to_json()
        if blob := data.get("blob", None):
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
        if (dataframe := data.get("dataframe", None)) is not None:
            data["dataframe"] = pandas.read_json(dataframe)
        if blob := data.get("blob", None):
            data["blob"] = bytes(blob)
        return cls(**data)
