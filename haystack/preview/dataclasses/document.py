import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy
import pandas

logger = logging.getLogger(__name__)


class DocumentEncoder(json.JSONEncoder):
    """
    Encodes more exotic datatypes like pandas dataframes or file paths.
    """

    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, pandas.DataFrame):
            return obj.to_json()
        if isinstance(obj, Path):
            return str(obj.absolute())
        try:
            return json.JSONEncoder.default(self, obj)
        except TypeError:
            return str(obj)


class DocumentDecoder(json.JSONDecoder):
    """
    Decodes more exotic datatypes like pandas dataframes or file paths.
    """

    def __init__(self, *_, object_hook=None, **__):
        super().__init__(object_hook=object_hook or self.document_decoder)

    def document_decoder(self, dictionary):
        if "dataframe" in dictionary and dictionary.get("dataframe"):
            dictionary["dataframe"] = pandas.read_json(dictionary.get("dataframe", None))

        return dictionary


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
    :param metadata: Additional custom metadata for the document.
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
        # Validate metadata
        for key in self.metadata:
            if key in [field.name for field in fields(self)]:
                raise ValueError(f"Cannot name metadata fields as top-level document fields, like '{key}'.")

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

    def to_dict(self):
        """
        Saves the Document into a dictionary.
        """
        return asdict(self)

    def to_json(self, json_encoder: Optional[Type[DocumentEncoder]] = None, **json_kwargs):
        """
        Saves the Document into a JSON string that can be later loaded back. Drops all binary data from the blob field.
        """
        dictionary = self.to_dict()
        del dictionary["blob"]
        return json.dumps(dictionary, cls=json_encoder or DocumentEncoder, **json_kwargs)

    @classmethod
    def from_dict(cls, dictionary):
        """
        Creates a new Document object from a dictionary of its fields.
        """
        return cls(**dictionary)

    @classmethod
    def from_json(cls, data, json_decoder: Optional[Type[DocumentDecoder]] = None, **json_kwargs):
        """
        Creates a new Document object from a JSON string.
        """
        dictionary = json.loads(data, cls=json_decoder or DocumentDecoder, **json_kwargs)
        return cls.from_dict(dictionary=dictionary)

    def flatten(self) -> Dict[str, Any]:
        """
        Returns a dictionary with all the fields of the document and the metadata on the same level.
        This allows filtering by all document fields, not only the metadata.
        """
        dictionary = self.to_dict()
        metadata = dictionary.pop("metadata", {})
        dictionary = {**dictionary, **metadata}
        return dictionary
