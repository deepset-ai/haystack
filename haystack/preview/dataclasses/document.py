from typing import List, Any, Dict, Optional, Type

import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field, fields, asdict

import numpy
import pandas


logger = logging.getLogger(__name__)


def _safe_equals(obj_1, obj_2) -> bool:
    """
    Compares two dictionaries for equality, taking arrays, dataframes and other objects into account.
    """
    if type(obj_1) != type(obj_2):
        return False

    if isinstance(obj_1, dict):
        if obj_1.keys() != obj_2.keys():
            return False
        return all(_safe_equals(obj_1[key], obj_2[key]) for key in obj_1)

    if isinstance(obj_1, Path):
        return obj_1.absolute() == obj_2.absolute()
    if isinstance(obj_1, numpy.ndarray):
        return obj_1.shape == obj_2.shape and (obj_1 == obj_2).all()
    if isinstance(obj_1, pandas.DataFrame):
        return obj_1.equals(obj_2)
    return obj_1 == obj_2


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
        if "array" in dictionary and dictionary.get("array"):
            dictionary["array"] = numpy.array(dictionary.get("array"))
        if "dataframe" in dictionary and dictionary.get("dataframe"):
            dictionary["dataframe"] = pandas.read_json(dictionary.get("dataframe", None))
        if "embedding" in dictionary and dictionary.get("embedding"):
            dictionary["embedding"] = numpy.array(dictionary.get("embedding"))

        return dictionary


@dataclass(frozen=True)
class Document:
    """
    Base data class containing some data to be queried.
    Can contain text snippets, tables, and file paths to images or audios.
    Documents can be sorted by score, saved to/from dictionary and JSON, and are immutable.

    Immutability is due to the fact that the document's ID depends on its content, so upon changing the content, also
    the ID should change. To avoid keeping IDs in sync with the content by using properties, and asking docstores to
    be aware of this corner case, we decide to make Documents immutable and remove the issue. If you need to modify a
    Document, consider using `to_dict()`, modifying the dict, and then create a new Document object using
    `Document.from_dict()`.

    Note that `id_hash_keys` are referring to keys either in the metadata or in the document itself.
    """

    id: str = field(default_factory=str)
    text: Optional[str] = field(default=None)
    array: Optional[numpy.ndarray] = field(default=None)
    dataframe: Optional[pandas.DataFrame] = field(default=None)
    blob: Optional[bytes] = field(default=None)
    mime_type: str = field(default="text/plain")
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False)
    id_hash_keys: List[str] = field(default_factory=lambda: ["text", "array", "dataframe", "blob"], hash=False)
    score: Optional[float] = field(default=None, compare=True)
    embedding: Optional[numpy.ndarray] = field(default=None, repr=False)

    def __str__(self):
        if self.text:
            return f"{self.__class__.__name__}(mimetype: {self.mime_type}, text: '{self.text}')"
        if self.array:
            return f"{self.__class__.__name__}(mimetype: {self.mime_type}, array: '{self.array}')"
        if self.dataframe:
            return f"{self.__class__.__name__}(mimetype: {self.mime_type}, dataframe: '{self.dataframe}')"
        if self.blob:
            return f"{self.__class__.__name__}(mimetype: {self.mime_type}, binary only)"
        return f"{self.__class__.__name__}(mimetype: {self.mime_type}, no content)"

    def __eq__(self, other):
        """
        Compares documents for equality. Compares `embedding` properly and checks the metadata taking care of
        embeddings, paths, dataframes, nested dictionaries and other objects.
        """
        if type(self) == type(other):
            return _safe_equals(self.to_dict(), other.to_dict())
        return False

    def __post_init__(self):
        """
        Generate the ID based on the init parameters.
        """
        # Validate metadata
        for key in self.metadata:
            if key in [field.name for field in fields(self)]:
                raise ValueError(f"Cannot name metadata fields as top-level document fields, like '{key}'.")

        # Note: we need to set the id this way because the dataclass is frozen. See the docstring.
        hashed_content = self._create_id()
        object.__setattr__(self, "id", hashed_content)

    def _create_id(self):
        """
        Creates a hash of the content given that acts as the document's ID.
        """
        document_data = self.flatten()
        contents = [self.__class__.__name__]
        if self.id_hash_keys:
            for key in self.id_hash_keys:
                if key not in document_data:
                    logger.info(f"ID hash key '{key}' not found in document.")
                else:
                    contents.append(str(document_data.get(key, None)))
            content_to_hash = ":".join(contents)
        return hashlib.sha256(str(content_to_hash).encode("utf-8")).hexdigest()

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
