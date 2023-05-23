from typing import List, Any, Dict, Literal, Optional, Type

import json
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field, fields, asdict

import numpy
import pandas


logger = logging.getLogger(__name__)

ContentType = Literal["text", "table", "image", "audio"]

PYTHON_TYPES_FOR_CONTENT: Dict[ContentType, type] = {
    "text": str,
    "table": pandas.DataFrame,
    "image": Path,
    "audio": Path,
}

EQUALS_BY_TYPE = {
    Path: lambda self, other: self.absolute() == other.absolute(),
    numpy.ndarray: lambda self, other: self.shape == other.shape and (self == other).all(),
    pandas.DataFrame: lambda self, other: self.equals(other),
}


def _create_id(
    classname: str, content: Any, metadata: Optional[Dict[str, Any]] = None, id_hash_keys: Optional[List[str]] = None
):
    """
    Creates a hash of the content given that acts as the document's ID.
    """
    if not metadata:
        metadata = {}
    content_to_hash = f"{classname}:{content}"
    if id_hash_keys:
        content_to_hash = ":".join([content_to_hash, *[str(metadata.get(key, "")) for key in id_hash_keys]])
    return hashlib.sha256(str(content_to_hash).encode("utf-8")).hexdigest()


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

    for type_, equals in EQUALS_BY_TYPE.items():
        if isinstance(obj_1, type_):
            return equals(obj_1, obj_2)

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
        # Decode content types
        if "content_type" in dictionary:
            if dictionary["content_type"] == "table":
                dictionary["content"] = pandas.read_json(dictionary.get("content", None))
            elif dictionary["content_type"] == "image":
                dictionary["content"] = Path(dictionary.get("content", None))

        # Decode embeddings
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
    the ID should change.  To avoid keeping IDs in sync with the content by using properties, and asking docstores to
    be aware of this corner case, we decide to make Documents immutable and remove the issue. If you need to modify a
    Document, consider using `to_dict()`, modifying the dict, and then create a new Document object using
    `Document.from_dict()`.

    Note that `id_hash_keys` are referring to keys in the metadata. `content` is always included in the ID hash.
    In case of file-based documents (images, audios), the content that is hashed is the file paths,
    so if the file is moved, the hash is different, but if the file is modified without renaming it, the has will
    not differ.
    """

    id: str = field(default_factory=str)
    content: Any = field(default_factory=lambda: None)
    content_type: ContentType = "text"
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False)
    id_hash_keys: List[str] = field(default_factory=list, hash=False)
    score: Optional[float] = field(default=None, compare=True)
    embedding: Optional[numpy.ndarray] = field(default=None, repr=False)

    def __str__(self):
        return f"{self.__class__.__name__}('{self.content}')"

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
        Generate the ID based on the init parameters and make sure that `content_type` matches the actual type of
        content.
        """
        # Validate content_type
        if self.content_type not in PYTHON_TYPES_FOR_CONTENT:
            raise ValueError(
                f"Content type unknown: '{self.content_type}'. "
                f"Choose among: {', '.join(PYTHON_TYPES_FOR_CONTENT.keys())}"
            )
        if not isinstance(self.content, PYTHON_TYPES_FOR_CONTENT[self.content_type]):
            raise ValueError(
                f"The type of content ({type(self.content)}) does not match the "
                f"content type: '{self.content_type}' expects '{PYTHON_TYPES_FOR_CONTENT[self.content_type]}'."
            )
        # Validate metadata
        for key in self.metadata:
            if key in [field.name for field in fields(self)]:
                raise ValueError(f"Cannot name metadata fields as top-level document fields, like '{key}'.")

        # Check if id_hash_keys are all present in the meta
        for key in self.id_hash_keys:
            if key not in self.metadata:
                raise ValueError(
                    f"'{key}' must be present in the metadata of the Document if you want to use it to generate the ID."
                )
        # Generate the ID
        hashed_content = _create_id(
            classname=self.__class__.__name__,
            content=str(self.content),
            metadata=self.metadata,
            id_hash_keys=self.id_hash_keys,
        )

        # Note: we need to set the id this way because the dataclass is frozen. See the docstring.
        object.__setattr__(self, "id", hashed_content)

    def to_dict(self):
        """
        Saves the Document into a dictionary.
        """
        return asdict(self)

    def to_json(self, json_encoder: Optional[Type[DocumentEncoder]] = None, **json_kwargs):
        """
        Saves the Document into a JSON string that can be later loaded back.
        """
        return json.dumps(self.to_dict(), cls=json_encoder or DocumentEncoder, **json_kwargs)

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
