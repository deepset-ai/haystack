from typing import List, Any, Dict, Literal, Optional, TYPE_CHECKING

import json
import hashlib
import logging
from pathlib import Path
from dataclasses import asdict, dataclass, field

from haystack.preview.utils.import_utils import optional_import

# We need to do this dance because ndarray is an optional dependency used as a type by dataclass
if TYPE_CHECKING:
    from numpy import ndarray
else:
    ndarray = optional_import("numpy", "ndarray", "You won't be able to use embeddings.", __name__)

DataFrame = optional_import("pandas", "DataFrame", "You won't be able to use table related features.", __name__)


logger = logging.getLogger(__name__)
ContentType = Literal["text", "table", "image", "audio"]
PYTHON_TYPES_FOR_CONTENT: Dict[ContentType, type] = {"text": str, "table": DataFrame, "image": Path, "audio": Path}


def _create_id(
    classname: str, content: Any, metadata: Optional[Dict[str, Any]] = None, id_hash_keys: Optional[List[str]] = None
):
    """
    Creates a hash of the content given that acts as the document's ID.
    """
    content_to_hash = f"{classname}:{content}"
    if id_hash_keys:
        if not metadata:
            raise ValueError("If 'id_hash_keys' is provided, you must provide 'metadata' too.")
        content_to_hash = ":".join([content_to_hash, *[str(metadata.get(key, "")) for key in id_hash_keys]])
    return hashlib.sha256(str(content_to_hash).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Document:
    """
    Base data class containing some data to be queried.
    Can contain text snippets, tables, file paths to files like images or audios.
    Documents can be sorted by score, serialized to/from dictionary and JSON, and are immutable.

    Immutability is due to the fact that the document's ID depends on its content, so upon changing the content, also
    the ID should change.  To avoid keeping IDs in sync with the content by using properties, and asking docstores to
    be aware of this corner case, we decide to make Documents immutable and remove the issue. If you need to modify a
    Document, consider using `to_dict()`, modifying the dict, and then create a new Document object using
    `Document.from_dict()`.

    Note that `id_hash_keys` are referring to keys in the metadata. `content` is always included in the id hash.
    In case of file-based documents (images, audios), the content that is hashed is the file paths,
    so if the file is moved, the hash is different, but if the file is modified without renaming it, the has will
    not differ.
    """

    id: str = field(default_factory=str)
    content: Any = field(default_factory=lambda: None)
    content_type: ContentType = "text"
    metadata: Dict[str, Any] = field(default_factory=dict, hash=False)
    id_hash_keys: List[str] = field(default_factory=lambda: [], hash=False)
    score: Optional[float] = field(default=None, compare=True)
    embedding: Optional[ndarray] = field(default=None, repr=False)

    def __str__(self):
        return f"{self.__class__.__name__}('{self.content}')"

    def __post_init__(self):
        """
        Generate the ID based on the init parameters and make sure that content_type
        matches the actual type of content.
        """
        # Validate content_type
        if not isinstance(self.content, PYTHON_TYPES_FOR_CONTENT[self.content_type]):
            raise ValueError(
                f"The type of content ({type(self.content)}) does not match the "
                f"content type: '{self.content_type}' expects '{PYTHON_TYPES_FOR_CONTENT[self.content_type]}'."
            )
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
        return asdict(self)

    def to_json(self, **json_kwargs):
        return json.dumps(self.to_dict(), *json_kwargs)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(**dictionary)

    @classmethod
    def from_json(cls, data, **json_kwargs):
        dictionary = json.loads(data, **json_kwargs)
        return cls.from_dict(dictionary=dictionary)
