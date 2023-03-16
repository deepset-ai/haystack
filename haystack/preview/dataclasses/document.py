from typing import List, Any, Dict, Literal, Optional

import json
import hashlib
import logging
from pathlib import Path
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

try:
    from numpy import ndarray
except ImportError as exc2:
    logging.debug("numpy can't be imported. You won't be able to use embeddings.")
    ndarray = None

try:
    from pandas import DataFrame
except ImportError as exc3:
    logging.debug("pandas can't be imported. You won't be able to use table related features.")
    DataFrame = None


ContentTypes = Literal["text", "table", "image"]


CONTENT_TYPES: Dict[ContentTypes, type] = {"text": str, "table": DataFrame, "image": Path}


@dataclass(frozen=True)
class Document:
    """
    Base data class containing some data to be queried.
    Can contain text snippets, tables, file paths to images.
    Documents can be sorted by score, serialized to/from dictionary and JSON, and are immutable.

    Note that `id_hash_keys` are referring to keys in the metadata: `content` is always included in the id hash.
    """

    id: str = field(default_factory=str)
    content: Any = field(default_factory=lambda: None)
    content_type: ContentTypes = "text"
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
        if not isinstance(self.content, CONTENT_TYPES[self.content_type]):
            raise ValueError(
                f"The type of content ({type(self.content)}) does not match the "
                f"content type: '{self.content_type}' expects '{CONTENT_TYPES[self.content_type]}'."
            )
        # Check if id_hash_keys are all present in the meta
        for key in self.id_hash_keys:
            if key not in self.metadata:
                raise ValueError(
                    f"'{key}' must be present in the metadata of the Document if you want to use it to generate the ID."
                )
        # Generate the ID
        content_to_hash = ":".join(
            [
                self.__class__.__name__,
                str(self.content),
                *[str(self.metadata.get(key, "")) for key in self.id_hash_keys],
            ]
        )
        hashed_content = hashlib.sha256(str(content_to_hash).encode("utf-8")).hexdigest()
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
