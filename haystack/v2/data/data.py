from typing import List, Any, Dict, Literal

from math import inf
from pathlib import Path
import logging
import json
from dataclasses import asdict, dataclass, field

import mmh3


logger = logging.getLogger(__name__)


#: List of all `content_type` supported
ContentTypes = Literal["text", "table", "image", "audio"]


@dataclass(frozen=True, kw_only=True, eq=True)
class Data:
    """
    Base data class containing some data.

    Can contain text snippets, tables, file paths to images and audio files.
    Please use the subclasses for proper typing.

    Data objects can be serialized to/from dictionary and JSON and are immutable.

    id_hash_keys are referring to keys in the meta.
    """

    id: str = field(default_factory=str)
    content: Any = field(default_factory=lambda: None)
    content_type: ContentTypes
    meta: Dict[str, Any] = field(default_factory=dict, hash=False)
    id_hash_keys: List[str] = field(default_factory=lambda: ["content"], hash=False)

    def __getattr__(self, name):
        # This is here to make Data "abstract".
        # Plus, self.content is needed anyway.
        if name == "content":
            raise NotImplementedError("Use any of the Data subclasses, like TextDocument or TextQuery.")

    def __str__(self):
        return f"{self.__class__.__name__}('{self.content}')"

    def __post_init__(self):
        """
        Generate the ID based on the init parameters.
        """
        content_to_hash = ":".join(
            [self.__class__.__name__, self.content, *[str(self.meta.get(key, "")) for key in self.id_hash_keys]]
        )
        hashed_content = "{:02x}".format(mmh3.hash128(content_to_hash, signed=False))
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


@dataclass(frozen=True, kw_only=True)
class TextData(Data):
    content: str
    content_type: ContentTypes = "text"


@dataclass(frozen=True, kw_only=True)
class TableData(Data):
    content: "pd.DataFrame"
    content_type: ContentTypes = "table"


@dataclass(frozen=True, kw_only=True)
class ImageData(Data):
    content: Path
    content_type: ContentTypes = "image"


@dataclass(frozen=True, kw_only=True)
class AudioData(Data):
    content: Path
    content_type: ContentTypes = "audio"
