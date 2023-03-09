from typing import Optional

from math import inf
import logging
from dataclasses import dataclass, field

import numpy as np

from haystack.v2.data.data import Data, TextData, TableData, ImageData, AudioData


logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Document(Data):
    """
    Base data class containing some data to be queried.

    Can contain text snippets, tables, file paths to images and audio files.
    Please use the subclasses for proper typing.

    Documents can be sorted by score, serialized to/from dictionary and JSON,
    and are immutable.

    id_hash_keys are referring to keys in the meta.
    """

    score: Optional[float] = None
    embedding: Optional[np.ndarray] = field(default=lambda: None, repr=False)

    def __lt__(self, other):
        if not hasattr(other, "score"):
            raise ValueError("Documents can only be compared with other Documents.")
        return (self.score if self.score is not None else -inf) < (other.score if other.score is not None else -inf)

    def __le__(self, other):
        if not hasattr(other, "score"):
            raise ValueError("Documents can only be compared with other Documents.")
        return (self.score if self.score is not None else -inf) <= (other.score if other.score is not None else -inf)


@dataclass(frozen=True, kw_only=True)
class TextDocument(TextData, Document):
    pass


@dataclass(frozen=True, kw_only=True)
class TableDocument(TableData, Document):
    pass


@dataclass(frozen=True, kw_only=True)
class ImageDocument(ImageData, Document):
    pass


@dataclass(frozen=True, kw_only=True)
class AudioDocument(AudioData, Document):
    pass
