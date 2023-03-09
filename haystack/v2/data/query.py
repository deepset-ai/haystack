from typing import Optional

import logging
from dataclasses import dataclass, field

import numpy as np

from haystack.v2.data.data import Data, TextData, TableData, ImageData, AudioData


logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Query(Data):
    """
    Base data class containing a query and in some cases an embedding.

    Can contain text snippets, tables, file paths to images and audio files.
    Please use the subclasses for proper typing.

    Queries can be serialized to/from dictionary and JSON and are immutable.

    id_hash_keys are referring to keys in the meta.
    """

    embedding: Optional[np.ndarray] = field(default=lambda: None, repr=False)


@dataclass(frozen=True, kw_only=True)
class TextQuery(TextData, Query):
    pass


@dataclass(frozen=True, kw_only=True)
class TableQuery(TableData, Query):
    pass


@dataclass(frozen=True, kw_only=True)
class ImageQuery(ImageData, Query):
    pass


@dataclass(frozen=True, kw_only=True)
class AudioQuery(AudioData, Query):
    pass
