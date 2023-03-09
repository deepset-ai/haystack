from typing import Optional

from math import inf
import logging
from dataclasses import dataclass

import numpy as np

from haystack.v2.data.data import Data, TextData, TableData, ImageData, AudioData
from haystack.v2.data.span import Span


logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class Answer(Data):
    """
    Base data class containing an answer, its score and its metadata.

    Can contain text snippets, tables, file paths to images and audio files.
    Please use the subclasses for proper typing.

    Answers can be sorted by score, serialized to/from dictionary and JSON and are immutable.

    id_hash_keys are referring to keys in the meta.
    """

    score: Optional[float] = None
    context: Optional[str] = None
    offset_in_document: Optional[Span] = None
    offset_in_context: Optional[Span] = None
    document_id: Optional[str] = None

    def __lt__(self, other):
        if not hasattr(other, "score"):
            raise ValueError("Answers can only be compared with other Answers.")
        return (self.score if self.score is not None else -inf) < (other.score if other.score is not None else -inf)

    def __le__(self, other):
        if not hasattr(other, "score"):
            raise ValueError("Answers can only be compared with other Answers.")
        return (self.score if self.score is not None else -inf) <= (other.score if other.score is not None else -inf)


@dataclass(frozen=True, kw_only=True)
class TextAnswer(TextData, Answer):
    pass


@dataclass(frozen=True, kw_only=True)
class TableAnswer(TableData, Answer):
    pass


@dataclass(frozen=True, kw_only=True)
class ImageAnswer(ImageData, Answer):
    pass


@dataclass(frozen=True, kw_only=True)
class AudioAnswer(AudioData, Answer):
    pass
