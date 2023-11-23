from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from haystack.preview.dataclasses.document import Document


@dataclass(frozen=True)
class Answer:
    data: Any
    query: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class ExtractedAnswer(Answer):
    data: Optional[str]
    document: Optional[Document]
    probability: float
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass(frozen=True)
class GeneratedAnswer(Answer):
    data: str
    documents: List[Document]
