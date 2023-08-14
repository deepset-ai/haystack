from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from haystack.preview.dataclasses.document import Document


@dataclass
class Answer:
    data: Any
    question: str
    metadata: Dict[str, Any]


@dataclass
class ExtractiveAnswer(Answer):
    data: Optional[str]
    document: Document
    probability: float
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class GenerativeAnswer(Answer):
    data: str
    documents: List[Document]
    probability: float
