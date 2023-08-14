from typing import List, Optional, Union
from dataclasses import dataclass
from haystack.preview.dataclasses.document import Document


@dataclass
class ExtractiveAnswer:
    answer: str
    start: int
    end: int


@dataclass
class GenerativeAnswer:
    answer: str


@dataclass
class Answer:
    answer: Optional[Union[ExtractiveAnswer, GenerativeAnswer]]
    probability: float
    question: str
    documents: List[Document]
