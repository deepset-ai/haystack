from __future__ import annotations

from typing import Dict, List, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

from pydantic import BaseModel, Field, Extra
from pydantic import BaseConfig

from haystack.schema import Answer, Document, Label


BaseConfig.arbitrary_types_allowed = True

PrimitiveType = Union[str, int, float, bool]


class QueryRequest(BaseModel):
    query: str
    params: Optional[dict] = None
    debug: Optional[bool] = False

    class Config:
        # Forbid any extra fields in the request to avoid silent failures
        extra = Extra.forbid


class FilterRequest(BaseModel):
    filters: Optional[Dict[str, Union[PrimitiveType, List[PrimitiveType], Dict[str, PrimitiveType]]]] = None


class AnswerSerialized(Answer):
    context: Optional[str] = None


class DocumentSerialized(Document):
    content: str
    embedding: Optional[List[float]]  # type: ignore


class LabelSerialized(Label, BaseModel):
    document: DocumentSerialized
    answer: Optional[AnswerSerialized] = None


class CreateLabelSerialized(BaseModel):
    id: Optional[str] = None
    query: str
    document: DocumentSerialized
    is_correct_answer: bool
    is_correct_document: bool
    origin: Literal["user-feedback", "gold-label"]
    answer: Optional[AnswerSerialized] = None
    no_answer: Optional[bool] = None
    pipeline_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    meta: Optional[dict] = None
    filters: Optional[dict] = None

    class Config:
        # Forbid any extra fields in the request to avoid silent failures
        extra = Extra.forbid


class QueryResponse(BaseModel):
    query: str
    answers: List[AnswerSerialized] = []
    documents: List[DocumentSerialized] = []
    debug: Optional[Dict] = Field(None, alias="_debug")
