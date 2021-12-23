from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, Extra
from haystack.schema import Answer, Document, Label, Span
from pydantic import BaseConfig
from pydantic.dataclasses import dataclass as pydantic_dataclass

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal #type: ignore

BaseConfig.arbitrary_types_allowed = True

class QueryRequest(BaseModel):
    query: str
    params: Optional[dict] = None
    debug: Optional[bool] = False
    class Config:
        # Forbid any extra fields in the request to avoid silent failures
        extra = Extra.forbid

class FilterRequest(BaseModel):
    filters: Optional[Dict[str, Optional[Union[str, List[str]]]]] = None



@pydantic_dataclass
class AnswerSerialized(Answer):
    context: Optional[str] = None

@pydantic_dataclass
class DocumentSerialized(Document):
    content: str
    embedding: Optional[List[float]]

@pydantic_dataclass
class LabelSerialized(Label):
    document: DocumentSerialized
    answer: Optional[AnswerSerialized] = None


class QueryResponse(BaseModel):
    query: str
    answers: List[AnswerSerialized]
    documents: Optional[List[DocumentSerialized]]
    debug: Optional[Dict] = Field(None, alias="_debug")

