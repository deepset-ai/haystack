from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from haystack import Answer, Document
from pydantic import BaseConfig

BaseConfig.arbitrary_types_allowed = True


class QueryRequest(BaseModel):
    query: str
    params: Optional[dict] = None


class FilterRequest(BaseModel):
    filters: Optional[Dict[str, Optional[Union[str, List[str]]]]] = None


class QueryResponse(BaseModel):
    query: str
    answers: List[Answer]
    documents: Optional[List[Document]]

