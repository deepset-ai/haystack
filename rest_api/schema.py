from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str
    params: Optional[dict] = None


class FilterRequest(BaseModel):
    filters: Optional[Dict[str, Optional[Union[str, List[str]]]]] = None


class QueryAnswer(BaseModel):
    answer: Optional[str]
    question: Optional[str]
    score: Optional[float] = None
    probability: Optional[float] = None
    context: Optional[str]
    offset_start: Optional[int]
    offset_end: Optional[int]
    offset_start_in_doc: Optional[int]
    offset_end_in_doc: Optional[int]
    document_id: Optional[str] = None
    meta: Optional[Dict[str, Any]]


class QueryResponse(BaseModel):
    query: str
    answers: List[QueryAnswer]


class ExtractiveQAFeedback(BaseModel):
    question: str = Field(..., description="The question input by the user, i.e., the query.")
    is_correct_answer: bool = Field(..., description="Whether the answer is correct or not.")
    document_id: str = Field(..., description="The document in the query result for which feedback is given.")
    model_id: Optional[int] = Field(None, description="The model used for the query.")
    is_correct_document: bool = Field(
        ...,
        description="In case of negative feedback, there could be two cases; incorrect answer but correct "
        "document & incorrect document. This flag denotes if the returned document was correct.",
    )
    answer: str = Field(..., description="The answer string.")
    offset_start_in_doc: int = Field(
        ..., description="The answer start offset in the original doc. Only required for doc-qa feedback."
    )
