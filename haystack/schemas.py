from pydantic import BaseModel
from typing import List, Optional

class answer(BaseModel):
    text: str
    answer_start: int

class qas(BaseModel):
    question: str
    answers: List[answer]
    is_impossible: Optional[bool] = False
    id: Optional[str] = None

class paragraphs(BaseModel):
    qas: List[qas]
    context: str
    passage_id: Optional[int] = ...

class data(BaseModel):
    title: str
    paragraphs: List[paragraphs]

class SquadSchema(BaseModel):
    version: Optional[str] = None
    data: List[data]