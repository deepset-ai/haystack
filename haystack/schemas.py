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

class context_passage(BaseModel):
    title: str
    text: str
    passage_id: str
    score: Optional[float] = 0
    title_score: Optional[float] = 0

class retriever_training_sample(BaseModel):
    question: str
    answers: List[str]
    positive_ctxs: List[context_passage]
    negative_ctxs: List[context_passage]
    hard_negative_ctxs: List[context_passage]

class retriever_json(BaseModel):
    data: List[retriever_training_sample]
