import logging
from datetime import datetime
from typing import List, Dict, Optional

import elasticapm
from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel

from haystack import Finder
from haystack.api.config import DB_HOST, DB_USER, DB_PW, DB_INDEX, ES_CONN_SCHEME, TEXT_FIELD_NAME, SEARCH_FIELD_NAME, \
    EMBEDDING_DIM, EMBEDDING_FIELD_NAME, EXCLUDE_META_DATA_FIELDS, EMBEDDING_MODEL_PATH, USE_GPU, READER_MODEL_PATH, \
    BATCHSIZE, CONTEXT_WINDOW_SIZE, TOP_K_PER_CANDIDATE, NO_ANS_BOOST, MAX_PROCESSES, MAX_SEQ_LEN, DOC_STRIDE, \
    DEFAULT_TOP_K_READER, DEFAULT_TOP_K_RETRIEVER, CONCURRENT_REQUEST_PER_WORKER
from haystack.api.controller.utils import RequestLimiter
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.reader.farm import FARMReader
from haystack.retriever.elasticsearch import ElasticsearchRetriever, EmbeddingRetriever

logger = logging.getLogger(__name__)
router = APIRouter()

# Init global components: DocumentStore, Retriever, Reader, Finder
document_store = ElasticsearchDocumentStore(
    host=DB_HOST,
    username=DB_USER,
    password=DB_PW,
    index=DB_INDEX,
    scheme=ES_CONN_SCHEME,
    ca_certs=False,
    verify_certs=False,
    text_field=TEXT_FIELD_NAME,
    search_fields=SEARCH_FIELD_NAME,
    embedding_dim=EMBEDDING_DIM,
    embedding_field=EMBEDDING_FIELD_NAME,
    excluded_meta_data=EXCLUDE_META_DATA_FIELDS,
)


if EMBEDDING_MODEL_PATH:
    retriever = EmbeddingRetriever(document_store=document_store, embedding_model=EMBEDDING_MODEL_PATH, gpu=USE_GPU)
else:
    retriever = ElasticsearchRetriever(document_store=document_store)

if READER_MODEL_PATH:  # for extractive doc-qa
    reader = FARMReader(
        model_name_or_path=str(READER_MODEL_PATH),
        batch_size=BATCHSIZE,
        use_gpu=USE_GPU,
        context_window_size=CONTEXT_WINDOW_SIZE,
        top_k_per_candidate=TOP_K_PER_CANDIDATE,
        no_ans_boost=NO_ANS_BOOST,
        num_processes=MAX_PROCESSES,
        max_seq_len=MAX_SEQ_LEN,
        doc_stride=DOC_STRIDE,
    )
else:
    reader = None  # don't need one for pure FAQ matching

FINDERS = {1: Finder(reader=reader, retriever=retriever)}


#############################################
# Data schema for request & response
#############################################
class Question(BaseModel):
    questions: List[str]
    filters: Dict[str, Optional[str]] = None
    top_k_reader: int = DEFAULT_TOP_K_READER
    top_k_retriever: int = DEFAULT_TOP_K_RETRIEVER


class Answer(BaseModel):
    answer: Optional[str]
    question: Optional[str]
    score: float = None
    probability: float = None
    context: Optional[str]
    offset_start: int
    offset_end: int
    offset_start_in_doc: Optional[int]
    offset_end_in_doc: Optional[int]
    document_id: Optional[str] = None
    meta: Optional[Dict[str, Optional[str]]]


class AnswersToIndividualQuestion(BaseModel):
    question: str
    answers: List[Optional[Answer]]


class Answers(BaseModel):
    results: List[AnswersToIndividualQuestion]


#############################################
# Endpoints
#############################################
doc_qa_limiter = RequestLimiter(CONCURRENT_REQUEST_PER_WORKER)

@router.post("/models/{model_id}/doc-qa", response_model=Answers, response_model_exclude_unset=True)
def doc_qa(model_id: int, request: Question):
    with doc_qa_limiter.run():
        finder = FINDERS.get(model_id, None)
        if not finder:
            raise HTTPException(
                status_code=404, detail=f"Couldn't get Finder with ID {model_id}. Available IDs: {list(FINDERS.keys())}"
            )

        results = []
        for question in request.questions:
            if request.filters:
                # put filter values into a list and remove filters with null value
                request.filters = {key: [value] for key, value in request.filters.items() if value is not None}
                logger.info(f" [{datetime.now()}] Request: {request}")

            result = finder.get_answers(
                question=question,
                top_k_retriever=request.top_k_retriever,
                top_k_reader=request.top_k_reader,
                filters=request.filters,
            )
            results.append(result)

        elasticapm.set_custom_context({"results": results})
        logger.info({"request": request.json(), "results": results})

        return {"results": results}


@router.post("/models/{model_id}/faq-qa", response_model=Answers, response_model_exclude_unset=True)
def faq_qa(model_id: int, request: Question):
    finder = FINDERS.get(model_id, None)
    if not finder:
        raise HTTPException(
            status_code=404, detail=f"Couldn't get Finder with ID {model_id}. Available IDs: {list(FINDERS.keys())}"
        )

    results = []
    for question in request.questions:
        if request.filters:
            # put filter values into a list and remove filters with null value
            request.filters = {key: [value] for key, value in request.filters.items() if value is not None}
            logger.info(f" [{datetime.now()}] Request: {request}")

        result = finder.get_answers_via_similar_questions(
            question=question, top_k_retriever=request.top_k_retriever, filters=request.filters,
        )
        results.append(result)

    elasticapm.set_custom_context({"results": results})
    logger.info({"request": request.json(), "results": results})

    return {"results": results}
