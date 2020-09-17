import json
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional

import elasticapm
from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel

from haystack import Finder
from rest_api.config import DB_HOST, DB_PORT, DB_USER, DB_PW, DB_INDEX, ES_CONN_SCHEME, TEXT_FIELD_NAME, SEARCH_FIELD_NAME, \
    EMBEDDING_DIM, EMBEDDING_FIELD_NAME, EXCLUDE_META_DATA_FIELDS, RETRIEVER_TYPE, EMBEDDING_MODEL_PATH, USE_GPU, READER_MODEL_PATH, \
    BATCHSIZE, CONTEXT_WINDOW_SIZE, TOP_K_PER_CANDIDATE, NO_ANS_BOOST, MAX_PROCESSES, MAX_SEQ_LEN, DOC_STRIDE, \
    DEFAULT_TOP_K_READER, DEFAULT_TOP_K_RETRIEVER, CONCURRENT_REQUEST_PER_WORKER, FAQ_QUESTION_FIELD_NAME, \
    EMBEDDING_MODEL_FORMAT, READER_TYPE, READER_TOKENIZER, GPU_NUMBER, NAME_FIELD_NAME
from rest_api.controller.utils import RequestLimiter
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.retriever.base import BaseRetriever
from haystack.retriever.sparse import ElasticsearchRetriever, ElasticsearchFilterOnlyRetriever
from haystack.retriever.dense import EmbeddingRetriever

logger = logging.getLogger(__name__)
router = APIRouter()

# Init global components: DocumentStore, Retriever, Reader, Finder
document_store = ElasticsearchDocumentStore(
    host=DB_HOST,
    port=DB_PORT,
    username=DB_USER,
    password=DB_PW,
    index=DB_INDEX,
    scheme=ES_CONN_SCHEME,
    ca_certs=False,
    verify_certs=False,
    text_field=TEXT_FIELD_NAME,
    name_field=NAME_FIELD_NAME,
    search_fields=SEARCH_FIELD_NAME,
    embedding_dim=EMBEDDING_DIM,
    embedding_field=EMBEDDING_FIELD_NAME,
    excluded_meta_data=EXCLUDE_META_DATA_FIELDS,  # type: ignore
    faq_question_field=FAQ_QUESTION_FIELD_NAME,
)


if RETRIEVER_TYPE == "EmbeddingRetriever":
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model=EMBEDDING_MODEL_PATH,
        model_format=EMBEDDING_MODEL_FORMAT,
        use_gpu=USE_GPU
    )  # type: BaseRetriever
elif RETRIEVER_TYPE == "ElasticsearchRetriever":
    retriever = ElasticsearchRetriever(document_store=document_store)
elif RETRIEVER_TYPE is None or RETRIEVER_TYPE == "ElasticsearchFilterOnlyRetriever":
    retriever = ElasticsearchFilterOnlyRetriever(document_store=document_store)
else:
    raise ValueError(f"Could not load Retriever of type '{RETRIEVER_TYPE}'. "
                     f"Please adjust RETRIEVER_TYPE to one of: "
                     f"'EmbeddingRetriever', 'ElasticsearchRetriever', 'ElasticsearchFilterOnlyRetriever', None"
                     f"OR modify rest_api/search.py to support your retriever"
                     )


if READER_MODEL_PATH:  # for extractive doc-qa
    if READER_TYPE == "TransformersReader":
        use_gpu = -1 if not USE_GPU else GPU_NUMBER
        reader = TransformersReader(
            model=str(READER_MODEL_PATH),
            use_gpu=use_gpu,
            context_window_size=CONTEXT_WINDOW_SIZE,
            tokenizer=str(READER_TOKENIZER)
        )  # type: Optional[FARMReader]
    elif READER_TYPE == "FARMReader":
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
        )  # type: Optional[FARMReader]
    else:
        raise ValueError(f"Could not load Reader of type '{READER_TYPE}'. "
                         f"Please adjust READER_TYPE to one of: "
                         f"'FARMReader', 'TransformersReader', None"
                         )
else:
    reader = None  # don't need one for pure FAQ matching

FINDERS = {1: Finder(reader=reader, retriever=retriever)}


#############################################
# Data schema for request & response
#############################################
class Question(BaseModel):
    questions: List[str]
    filters: Optional[Dict[str, str]] = None
    top_k_reader: int = DEFAULT_TOP_K_READER
    top_k_retriever: int = DEFAULT_TOP_K_RETRIEVER


class Answer(BaseModel):
    answer: Optional[str]
    question: Optional[str]
    score: Optional[float] = None
    probability: Optional[float] = None
    context: Optional[str]
    offset_start: int
    offset_end: int
    offset_start_in_doc: Optional[int]
    offset_end_in_doc: Optional[int]
    document_id: Optional[str] = None
    meta: Optional[Dict[str, str]]


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
        start_time = time.time()
        finder = FINDERS.get(model_id, None)
        if not finder:
            raise HTTPException(
                status_code=404, detail=f"Couldn't get Finder with ID {model_id}. Available IDs: {list(FINDERS.keys())}"
            )

        results = []
        for question in request.questions:
            if request.filters:
                # put filter values into a list and remove filters with null value
                filters = {key: [value] for key, value in request.filters.items() if value is not None}
                logger.info(f" [{datetime.now()}] Request: {request}")
            else:
                filters = {}

            result = finder.get_answers(
                question=question,
                top_k_retriever=request.top_k_retriever,
                top_k_reader=request.top_k_reader,
                filters=filters,
            )
            results.append(result)

        elasticapm.set_custom_context({"results": results})
        end_time = time.time()
        logger.info(json.dumps({"request": request.dict(), "results": results, "time": f"{(end_time - start_time):.2f}"}))

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
            filters = {key: [value] for key, value in request.filters.items() if value is not None}
            logger.info(f" [{datetime.now()}] Request: {request}")
        else:
            filters = {}

        result = finder.get_answers_via_similar_questions(
            question=question, top_k_retriever=request.top_k_retriever, filters=filters,
        )
        results.append(result)

    elasticapm.set_custom_context({"results": results})
    logger.info(json.dumps({"request": request.dict(), "results": results}))

    return {"results": results}
