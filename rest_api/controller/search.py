import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import elasticapm
from fastapi import APIRouter
from fastapi import HTTPException

from haystack import Finder
from rest_api.config import DB_HOST, DB_PORT, DB_USER, DB_PW, DB_INDEX, DB_INDEX_FEEDBACK, DEFAULT_TOP_K_READER, \
    ES_CONN_SCHEME, TEXT_FIELD_NAME, SEARCH_FIELD_NAME, EMBEDDING_DIM, EMBEDDING_FIELD_NAME, EXCLUDE_META_DATA_FIELDS, \
    RETRIEVER_TYPE, EMBEDDING_MODEL_PATH, USE_GPU, READER_MODEL_PATH, BATCHSIZE, CONTEXT_WINDOW_SIZE, \
    TOP_K_PER_CANDIDATE, NO_ANS_BOOST, READER_CAN_HAVE_NO_ANSWER, MAX_PROCESSES, MAX_SEQ_LEN, DOC_STRIDE, \
    CONCURRENT_REQUEST_PER_WORKER, FAQ_QUESTION_FIELD_NAME, EMBEDDING_MODEL_FORMAT, READER_TYPE, READER_TOKENIZER, \
    GPU_NUMBER, NAME_FIELD_NAME, VECTOR_SIMILARITY_METRIC, CREATE_INDEX, LOG_LEVEL, UPDATE_EXISTING_DOCUMENTS

from rest_api.controller.request import Question
from rest_api.controller.response import Answers, AnswersToIndividualQuestion

from rest_api.controller.utils import RequestLimiter
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.reader.base import BaseReader
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.retriever.base import BaseRetriever
from haystack.retriever.sparse import ElasticsearchRetriever, ElasticsearchFilterOnlyRetriever
from haystack.retriever.dense import EmbeddingRetriever

logger = logging.getLogger('haystack')
logger.setLevel(LOG_LEVEL)

router = APIRouter()

# Init global components: DocumentStore, Retriever, Reader, Finder
document_store = ElasticsearchDocumentStore(
    host=DB_HOST,
    port=DB_PORT,
    username=DB_USER,
    password=DB_PW,
    index=DB_INDEX,
    label_index=DB_INDEX_FEEDBACK,
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
    create_index=CREATE_INDEX,
    update_existing_documents=UPDATE_EXISTING_DOCUMENTS,
    similarity=VECTOR_SIMILARITY_METRIC
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
            model_name_or_path=READER_MODEL_PATH,
            use_gpu=use_gpu,
            context_window_size=CONTEXT_WINDOW_SIZE,
            return_no_answers=READER_CAN_HAVE_NO_ANSWER,
            tokenizer=READER_TOKENIZER
        )  # type: Optional[BaseReader]
    elif READER_TYPE == "FARMReader":
        reader = FARMReader(
            model_name_or_path=READER_MODEL_PATH,
            batch_size=BATCHSIZE,
            use_gpu=USE_GPU,
            context_window_size=CONTEXT_WINDOW_SIZE,
            top_k_per_candidate=TOP_K_PER_CANDIDATE,
            no_ans_boost=NO_ANS_BOOST,
            num_processes=MAX_PROCESSES,
            max_seq_len=MAX_SEQ_LEN,
            doc_stride=DOC_STRIDE,
        )  # type: Optional[BaseReader]
    else:
        raise ValueError(f"Could not load Reader of type '{READER_TYPE}'. "
                         f"Please adjust READER_TYPE to one of: "
                         f"'FARMReader', 'TransformersReader', None"
                         )
else:
    reader = None  # don't need one for pure FAQ matching

FINDERS = {1: Finder(reader=reader, retriever=retriever)}


#############################################
# Endpoints
#############################################
doc_qa_limiter = RequestLimiter(CONCURRENT_REQUEST_PER_WORKER)


@router.post("/models/{model_id}/doc-qa", response_model=Answers, response_model_exclude_unset=True)
def doc_qa(model_id: int, question_request: Question):
    with doc_qa_limiter.run():
        start_time = time.time()
        finder = FINDERS.get(model_id, None)
        if not finder:
            raise HTTPException(
                status_code=404, detail=f"Could not get Finder with ID {model_id}. Available IDs: {list(FINDERS.keys())}"
            )

        results = search_documents(finder, question_request, start_time)

        return {"results": results}


@router.post("/models/{model_id}/faq-qa", response_model=Answers, response_model_exclude_unset=True)
def faq_qa(model_id: int, request: Question):
    finder = FINDERS.get(model_id, None)
    if not finder:
        raise HTTPException(
            status_code=404, detail=f"Could not get Finder with ID {model_id}. Available IDs: {list(FINDERS.keys())}"
        )

    results = []
    for question in request.questions:
        if request.filters:
            # put filter values into a list and remove filters with null value
            filters = {}
            for key, values in request.filters.items():
                if values is None:
                    continue
                if not isinstance(values, list):
                    values = [values]
                filters[key] = values
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


@router.post("/models/{model_id}/query", response_model=Dict[str, Any], response_model_exclude_unset=True)
def query(model_id: int, query_request: Dict[str, Any], top_k_reader: int = DEFAULT_TOP_K_READER):
    with doc_qa_limiter.run():
        start_time = time.time()
        finder = FINDERS.get(model_id, None)
        if not finder:
            raise HTTPException(
                status_code=404, detail=f"Could not get Finder with ID {model_id}. Available IDs: {list(FINDERS.keys())}"
            )

        question_request = Question.from_elastic_query_dsl(query_request, top_k_reader)

        answers = search_documents(finder, question_request, start_time)
        response: Dict[str, Any] = {}
        if answers and len(answers) > 0:
            response = AnswersToIndividualQuestion.to_elastic_response_dsl(dict(answers[0]))

        return response


def search_documents(finder, question_request, start_time) -> List[AnswersToIndividualQuestion]:
    results = []
    for question in question_request.questions:
        if question_request.filters:
            # put filter values into a list and remove filters with null value
            filters = {}
            for key, values in question_request.filters.items():
                if values is None:
                    continue
                if not isinstance(values, list):
                    values = [values]
                filters[key] = values
            logger.info(f" [{datetime.now()}] Request: {question_request}")
        else:
            filters = {}

        result = finder.get_answers(
            question=question,
            top_k_retriever=question_request.top_k_retriever,
            top_k_reader=question_request.top_k_reader,
            filters=filters,
        )
        results.append(result)
    elasticapm.set_custom_context({"results": results})
    end_time = time.time()
    logger.info(
        json.dumps({"request": question_request.dict(), "results": results,
                    "time": f"{(end_time - start_time):.2f}"}))
    return results
