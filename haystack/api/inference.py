from fastapi import FastAPI, HTTPException

import logging
from datetime import datetime

from haystack import Finder
from haystack.reader.farm import FARMReader
from haystack.retriever.elasticsearch import ElasticsearchRetriever
from haystack.database.elasticsearch import ElasticsearchDocumentStore

from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
import os
import time

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger(__name__)
logging.getLogger('elasticsearch').setLevel(logging.WARNING)


# CONFIG
USE_GPU = True
BATCHSIZE = int(os.getenv("BATCHSIZE", 50))
MODEL_PATH = os.getenv("MODEL_PATH", "models/bert-base-cased-squad2")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "")
DB_PW = os.getenv("DB_PW", "")
DB_INDEX = os.getenv("DB_INDEX", "document")

ES_CONN_SCHEME = os.getenv("ES_CONN_SCHEME", "http")
TEXT_FIELD_NAME = os.getenv("TEXT_FIELD_NAME", "text")

CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", 500))
DEFAULT_TOP_K_READER = int(os.getenv("DEFAULT_TOP_K_READER", 5))
DEFAULT_TOP_K_RETRIEVER = int(os.getenv("DEFAULT_TOP_K_RETRIEVER", 10))
TOP_K_PER_CANDIDATE = int(os.getenv("TOP_K_PER_CANDIDATE", 3))
NO_ANS_BOOST = int(os.getenv("NO_ANS_BOOST", -10))
DOC_STRIDE = int(os.getenv("DOC_STRIDE", 128))
MAX_SEQ_LEN = int(os.getenv("MAX_SEQ_LEN", 256))


MAX_PROCESSES = int(os.getenv("MAX_PROCESSES", 4))

app = FastAPI(title="Haystack API", version="0.1")


if len(MODEL_PATH) == 0:
    logger.error(f"No model to load. Please specify one via MODEL_PATHS (e.g. 'deepset/bert-base-cased-squad2'")


logger.info(f"Try to connect to: DB_HOST={DB_HOST}, DB_USER={len(DB_USER)*'*'}, DB_PW={len(DB_PW)*'*'}, DB_INDEX={DB_INDEX}")

# Init global components: DocumentStore, Retriever, Reader, Finder
datastore = ElasticsearchDocumentStore(host=DB_HOST, username=DB_USER, password=DB_PW, index=DB_INDEX,
                                       scheme=ES_CONN_SCHEME, ca_certs=False, verify_certs=False,
                                       text_field=TEXT_FIELD_NAME, search_fields=TEXT_FIELD_NAME)
retriever = ElasticsearchRetriever(datastore)

reader = FARMReader(model_name_or_path=str(MODEL_PATH),
                    batch_size=BATCHSIZE,
                    use_gpu=USE_GPU,
                    context_window_size=CONTEXT_WINDOW_SIZE,
                    top_k_per_candidate=TOP_K_PER_CANDIDATE,
                    no_ans_boost=NO_ANS_BOOST,
                    max_processes=MAX_PROCESSES,
                    max_seq_len=MAX_SEQ_LEN,
                    doc_stride=DOC_STRIDE)

FINDERS = {1:  Finder(reader, retriever)}
logger.info(f"Initialized Finder (ID=1) with model '{MODEL_PATH}'")

logger.info("Open http://127.0.0.1:8000/docs to see Swagger API Documentation.")
logger.info("""
Or just try it out directly: curl --request POST --url 'http://127.0.0.1:8000/models/1/doc-qa' --data '{"questions": ["Who is the father of Arya Starck?"]}'
""")



#############################################
# Basic data schema for request & response
#############################################
class Request(BaseModel):
    questions: List[str]
    filters: Dict[str, Optional[str]] = None
    top_k_reader: int = DEFAULT_TOP_K_READER
    top_k_retriever: int = DEFAULT_TOP_K_RETRIEVER


class Answer(BaseModel):
    answer: Optional[str]
    score: float = None
    probability: float = None
    context: Optional[str]
    offset_start: int
    offset_end: int
    document_id: Optional[str] = None
    document_name: Optional[str]


class ResponseToIndividualQuestion(BaseModel):
    question: str
    answers: List[Optional[Answer]]

class Response(BaseModel):
    results: List[ResponseToIndividualQuestion]

#############################################
# Endpoints
#############################################
@app.post("/models/{model_id}/doc-qa", response_model=Response, response_model_exclude_unset=True)
def ask(model_id: int, request: Request):
    t1 = time.time()
    finder = FINDERS.get(model_id, None)
    if not finder:
        raise HTTPException(status_code=404, detail=f"Couldn't get Finder with ID {model_id}. Available IDs: {list(FINDERS.keys())}")

    results = []
    for question in request.questions:
        if request.filters:
            # put filter values into a list and remove filters with null value
            request.filters = {key: [value] for key, value in request.filters.items() if value is not None}
            logger.info(f" [{datetime.now()}] Request: {request}")

        result = finder.get_answers(
            question=question, top_k_retriever=request.top_k_retriever,
            top_k_reader=request.top_k_reader, filters=request.filters,
        )
        results.append(result)

        resp_time = round(time.time()-t1, 2)
        resp = {"time": resp_time, "request": request.json(), "results": results}
        logger.info(resp)
        return resp


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)