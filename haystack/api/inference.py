from pathlib import Path
from fastapi import FastAPI

import logging

from haystack import Finder
from haystack.database import app
from haystack.reader.farm import FARMReader
from haystack.retriever.tfidf import TfidfRetriever

from pydantic import BaseModel
from typing import List, Dict
import uvicorn

logger = logging.getLogger(__name__)

#TODO Enable CORS

MODELS_DIRS = ["saved_models", "models", "model"]
USE_GPU = False
BATCH_SIZE = 16

app = FastAPI(title="Haystack API", version="0.1")

#############################################
# Load all models in memory
#############################################
model_paths = []
for model_dir in MODELS_DIRS:
    path = Path(model_dir)
    if path.is_dir():
        models = [f for f in path.iterdir() if f.is_dir()]
        model_paths.extend(models)

if len(model_paths) == 0:
    logger.error(f"Could not find any model to load. Checked folders: {MODELS_DIRS}")

retriever = TfidfRetriever()
FINDERS = {}
for idx, model_dir in enumerate(model_paths, start=1):
    reader = FARMReader(model_dir=str(model_dir), batch_size=BATCH_SIZE, use_gpu=USE_GPU)
    FINDERS[idx] = Finder(reader, retriever)
    logger.info(f"Initialized Finder (ID={idx}) with model '{model_dir}'")

logger.info("Open http://127.0.0.1:8000/docs to see Swagger API Documentation.")
logger.info(""" Or just try it out directly: curl --request POST --url 'http://127.0.0.1:8000/finders/1/ask' --data '{"question": "Who is the father of Arya Starck?"}'""")

#############################################
# Basic data schema for request & response
#############################################
class Request(BaseModel):
    question: str
    filters: Dict[str, str] = None
    top_k_reader: int = 5
    top_k_retriever: int = 10


class Answer(BaseModel):
    answer: str
    score: float = None
    probability: float = None
    context: str
    offset_start: int
    offset_end: int
    document_id: str = None


class Response(BaseModel):
    question: str
    answers: List[Answer]

#############################################
# Endpoints
#############################################
@app.post("/finders/{finder_id}/ask", response_model=Response, response_model_exclude_unset=True)
def ask(finder_id: int, request: Request):
    finder = FINDERS.get(finder_id, None)
    if not finder:
        return "Model not found", 404

    results = finder.get_answers(
        question=request.question, top_k_retriever=request.top_k_retriever,
        top_k_reader=request.top_k_reader, filters=request.filters
    )

    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)