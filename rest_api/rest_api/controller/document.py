from typing import List

import logging

from fastapi import FastAPI, APIRouter
from haystack.document_stores import BaseDocumentStore
from haystack.schema import Document

from rest_api.utils import get_app, get_pipelines
from rest_api.config import LOG_LEVEL
from rest_api.schema import FilterRequest


logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")


router = APIRouter()
app: FastAPI = get_app()
document_store: BaseDocumentStore = get_pipelines().get("document_store", None)


@router.post("/documents/get_by_filters", response_model=List[Document], response_model_exclude_none=True)
def get_documents(filters: FilterRequest):
    """
    This endpoint allows you to retrieve documents contained in your document store.
    You can filter the documents to retrieve by metadata (like the document's name),
    or provide an empty JSON object to clear the document store.

    Example of filters:
    `'{"filters": {{"name": ["some", "more"], "category": ["only_one"]}}'`

    To get all documents you should provide an empty dict, like:
    `'{"filters": {}}'`
    """
    docs = document_store.get_all_documents(filters=filters.filters)
    for doc in docs:
        doc.embedding = None
    return docs


@router.post("/documents/delete_by_filters", response_model=bool)
def delete_documents(filters: FilterRequest):
    """
    This endpoint allows you to delete documents contained in your document store.
    You can filter the documents to delete by metadata (like the document's name),
    or provide an empty JSON object to clear the document store.

    Example of filters:
    `'{"filters": {{"name": ["some", "more"], "category": ["only_one"]}}'`

    To get all documents you should provide an empty dict, like:
    `'{"filters": {}}'`
    """
    document_store.delete_documents(filters=filters.filters)
    return True
