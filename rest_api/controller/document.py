from typing import List

import logging

from fastapi import APIRouter
from haystack import Document

from rest_api.controller.search import DOCUMENT_STORE
from rest_api.config import LOG_LEVEL
from rest_api.schema import FilterRequest, DocumentResponse


logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")


router = APIRouter()


@router.post("/documents/get_by_filters", response_model=List[DocumentResponse])
def get_documents_by_filter(filters: FilterRequest):
    """
    Can be used to get documents from a document store.

    :param filters: Filters to narrow down the documents to delete.
                    Example: {"name": ["some", "more"], "category": ["only_one"]}
    """
    docs = [doc.to_dict() for doc in DOCUMENT_STORE.get_all_documents(filters=filters.filters)]
    for doc in docs:
        del doc["embedding"]
    return [DocumentResponse(**doc) for doc in docs]


@router.post("/documents/delete_by_filters", response_model=bool)
def delete_documents_by_filter(filters: FilterRequest):
    """
    Can be used to delete documents from a document store.

    :param filters: Filters to narrow down the documents to delete.
                    Example: {"name": ["some", "more"], "category": ["only_one"]}
    """
    DOCUMENT_STORE.delete_documents(filters=filters.filters)
    return True