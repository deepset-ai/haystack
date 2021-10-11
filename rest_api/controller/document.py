import logging

from fastapi import APIRouter

from rest_api.controller.search import DOCUMENT_STORE
from rest_api.config import LOG_LEVEL
from rest_api.schema import FilterRequest


logging.getLogger("haystack").setLevel(LOG_LEVEL)
logger = logging.getLogger("haystack")


router = APIRouter()


@router.post("/documents/delete_by_filters", response_model=bool)
def delete_documents(filters: FilterRequest):
    """
    Can be used to delete documents from a document store.

    :param filters: Filters to narrow down the documents to delete.
                    Example: {"name": ["some", "more"], "category": ["only_one"]}
    """
    DOCUMENT_STORE.delete_documents(filters=filters.filters)
    return True