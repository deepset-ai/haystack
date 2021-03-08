from datetime import datetime
from haystack.schema import Document
from haystack.document_store import sql
from rest_api.config import LOG_LEVEL
from typing import Dict
from fastapi import APIRouter, HTTPException
import logging

sql = sql.SQLDocumentStore()
now = datetime.now()
SUCCESS: int = 200
CREATED: int = 201

logger = logging.getLogger('haystack')
logger.setLevel(LOG_LEVEL)

router = APIRouter()


@router.post("/documents", status_code=CREATED)
def create_document(payload: Dict[Document]):
    """
    Creates a Document object
    :param payload: input from the client to create a document.
    """
    logger.info("Constructing document object")
    created_document = Document.from_dict(payload)
    logger.info("Constructed!")
    return {"created": created_document,
            "created_at": now}


@router.get("/documents", status_code=SUCCESS)
def get_all_documents(start: int = 0, limit: int = 20):
    """
    Returns the documents which are available in the database
    Supports pagination
    :param start: used to set the starting point
    :param limit: used to delimit the number of retrieved documents
    """
    try:
        logger.info("Retrieving documents")
        retrieved_documents = sql.get_all_documents()
        return retrieved_documents[start: start + limit]
    except Exception as e:
        if TimeoutError:
            raise Exception("Request has timed out")
        else:
            raise Exception(f"Something went wrong while retrieving documents", e.message)


@router.get("/documents/{id}", status_code=SUCCESS)
def get_document(id: str):
    """
    Returns a document given its id
    :param id: UUID pointing to the document to retrieve
    """
    try:
        retrieved_document = sql.get_document_by_id(id)
        if not retrieved_document:
            raise HTTPException(status_code=404, detail=f"No document with id {id} could be found")
        return retrieved_document
    except Exception as e:
        if TimeoutError:
            raise Exception("Request has timed out")
        else:
            raise Exception(f"Something went wrong while retrieving documents", e.message)


@router.put("/documents/{id}", status_code=SUCCESS)
def update_document(id: str, payload: Document):
    """
    Updates an existing document with its metadata
    :param id: UUID pointing to the document to retrieve
    :param payload: input from the client to create a Document
    """
    try:
        if not id:
            raise HTTPException(status_code=404, detail=f"No document with id {id} could be found")
        sql.delete_document_by_id(id)
        cursor = sql.SQLDocumentStore(update_existing_documents=True)
        document_to_be_updated = cursor.write_documents(payload.to_dict())
        return {"updated": document_to_be_updated,
                "updated_at": now}
    except Exception as e:
        if TimeoutError:
            raise Exception("Request has timed out")
        else:
            raise Exception(f"Something went wrong while deleting document id {id}", e.message)


@router.delete("/documents/{id}", status_code=SUCCESS)
def delete_document(id: str):
    """
    Deletes a document given its id
    :param id: UUID pointing to the document to retrieve
    """
    try:
        if not id:
            raise HTTPException(status_code=404, detail=f"No document with id {id} could be found")
        sql.delete_document_by_id(id)
        return {"deleted": id}
    except Exception as e:
        if TimeoutError:
            raise Exception("Request has timed out")
        else:
            raise Exception(f"Something went wrong while deleting document id {id}", e.message)