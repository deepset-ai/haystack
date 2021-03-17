from datetime import datetime
from fastapi import APIRouter, HTTPException, Response
from haystack.schema import Document
from haystack.document_store.sql import SQLDocumentStore
from http import HTTPStatus

now = datetime.now()
router = APIRouter()


@router.post("/documents", status_code=HTTPStatus.CREATED.value, response_model=Document)
def create_document(payload: Document, response: Response):
    """
    Creates a Document object
    :param payload: input from the client to create a document.
    """
    created_document = SQLDocumentStore.write_documents(payload.to_dict())
    if response.status_code > 399:
        status_code = response.status_code
        raise HTTPException(status_code=status_code, detail="Something went wrong while creating document")
    return {"status": "success",
            "data": created_document,
            "created_at": now}


@router.get("/documents", status_code = HTTPStatus.OK.value, response_model = Document)
def get_all_documents(response: Response):
    """
    Returns the documents which are available in the database
    """
    response_query = SQLDocumentStore.get_all_documents_generator()
    if response.status_code > 399:
        status_code = response.status_code
        raise HTTPException(status_code=status_code, detail="Something went wrong while retrieving documents")
    return {"status": "success",
            "data": response_query}


@router.get("/documents/{id}", status_code=HTTPStatus.OK.value, response_model=Document)
def get_document(id):
    """
    Returns a document given its id
    :param id: UUID pointing to the document to retrieve
    """
    if id is not str:
        raise HTTPException(status_code=400, detail=f"Wrong type of ID. ID = {type(id)}")
    response_query = SQLDocumentStore.get_document_by_id(id)
    if not response_query:
        raise HTTPException(status_code=404, detail=f"No document with id {id} could be found")
    return {"status": "success",
            "data": response_query}


@router.put("/documents/{id}", status_code=HTTPStatus.OK.value, response_model=Document)
def update_document(id, payload: Document, response: Response):
    """
    Updates an existing document with its metadata
    :param id: UUID pointing to the document to retrieve
    :param payload: input from the client to create a Document
    """
    if id is not str:
        raise HTTPException(status_code=400, detail=f"Wrong type of ID. ID = {type(id)}")
    document = SQLDocumentStore.delete_document_by_id(id)
    if not document:
        raise HTTPException(status_code=404, detail=f"No document with id {id} could be found")
    document_to_be_updated = SQLDocumentStore.write_documents(payload.to_dict(), update_existing_documents=True)
    if response.status_code > 399:
        status_code = response.status_code
        raise HTTPException(status_code=status_code, detail=f"Something went wrong while updating document {id}")
    return {"status": "success",
            "data": document_to_be_updated,
            "updated_at": now}


@router.delete("/documents/{id}", status_code = HTTPStatus.OK.value, response_model = Document)
def delete_document(id, response: Response):
    """
    Deletes a document given its id
    :param id: UUID pointing to the document to retrieve
    """
    if id is not str:
        raise HTTPException(status_code=400, detail=f"Wrong type of ID. ID = {type(id)}")
    SQLDocumentStore.delete_document_by_id(id)
    if response.status_code > 399:
        status_code = response.status_code
        raise HTTPException(status_code=status_code, detail=f"Something went wrong while deleting document {id}")
    return {"status": "success",
            "deleted": id}
