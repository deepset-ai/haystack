from datetime import datetime
from fastapi import APIRouter, HTTPException, Response
from haystack.schema import Document
from haystack.document_store.sql import SQLDocumentStore
from haystack.document_store.elasticsearch import ElasticsearchDocumentStore
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.document_store.memory import InMemoryDocumentStore
from haystack.document_store.milvus import MilvusDocumentStore
from haystack.document_store.weaviate import WeaviateDocumentStore
from http import HTTPStatus

now = datetime.now()
router = APIRouter()


@router.post("/documents", status_code=HTTPStatus.CREATED.value, response_model=Document)
def create_document(doc_store: DocumentStore Object, payload: Document, response: Response):
    """
    Creates a Document object and adds it to the document store specified
    :param doc_store: DocumentStore Object to identify the document store to use
    :param payload: input from the client to create a document.
    """
    doc_store.write_documents(payload.to_dict())
    if response.status_code > 399:
        status_code = response.status_code
        raise HTTPException(status_code=status_code, detail="Something went wrong while creating the document(s)")
    return {"status": "success",
            "created_at": now}


@router.get("/documents", status_code = HTTPStatus.OK.value, response_model = Document)
def get_all_documents(doc_store: DocumentStore Object, response: Response):
    """
    Returns all the documents stored in the document store specified
    :param doc_store: DocumentStore Object to identify the document store to use
    """
    documents = doc_store.get_all_documents()
    if response.status_code > 399:
        status_code = response.status_code
        raise HTTPException(status_code=status_code, detail="Something went wrong while retrieving the documents")
    return {"status": "success",
            "data": documents}


@router.get("/documents/{id}", status_code=HTTPStatus.OK.value, response_model=Document)
def get_document(id, doc_store: DocumentStore Object, response: Response):
    """
    Returns a document or list of documents given its id(s)
    :param id: UUID pointing to the document to retrieve
    :param doc_store: DocumentStore Object to identify the document store to modify
    """
    if not (isinstance(id,list) or isinstance(id,str)):
        raise HTTPException(status_code=400, detail=f"Wrong type of ID. ID = {type(id)}")
    if isinstance(s,str):
        doc=doc_store.get_document_by_id(id)
    if isinstance(s,list):
        doc=doc_store.get_documents_by_id(id)
    if response.status_code > 399:
        status_code = response.status_code
        raise HTTPException(status_code=status_code, detail=f"Something went wrong while retrieving the document(s)")
    return {"status": "success",
            "data": doc}


@router.put("/documents/{id}", status_code=HTTPStatus.OK.value, response_model=Document)
def update_document(id, doc_store: DocumentStore Object, payload: Document, response: Response):
    """
    Updates existing document(s) with its metadata
    :param id: unique id or list of ids pointing to the document(s) to retrieve
    :param doc_store: DocumentStore Object to identify the document store to modify
    :param payload: input from the client to create a Document
    """
    if not (isinstance(id,list) or isinstance(id,str)):
        raise HTTPException(status_code=400, detail=f"Wrong type of ID. ID = {type(id)}")
    docs=payload.to_dict()
    doc_store.write_documents(documents=docs, duplicate_documents="overwrite")
    if response.status_code > 399:
        status_code = response.status_code
        raise HTTPException(status_code=status_code, detail=f"Something went wrong while updating. ")
    return {"status": "success",
            "updated": id,
            "updated_at": now}


@router.delete("/documents/{id}", status_code = HTTPStatus.OK.value, response_model = Document)
def delete_document(id, doc_store: DocumentStore Object, payload: Document, response: Response):
    """
    Deletes existing document(s) with its metadata
    :param id: unique id or list of ids pointing to the document(s) to retrieve
    :param doc_store: DocumentStore Object to identify the document store to modify
    :param payload: input from the client to create a Document
    """
    if not (isinstance(id,list) or isinstance(id,str)):
        raise HTTPException(status_code=400, detail=f"Wrong type of ID. ID = {type(id)}")
    if isinstance(s,str):
        doc_store.delete_document_by_id(id)
    if isinstance(s,list):
        doc_store.delete_documents_by_id(id)
    if response.status_code > 399:
        status_code = response.status_code
        raise HTTPException(status_code=status_code, detail=f"Something went wrong while deleting the document(s)")
    return {"status": "success",
            "deleted": id}