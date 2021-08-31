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
from pydantic import BaseModel


now = datetime.now()
router = APIRouter()

class Request(BaseModel,Document):
    doc_store: str
    id: Union[str,List[id]] = None
    doc_list : List[Document] = None
    
@router.post("/documents/add", status_code=HTTPStatus.CREATED.value, response_model=Document)
def create_document(payload: Request, response: Response):
    """
    Creates a Document object and adds it to the document store specified
    :param payload: input from the client 
    """
    if (payload.doc_store=="sql"):
        SQLDocumentStore.write_documents(payload.doc_list.to_dict())
    if (payload.doc_store=="elasticsearch"):
        ElasticsearchDocumentStore.write_documents(payload.doc_list.to_dict())
    if (payload.doc_store=="faiss"):
        FAISSDocumentStore.write_documents(payload.doc_list.to_dict())
    if (payload.doc_store=="memory"):
        InMemoryDocumentStore.write_documents(payload.doc_list.to_dict())
    if (payload.doc_store=="milvus"):
        MilvusDocumentStore.write_documents(payload.doc_list.to_dict())
    if (payload.doc_store=="weaviate"):
        WeaviateDocumentStore.write_documents(payload.doc_list.to_dict())

    if response.status_code > 399:
        status_code = response.status_code
        raise HTTPException(status_code=status_code, detail="Something went wrong while creating the document(s)")
    return {"status": "success",
            "created_at": now}


@router.get("/documents/getall", status_code = HTTPStatus.OK.value, response_model = Document)
def get_all_documents(payload:Request, response: Response):
    """
    Returns all the documents stored in the document store specified
    """
    if (payload.doc_store=="sql"):
        documents = SQLDocumentStore.get_all_documents()
    if (payload.doc_store=="elasticsearch"):
        documents = ElasticsearchDocumentStore.get_all_documents()
    if (payload.doc_store=="faiss"):
        documents = FAISSDocumentStore.get_all_documents()
    if (payload.doc_store=="memory"):
        documents = InMemoryDocumentStore.get_all_documents()
    if (payload.doc_store=="milvus"):
        documents = MilvusDocumentStore.get_all_documents()
    if (payload.doc_store=="weaviate"):
        documents = WeaviateDocumentStore.get_all_documents()

    if response.status_code > 399:
        status_code = response.status_code
        raise HTTPException(status_code=status_code, detail="Something went wrong while retrieving the documents")
    return {"status": "success",
            "data": documents}


@router.get("/documents/get", status_code=HTTPStatus.OK.value, response_model=Document)
def get_document(payload:Request, response: Response):
    """
    Returns a document or list of documents given its id(s)
    """
    if not (isinstance(payload.id,list) or isinstance(payload.id,str)):
        raise HTTPException(status_code=400, detail=f"Wrong type of ID. ID = {type(id)}")
    if isinstance(payload.id,str):
        if (payload.doc_store=="sql"):
            doc=SQLDocumentStore.get_document_by_id(payload.id)
        if (payload.doc_store=="elasticsearch"):
            doc=ElasticsearchDocumentStore.get_document_by_id(payload.id)
        if (payload.doc_store=="faiss"):
            doc=FAISSDocumentStore.get_document_by_id(payload.id)
        if (payload.doc_store=="memory"):
            doc=InMemoryDocumentStore.get_document_by_id(payload.id)
        if (payload.doc_store=="milvus"):
            doc=MilvusDocumentStore.get_document_by_id(payload.id)
        if (payload.doc_store=="weaviate"):
            doc=WeaviateDocumentStore.get_document_by_id(payload.id)
        
    if isinstance(payload.id,list):
        if (payload.doc_store=="sql"):
            doc=SQLDocumentStore.get_documents_by_id(payload.id)
        if (payload.doc_store=="elasticsearch"):
            doc=ElasticsearchDocumentStore.get_documents_by_id(payload.id)
        if (payload.doc_store=="faiss"):
            doc=FAISSDocumentStore.get_documents_by_id(payload.id)
        if (payload.doc_store=="memory"):
            doc=InMemoryDocumentStore.get_documents_by_id(payload.id)
        if (payload.doc_store=="milvus"):
            doc=MilvusDocumentStore.get_documents_by_id(payload.id)
        if (payload.doc_store=="weaviate"):
            doc=WeaviateDocumentStore.get_documents_by_id(payload.id)
         
    if response.status_code > 399:
        status_code = response.status_code
        raise HTTPException(status_code=status_code, detail=f"Something went wrong while retrieving the document(s)")
    return {"status": "success",
            "data": doc}


@router.put("/documents/update", status_code=HTTPStatus.OK.value, response_model=Document)
def update_document(payload: Request, response: Response):
    """
    Updates existing document(s) with its metadata
    :param payload: input from the client 
    """
    if not (isinstance(payload.id,list) or isinstance(payload.id,str)):
        raise HTTPException(status_code=400, detail=f"Wrong type of ID. ID = {type(id)}")
    docs=payload.doc_list.to_dict()
    if (payload.doc_store=="sql"):
        SQLDocumentStore.write_documents(documents=docs, duplicate_documents="overwrite")
    if (payload.doc_store=="elasticsearch"):
        ElasticsearchDocumentStore.write_documents(documents=docs, duplicate_documents="overwrite")
    if (payload.doc_store=="faiss"):
        FAISSDocumentStore.write_documents(documents=docs, duplicate_documents="overwrite")
    if (payload.doc_store=="memory"):
        InMemoryDocumentStore.write_documents(documents=docs, duplicate_documents="overwrite")
    if (payload.doc_store=="milvus"):
        MilvusDocumentStore.write_documents(documents=docs, duplicate_documents="overwrite")
    if (payload.doc_store=="weaviate"):
        WeaviateDocumentStore.write_documents(documents=docs, duplicate_documents="overwrite")

    if response.status_code > 399:
        status_code = response.status_code
        raise HTTPException(status_code=status_code, detail=f"Something went wrong while updating. ")
    return {"status": "success",
            "updated": payload.id,
            "updated_at": now}


@router.delete("/documents/delete", status_code = HTTPStatus.OK.value, response_model = Document)
def delete_document(payload: Request, response: Response):
    """
    Deletes existing document(s) with its metadata
    :param payload: input from the client 
    """
    if not (isinstance(payload.id,list) or isinstance(payload.id,str)):
        raise HTTPException(status_code=400, detail=f"Wrong type of ID. ID = {type(id)}")
    
    if isinstance(payload.id,str):
        if (payload.doc_store=="sql"):
            doc=SQLDocumentStore.delete_document_by_id(payload.id)
        if (payload.doc_store=="elasticsearch"):
            doc=ElasticsearchDocumentStore.delete_document_by_id(payload.id)
        if (payload.doc_store=="faiss"):
            doc=FAISSDocumentStore.delete_document_by_id(payload.id)
        if (payload.doc_store=="memory"):
            doc=InMemoryDocumentStore.delete_document_by_id(payload.id)
        if (payload.doc_store=="milvus"):
            doc=MilvusDocumentStore.delete_document_by_id(payload.id)
        if (payload.doc_store=="weaviate"):
            doc=WeaviateDocumentStore.delete_document_by_id(payload.id)
        
    if isinstance(payload.id,list):
        if (payload.doc_store=="sql"):
            doc=SQLDocumentStore.delete_documents_by_id(payload.id)
        if (payload.doc_store=="elasticsearch"):
            doc=ElasticsearchDocumentStore.delete_documents_by_id(payload.id)
        if (payload.doc_store=="faiss"):
            doc=FAISSDocumentStore.delete_documents_by_id(payload.id)
        if (payload.doc_store=="memory"):
            doc=InMemoryDocumentStore.delete_documents_by_id(payload.id)
        if (payload.doc_store=="milvus"):
            doc=MilvusDocumentStore.delete_documents_by_id(payload.id)
        if (payload.doc_store=="weaviate"):
            doc=WeaviateDocumentStore.delete_documents_by_id(payload.id)

    if response.status_code > 399:
        status_code = response.status_code
        raise HTTPException(status_code=status_code, detail=f"Something went wrong while deleting the document(s)")
    return {"status": "success",
            "deleted": payload.id}