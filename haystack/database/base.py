from abc import abstractmethod
from typing import Optional, Dict

from pydantic import BaseModel, Field


class BaseDocumentStore:
    """
    Base class for implementing Document Stores.
    """

    @abstractmethod
    def write_documents(self, documents):
        pass

    @abstractmethod
    def get_document_by_id(self, id):
        pass

    @abstractmethod
    def get_document_ids_by_tags(self, tag):
        pass

    @abstractmethod
    def get_document_count(self):
        pass


class Document(BaseModel):
    id: str = Field(..., description="_id field from Elasticsearch")
    text: str = Field(..., description="Text of the document")
    external_source_id: Optional[str] = Field(
        None,
        description="id for the source file the document was created from. In the case when a large file is divided "
        "across multiple Elasticsearch documents, this id can be used to reference original source file.",
    )
    # name: Optional[str] = Field(None, description="Title of the document")
    question: Optional[str] = Field(None, description="Question text for FAQs.")
    query_score: Optional[int] = Field(None, description="Elasticsearch query score for a retrieved document")
    meta: Optional[Dict[str, Optional[str]]] = Field(None, description="")

    def __getitem__(self, item):
        if item == 'text':
            return self.text
        if item == 'id':
            return self.id
