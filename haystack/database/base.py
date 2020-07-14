from abc import abstractmethod, ABC
from typing import Any, Optional, Dict, List

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str = Field(..., description="_id field from Elasticsearch")
    text: str = Field(..., description="Text of the document")
    external_source_id: Optional[str] = Field(
        None,
        description="id for the source file the document was created from. In the case when a large file is divided "
        "across multiple Elasticsearch documents, this id can be used to reference original source file.",
    )
    question: Optional[str] = Field(None, description="Question text for FAQs.")
    query_score: Optional[float] = Field(None, description="Elasticsearch query score for a retrieved document")
    meta: Dict[str, Any] = Field({}, description="Meta fields for a document like name, url, or author.")
    tags: Optional[Dict[str, Any]] = Field(None, description="Tags that allow filtering of the data")


class BaseDocumentStore(ABC):
    """
    Base class for implementing Document Stores.
    """
    index: Optional[str]

    @abstractmethod
    def write_documents(self, documents: List[dict]):
        """
        Indexes documents for later queries.

        :param documents: List of dictionaries.
                          Default format: {"text": "<the-actual-text>"}
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.

        :return: None
        """
        pass

    @abstractmethod
    def get_all_documents(self) -> List[Document]:
        pass

    @abstractmethod
    def get_document_by_id(self, id: str) -> Optional[Document]:
        pass

    @abstractmethod
    def get_document_ids_by_tags(self, tag) -> List[str]:
        pass

    @abstractmethod
    def get_document_count(self) -> int:
        pass

    @abstractmethod
    def query_by_embedding(self,
                           query_emb: List[float],
                           filters: Optional[dict] = None,
                           top_k: int = 10,
                           index: Optional[str] = None) -> List[Document]:
        pass
