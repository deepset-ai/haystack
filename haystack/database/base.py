from abc import abstractmethod, ABC
from typing import Any, Optional, Dict, List


class Document:
    def __init__(self, id: str, text: str, external_source_id: Optional[str] = None, query_score:Optional[float] = None,
                 question: Optional[str] = None, meta:Dict[str, Any] = None,
                 tags: Optional[Dict[str, Any]] = None):
        """
        TODO update

        :param id: _id field from Elasticsearch
        :param text: "Text of the document"
        :param external_source_id: id for the source file the document was created from. In the case when a large file is divided "
        "across multiple Elasticsearch documents, this id can be used to reference original source file.
        :param query_score: Retriever's query score for a retrieved document
        :param question: Question text for FAQs.
        :param meta: Meta fields for a document like name, url, or author.
        :param tags: Tags that allow filtering of the data
        """
        self.text = text
        self.external_source_id = external_source_id
        self.query_score = query_score
        self.question = question
        self.meta = meta
        self.tags = tags
        self.id = id

    def to_dict(self):
        #TODO what about tags, query_score etc?
        d = {"text": self.text,
             "id": self.id,
             "meta": self.meta}
        return d

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)


class Label:
    def __init__(self, question: str,
                 answer: str,
                 positive_sample: bool,
                 origin: str,
                 document_id: Optional[str] = None,
                 offset_start_in_doc: Optional[int] = None,
                 no_answer: Optional[bool] = None,
                 model_id: Optional[int] = None):
        """
        #TODO

        :param question:
        :param answer:
        :param positive_sample:
        :param origin:
        :param document_id:
        :param offset_start_in_doc:
        :param no_answer:
        :param model_id:
        """
        self.no_answer = no_answer
        self.origin = origin
        self.question = question
        self.positive_sample = positive_sample
        self.document_id = document_id
        self.answer = answer
        self.offset_start_in_doc = offset_start_in_doc
        self.model_id = model_id

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return self.__dict__


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
