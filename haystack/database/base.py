from abc import abstractmethod, ABC
from typing import Any, Optional, Dict, List, Union
from uuid import UUID, uuid4

class Document:
    def __init__(self, text: str,
                 id: Optional[Union[str, UUID]] = None,
                 query_score: Optional[float] = None,
                 question: Optional[str] = None,
                 meta: Dict[str, Any] = None,
                 tags: Optional[Dict[str, Any]] = None):
        """
        Object used to represent documents / passages in a standardized way within Haystack.
        For example, this is what the retriever will return from the DocumentStore,
        regardless if it's ElasticsearchDocumentStore or InMemoryDocumentStore.

        Note that there can be multiple Documents originating from one file (e.g. PDF),
        if you split the text into smaller passsages. We'll have one Document per passage in this case.

        :param id: ID used within the DocumentStore
        :param text: Text of the document
        :param query_score: Retriever's query score for a retrieved document
        :param question: Question text for FAQs.
        :param meta: Meta fields for a document like name, url, or author.
        :param tags: Tags that allow filtering of the data
        """
        self.text = text
        # Create a unique ID (either new one, or one from user input)
        if id:
            if type(id) == str:
                self.id = UUID(hex=id, version=4)
            if type(id) == UUID:
                self.id = id
        else:
            self.id = uuid4()

        self.query_score = query_score
        self.question = question
        self.meta = meta
        self.tags = tags # deprecate?

    def to_dict(self):
        #TODO what about tags, query_score etc?
        # d = {"text": self.text,
        #      "id": self.id,
        #      "meta": self.meta}
        return self.__dict__

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)


class Label:
    def __init__(self, question: str,
                 answer: str,
                 positive_sample: bool,
                 origin: str,
                 document_id: Optional[UUID] = None,
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
        if document_id:
            if type(document_id) == str:
                self.document_id = UUID(hex=document_id, version=4)
            if type(document_id) == UUID:
                self.document_id = document_id
        else:
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
    def write_documents(self, documents: List[dict], index: Optional[str] = None):
        """
        Indexes documents for later queries.

        :param documents: List of dictionaries.
                          Default format: {"text": "<the-actual-text>"}
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
        :param index: Optional name of index where the documents shall be written to.
                      If None, the DocumentStore's default index (self.index) will be used.

        :return: None
        """
        pass

    @abstractmethod
    def get_all_documents(self, index: Optional[str] = None) -> List[Document]:
        pass

    @abstractmethod
    def get_document_by_id(self, id: UUID, index: Optional[str] = None) -> Optional[Document]:
        pass

    @abstractmethod
    def get_document_ids_by_tags(self, tag, ) -> List[str]:
        pass

    @abstractmethod
    def get_document_count(self, index: Optional[str] = None) -> int:
        pass

    @abstractmethod
    def query_by_embedding(self,
                           query_emb: List[float],
                           filters: Optional[dict] = None,
                           top_k: int = 10,
                           index: Optional[str] = None) -> List[Document]:
        pass
