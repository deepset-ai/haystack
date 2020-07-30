from abc import abstractmethod, ABC
from typing import Any, Optional, Dict, List, Union
from uuid import UUID, uuid4


class Document:
    def __init__(self, text: str,
                 id: Optional[Union[str, UUID]] = None,
                 query_score: Optional[float] = None,
                 question: Optional[str] = None,
                 meta: Dict[str, Any] = None,
                 tags: Optional[Dict[str, Any]] = None,
                 embedding: Optional[List[float]] = None):
        """
        Object used to represent documents / passages in a standardized way within Haystack.
        For example, this is what the retriever will return from the DocumentStore,
        regardless if it's ElasticsearchDocumentStore or InMemoryDocumentStore.

        Note that there can be multiple Documents originating from one file (e.g. PDF),
        if you split the text into smaller passages. We'll have one Document per passage in this case.

        :param id: ID used within the DocumentStore
        :param text: Text of the document
        :param query_score: Retriever's query score for a retrieved document
        :param question: Question text for FAQs.
        :param meta: Meta fields for a document like name, url, or author.
        :param tags: Tags that allow filtering of the data
        :param embedding: Vector encoding of the text
        """

        self.text = text
        # Create a unique ID (either new one, or one from user input)
        if id:
            if isinstance(id, str):
                self.id = UUID(hex=str(id), version=4)
            if isinstance(id, UUID):
                self.id = id
        else:
            self.id = uuid4()

        self.query_score = query_score
        self.question = question
        self.meta = meta
        self.tags = tags # deprecate?
        self.embedding = embedding

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, dict):
        _doc = dict.copy()
        init_args = ["text", "id", "query_score", "question", "meta", "tags", "embedding"]
        if "meta" not in _doc.keys():
            _doc["meta"] = {}
        # copy additional fields into "meta"
        for k, v in _doc.items():
            if k not in init_args:
                _doc["meta"][k] = v
        # remove additional fields from top level
        _doc = {k: v for k, v in _doc.items() if k in init_args}

        return cls(**_doc)


class Label:
    def __init__(self, question: str,
                 answer: str,
                 is_correct_answer: bool,
                 is_correct_document: bool,
                 origin: str,
                 document_id: Optional[UUID] = None,
                 offset_start_in_doc: Optional[int] = None,
                 no_answer: Optional[bool] = None,
                 model_id: Optional[int] = None):
        """
        Object used to represent label/feedback in a standardized way within Haystack.
        This includes labels from dataset like SQuAD, annotations from labeling tools,
        or, user-feedback from the Haystack REST API.

        :param question: the question(or query) for finding answers.
        :param answer: teh answer string.
        :param is_correct_answer: whether the sample is positive or negative.
        :param is_correct_document: in case of negative sample(is_correct_answer is False), there could be two cases;
                                    incorrect answer but correct document & incorrect document. This flag denotes if
                                    the returned document was correct.
        :param origin: the source for the labels. It can be used to later for filtering.
        :param document_id: the document_store's ID for the returned answer document.
        :param offset_start_in_doc: the answer start offset in the document.
        :param no_answer: whether the question in unanswerable.
        :param model_id: model_id used for prediction(in-case of user feedback).
        """
        self.no_answer = no_answer
        self.origin = origin
        self.question = question
        self.is_correct_answer = is_correct_answer
        self.is_correct_document = is_correct_document
        if document_id:
            if isinstance(document_id, str):
                self.document_id: Optional[UUID] = UUID(hex=str(document_id), version=4)
            if isinstance(document_id, UUID):
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
    def get_all_labels(self, index: str = "label", filters: Optional[dict] = None) -> List[Label]:
        pass

    @abstractmethod
    def get_document_by_id(self, id: UUID, index: Optional[str] = None) -> Optional[Document]:
        pass

    @abstractmethod
    def get_document_ids_by_tags(self, tag, index) -> List[str]:
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

    @abstractmethod
    def get_label_count(self, index: Optional[str] = None) -> int:
        pass

    @abstractmethod
    def add_eval_data(self, filename: str, doc_index: str = "document", label_index: str = "label"):
        pass

    def delete_all_documents(self, index: str):
        pass

