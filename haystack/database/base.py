import logging
from abc import abstractmethod, ABC
from typing import Any, Optional, Dict, List, Union
from uuid import uuid4

import numpy as np


logger = logging.getLogger(__name__)


class Document:
    def __init__(self, text: str,
                 id: str = None,
                 query_score: Optional[float] = None,
                 question: Optional[str] = None,
                 meta: Dict[str, Any] = None,
                 embedding: Optional[np.array] = None):
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
        :param embedding: Vector encoding of the text
        """

        self.text = text
        # Create a unique ID (either new one, or one from user input)
        if id:
            self.id = str(id)
        else:
            self.id = str(uuid4())

        self.query_score = query_score
        self.question = question
        self.meta = meta
        self.embedding = embedding

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, dict):
        _doc = dict.copy()
        init_args = ["text", "id", "query_score", "question", "meta", "embedding"]
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
                 document_id: Optional[str] = None,
                 offset_start_in_doc: Optional[int] = None,
                 no_answer: Optional[bool] = None,
                 model_id: Optional[int] = None):
        """
        Object used to represent label/feedback in a standardized way within Haystack.
        This includes labels from dataset like SQuAD, annotations from labeling tools,
        or, user-feedback from the Haystack REST API.

        :param question: the question(or query) for finding answers.
        :param answer: the answer string.
        :param is_correct_answer: whether the sample is positive or negative.
        :param is_correct_document: in case of negative sample(is_correct_answer is False), there could be two cases;
                                    incorrect answer but correct document & incorrect document. This flag denotes if
                                    the returned document was correct.
        :param origin: the source for the labels. It can be used to later for filtering.
        :param document_id: the document_store's ID for the returned answer document.
        :param offset_start_in_doc: the answer start offset in the document.
        :param no_answer: whether the question in unanswerable.
        :param model_id: model_id used for prediction (in-case of user feedback).
        """
        self.question = question
        self.answer = answer
        self.is_correct_answer = is_correct_answer
        self.is_correct_document = is_correct_document
        self.origin = origin
        self.document_id = document_id
        self.offset_start_in_doc = offset_start_in_doc
        self.no_answer = no_answer
        self.model_id = model_id

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return self.__dict__

    # define __eq__ and __hash__ functions to deduplicate Label Objects
    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                getattr(other, 'question', None) == self.question and
                getattr(other, 'answer', None) == self.answer and
                getattr(other, 'is_correct_answer', None) == self.is_correct_answer and
                getattr(other, 'is_correct_document', None) == self.is_correct_document and
                getattr(other, 'origin', None) == self.origin and
                getattr(other, 'document_id', None) == self.document_id and
                getattr(other, 'offset_start_in_doc', None) == self.offset_start_in_doc and
                getattr(other, 'no_answer', None) == self.no_answer and
                getattr(other, 'model_id', None) == self.model_id)

    def __hash__(self):
        return hash(self.question +
                    self.answer +
                    str(self.is_correct_answer) +
                    str(self.is_correct_document) +
                    str(self.origin) +
                    str(self.document_id) +
                    str(self.offset_start_in_doc) +
                    str(self.no_answer) +
                    str(self.model_id))


class MultiLabel:
    def __init__(self, question: str,
                 multiple_answers: List[str],
                 is_correct_answer: bool,
                 is_correct_document: bool,
                 origin: str,
                 multiple_document_ids: List[Any],
                 multiple_offset_start_in_docs: List[Any],
                 no_answer: Optional[bool] = None,
                 model_id: Optional[int] = None):
        """
        Object used to aggregate multiple possible answers for the same question

        :param question: the question(or query) for finding answers.
        :param multiple_answers: list of possible answer strings
        :param is_correct_answer: whether the sample is positive or negative.
        :param is_correct_document: in case of negative sample(is_correct_answer is False), there could be two cases;
                                    incorrect answer but correct document & incorrect document. This flag denotes if
                                    the returned document was correct.
        :param origin: the source for the labels. It can be used to later for filtering.
        :param multiple_document_ids: the document_store's IDs for the returned answer documents.
        :param multiple_offset_start_in_docs: the answer start offsets in the document.
        :param no_answer: whether the question in unanswerable.
        :param model_id: model_id used for prediction (in-case of user feedback).
        """
        self.question = question
        self.multiple_answers = multiple_answers
        self.is_correct_answer = is_correct_answer
        self.is_correct_document = is_correct_document
        self.origin = origin
        self.multiple_document_ids = multiple_document_ids
        self.multiple_offset_start_in_docs = multiple_offset_start_in_docs
        self.no_answer = no_answer
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
    label_index: Optional[str]

    @abstractmethod
    def write_documents(self, documents: Union[List[dict], List[Document]], index: Optional[str] = None):
        """
        Indexes documents for later queries.

        :param documents: a list of Python dictionaries or a list of Haystack Document objects.
                          For documents as dictionaries, the format is {"text": "<the-actual-text>"}.
                          Optionally: Include meta data via {"text": "<the-actual-text>",
                          "meta":{"name": "<some-document-name>, "author": "somebody", ...}}
                          It can be used for filtering and is accessible in the responses of the Finder.
        :param index: Optional name of index where the documents shall be written to.
                      If None, the DocumentStore's default index (self.index) will be used.

        :return: None
        """
        pass

    @abstractmethod
    def get_all_documents(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None) -> List[Document]:
        pass

    @abstractmethod
    def get_all_labels(self, index: Optional[str] = None, filters: Optional[Dict[str, List[str]]] = None) -> List[Label]:
        pass

    def get_all_labels_aggregated(self,
                                  index: Optional[str] = None,
                                  filters: Optional[Dict[str, List[str]]] = None) -> List[MultiLabel]:
        aggregated_labels = []
        all_labels = self.get_all_labels(index=index, filters=filters)

        # Collect all answers to a question in a dict
        question_ans_dict = {} # type: ignore
        for l in all_labels:
            if l.question in question_ans_dict:
                question_ans_dict[l.question].append(l)
            else:
                question_ans_dict[l.question] = [l]

        # Aggregate labels
        for q, ls in question_ans_dict.items():
            ls = list(set(ls))  # get rid of exact duplicates
            # check if there are both text answer and "no answer" present
            t_present = False
            no_present = False
            no_idx = []
            for idx, l in enumerate(ls):
                if len(l.answer) == 0:
                    no_present = True
                else:
                    t_present = True
                    no_idx.append(idx)
            # if both text and no answer are present, remove no answer labels
            if t_present and no_present:
                logger.warning(
                    f"Both text label and 'no answer possible' label is present for question: {ls[0].question}")
                for remove_idx in no_idx[::-1]:
                    ls.pop(remove_idx)
            # when all labels to a question say "no answer" we just need the first occurence
            elif no_present and not t_present:
                ls = ls[:1]

            # construct Aggregated_label
            for i, l in enumerate(ls):
                if i == 0:
                    agg_label = MultiLabel(question=l.question,
                                           multiple_answers=[l.answer],
                                           is_correct_answer=l.is_correct_answer,
                                           is_correct_document=l.is_correct_document,
                                           origin=l.origin,
                                           multiple_document_ids=[l.document_id] if l.document_id else [],
                                           multiple_offset_start_in_docs=[
                                                     l.offset_start_in_doc] if l.offset_start_in_doc else [],
                                           no_answer=l.no_answer,
                                           model_id=l.model_id,
                                           )
                else:
                    agg_label.multiple_answers.append(l.answer)
                    agg_label.multiple_document_ids.append(l.document_id)
                    agg_label.multiple_offset_start_in_docs.append(l.offset_start_in_doc)
            aggregated_labels.append(agg_label)
        return aggregated_labels

    @abstractmethod
    def get_document_by_id(self, id: str, index: Optional[str] = None) -> Optional[Document]:
        pass

    @abstractmethod
    def get_document_count(self, index: Optional[str] = None) -> int:
        pass

    @abstractmethod
    def query_by_embedding(self,
                           query_emb: List[float],
                           filters: Optional[Optional[Dict[str, List[str]]]] = None,
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

