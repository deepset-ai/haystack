from typing import Any, Optional, Dict, List
from uuid import uuid4

import numpy as np


class Document:
    def __init__(self, text: str,
                 id: Optional[str] = None,
                 score: Optional[float] = None,
                 probability: Optional[float] = None,
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
        :param score: Retriever's query score for a retrieved document
        :param probability: a pseudo probability by scaling score in the range 0 to 1
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

        self.score = score
        self.probability = probability
        self.question = question
        self.meta = meta or {}
        self.embedding = embedding

    def to_dict(self, field_map={}):
        inv_field_map = {v:k for k, v in field_map.items()}
        _doc: Dict[str, str] = {}
        for k, v in self.__dict__.items():
            k = k if k not in inv_field_map else inv_field_map[k]
            _doc[k] = v
        return _doc

    @classmethod
    def from_dict(cls, dict, field_map={}):
        _doc = dict.copy()
        init_args = ["text", "id", "score", "probability", "question", "meta", "embedding"]
        if "meta" not in _doc.keys():
            _doc["meta"] = {}
        # copy additional fields into "meta"
        for k, v in _doc.items():
            if k not in init_args and k not in field_map:
                _doc["meta"][k] = v
        # remove additional fields from top level
        _new_doc = {}
        for k, v in _doc.items():
            if k in init_args:
                _new_doc[k] = v
            elif k in field_map:
                k = field_map[k]
                _new_doc[k] = v

        return cls(**_new_doc)

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())

class Label:
    def __init__(self, question: str,
                 answer: str,
                 is_correct_answer: bool,
                 is_correct_document: bool,
                 origin: str,
                 id: Optional[str] = None,
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
        :param id: Unique ID used within the DocumentStore. If not supplied, a uuid will be generated automatically.
        :param document_id: the document_store's ID for the returned answer document.
        :param offset_start_in_doc: the answer start offset in the document.
        :param no_answer: whether the question in unanswerable.
        :param model_id: model_id used for prediction (in-case of user feedback).
        """

        # Create a unique ID (either new one, or one from user input)
        if id:
            self.id = str(id)
        else:
            self.id = str(uuid4())

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

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())

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

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())