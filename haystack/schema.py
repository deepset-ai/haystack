from typing import Any, Optional, Dict, List
from uuid import uuid4
from copy import deepcopy
import mmh3
import numpy as np
from abc import abstractmethod
import inspect

class Document:
    def __init__(
        self,
        text: str,
        id: Optional[str] = None,
        score: Optional[float] = None,
        question: Optional[str] = None,
        meta: Dict[str, Any] = None,
        embedding: Optional[np.ndarray] = None,
        id_hash_keys: Optional[List[str]] = None
    ):
        """
        One of the core data classes in Haystack. It's used to represent documents / passages in a standardized way within Haystack.
        Documents are stored in DocumentStores, are returned by Retrievers, are the input for Readers and are used in
        many other places that manipulate or interact with document-level data.

        Note: There can be multiple Documents originating from one file (e.g. PDF), if you split the text
        into smaller passages. We'll have one Document per passage in this case.

        Each document has a unique ID. This can be supplied by the user or generated automatically.
        It's particularly helpful for handling of duplicates and referencing documents in other objects (e.g. Labels)

        There's an easy option to convert from/to dicts via `from_dict()` and `to_dict`.

        :param text: Text of the document
        :param id: Unique ID for the document. If not supplied by the user, we'll generate one automatically by
                   creating a hash from the supplied text. This behaviour can be further adjusted by `id_hash_keys`.
        :param score: Retriever's query score for a retrieved document
        :param question: Question text (e.g. for FAQs where one document usually consists of one question and one answer text).
        :param meta: Meta fields for a document like name, url, or author.
        :param embedding: Vector encoding of the text
        :param id_hash_keys: Generate the document id from a custom list of strings.
                             If you want ensure you don't have duplicate documents in your DocumentStore but texts are
                             not unique, you can provide custom strings here that will be used (e.g. ["filename_xy", "text_of_doc"].
        """

        self.text = text
        self.score = score
        self.question = question
        self.meta = meta or {}
        self.embedding = embedding

        # Create a unique ID (either new one, or one from user input)
        if id:
            self.id = str(id)
        else:
            self.id = self._get_id(id_hash_keys)

    def _get_id(self, id_hash_keys):
        final_hash_key = ":".join(id_hash_keys) if id_hash_keys else self.text
        return '{:02x}'.format(mmh3.hash128(final_hash_key, signed=False))

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
        init_args = ["text", "id", "score", "question", "meta", "embedding"]
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
                 model_id: Optional[int] = None,
                 created_at: Optional[str] = None,
                 updated_at: Optional[str] = None,
                 meta: Optional[dict] = None
                 ):
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
        :param created_at: Timestamp of creation with format yyyy-MM-dd HH:mm:ss.
                           Generate in Python via time.strftime("%Y-%m-%d %H:%M:%S").
        :param created_at: Timestamp of update with format yyyy-MM-dd HH:mm:ss.
                           Generate in Python via time.strftime("%Y-%m-%d %H:%M:%S")
        """

        # Create a unique ID (either new one, or one from user input)
        if id:
            self.id = str(id)
        else:
            self.id = str(uuid4())

        self.created_at = created_at
        self.updated_at = updated_at
        self.question = question
        self.answer = answer
        self.is_correct_answer = is_correct_answer
        self.is_correct_document = is_correct_document
        self.origin = origin
        self.document_id = document_id
        self.offset_start_in_doc = offset_start_in_doc
        self.no_answer = no_answer
        self.model_id = model_id
        if not meta:
            self.meta = dict()
        else:
            self.meta = meta

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
                getattr(other, 'model_id', None) == self.model_id and
                getattr(other, 'created_at', None) == self.created_at and
                getattr(other, 'updated_at', None) == self.updated_at)

    def __hash__(self):
        return hash(self.question +
                    self.answer +
                    str(self.is_correct_answer) +
                    str(self.is_correct_document) +
                    str(self.origin) +
                    str(self.document_id) +
                    str(self.offset_start_in_doc) +
                    str(self.no_answer) +
                    str(self.model_id)
                    )

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
                 model_id: Optional[int] = None,
                 meta: dict = None
                 ):
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
        if not meta:
            self.meta = dict()
        else:
            self.meta = meta

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())


class BaseComponent:
    """
    A base class for implementing nodes in a Pipeline.
    """

    outgoing_edges: int
    subclasses: dict = {}
    pipeline_config: dict = {}
    name: Optional[str] = None

    def __init_subclass__(cls, **kwargs):
        """ This automatically keeps track of all available subclasses.
        Enables generic load() for all specific component implementations.
        """
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def get_subclass(cls, component_type: str):
        if component_type not in cls.subclasses.keys():
            raise Exception(f"Haystack component with the name '{component_type}' does not exist.")
        subclass = cls.subclasses[component_type]
        return subclass

    @classmethod
    def load_from_args(cls, component_type: str, **kwargs):
        """
        Load a component instance of the given type using the kwargs.
        
        :param component_type: name of the component class to load.
        :param kwargs: parameters to pass to the __init__() for the component. 
        """
        subclass = cls.get_subclass(component_type)
        instance = subclass(**kwargs)
        return instance

    @classmethod
    def load_from_pipeline_config(cls, pipeline_config: dict, component_name: str):
        """
        Load an individual component from a YAML config for Pipelines.

        :param pipeline_config: the Pipelines YAML config parsed as a dict.
        :param component_name: the name of the component to load.
        """
        if pipeline_config:
            all_component_configs = pipeline_config["components"]
            all_component_names = [comp["name"] for comp in all_component_configs]
            component_config = next(comp for comp in all_component_configs if comp["name"] == component_name)
            component_params = component_config["params"]

            for key, value in component_params.items():
                if value in all_component_names:  # check if the param value is a reference to another component
                    component_params[key] = cls.load_from_pipeline_config(pipeline_config, value)

            component_instance = cls.load_from_args(component_config["type"], **component_params)
        else:
            component_instance = cls.load_from_args(component_name)
        return component_instance

    @abstractmethod
    def run(
        self,
        query: Optional[str] = None,
        file_paths: Optional[List[str]] = None,
        labels: Optional[MultiLabel] = None,
        documents: Optional[List[Document]] = None,
        meta: Optional[dict] = None,
        params: Optional[dict] = None,
    ):
        """
        Method that will be executed when the node in the graph is called.

        The argument that are passed can vary between different types of nodes
        (e.g. retriever nodes expect different args than a reader node)


        See an example for an implementation in haystack/reader/base/BaseReader.py
        :return:
        """
        pass

    def _dispatch_run(self, **kwargs):
        """
        The Pipelines call this method which in turn executes the run() method of Component.

        It takes care of the following:
          - inspect run() signature to validate if all necessary arguments are available
          - call run() with the corresponding arguments and gather output
          - collate _debug information if present
          - merge component output with the preceding output and pass it on to the subsequent Component in the Pipeline
        """
        arguments = deepcopy(kwargs)
        params = arguments.get("params") or {}

        run_signature_args = inspect.signature(self.run).parameters.keys()

        run_params = {}
        for key, value in params.items():
            if key == self.name:  # targeted params for this node
                if isinstance(value, dict):
                    for _k, _v in value.items():
                        if _k not in run_signature_args:
                            raise Exception(f"Invalid parameter '{_k}' for the node '{self.name}'.")
                run_params.update(**value)
            elif key in run_signature_args:  # global params
                run_params[key] = value

        run_inputs = {}
        for key, value in arguments.items():
            if key in run_signature_args:
                run_inputs[key] = value

        output, stream = self.run(**run_inputs, **run_params)

        # append _debug information from nodes
        all_debug = arguments.get("_debug", {})
        current_debug = output.get("_debug")
        if current_debug:
            all_debug[self.name] = current_debug
        if all_debug:
            output["_debug"] = all_debug

        # add "extra" args that were not used by the node
        for k, v in arguments.items():
            if k not in output.keys():
                output[k] = v

        output["params"] = params
        return output, stream

    def set_config(self, **kwargs):
        """
        Save the init parameters of a component that later can be used with exporting
        YAML configuration of a Pipeline.

        :param kwargs: all parameters passed to the __init__() of the Component.
        """
        if not self.pipeline_config:
            self.pipeline_config = {"params": {}, "type": type(self).__name__}
            for k, v in kwargs.items():
                if isinstance(v, BaseComponent):
                    self.pipeline_config["params"][k] = v.pipeline_config
                elif v is not None:
                    self.pipeline_config["params"][k] = v
