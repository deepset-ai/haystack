from __future__ import annotations

from typing import Any, Optional, Dict, List, Union, Callable, Tuple, Optional
from dataclasses import dataclass, asdict
from dataclasses_json import dataclass_json
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal #type: ignore

from uuid import uuid4
from copy import deepcopy
import mmh3
import numpy as np
from abc import abstractmethod
import inspect
import logging
import io
from functools import wraps
import time
import json
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Document:
    content: Union[str, pd.DataFrame]
    content_type: Literal["text", "table", "image"]
    id: str
    meta: Dict[str, Any]
    score: Optional[float] = None
    embedding: Optional[np.ndarray] = None
    id_hash_keys: Optional[List[str]] = None

    # We use a custom init here as we want some custom logic. The annotations above are however still needed in order
    # to use some dataclass magic like "asdict()". See https://www.python.org/dev/peps/pep-0557/#custom-init-method
    # They also help in annotating which object attributes will always be present (e.g. "id") even though they
    # don't need to passed by the user in init and are rather initialized automatically in the init
    def __init__(
            self,
            content: Union[str, pd.DataFrame],
            content_type: Literal["text", "table", "image"] = "text",
            id: Optional[str] = None,
            score: Optional[float] = None,
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

        :param content: Content of the document. For most cases, this will be text, but it can be a table or image.
        :param content_type: One of "image", "table" or "image". Haystack components can use this to adjust their
                             handling of Documents and check compatibility.
        :param id: Unique ID for the document. If not supplied by the user, we'll generate one automatically by
                   creating a hash from the supplied text. This behaviour can be further adjusted by `id_hash_keys`.
        :param score: The relevance score of the Document determined by a model (e.g. Retriever or Re-Ranker).
                      In the range of [0,1], where 1 means extremely relevant.
        :param meta: Meta fields for a document like name, url, or author in the form of a custom dict (any keys and values allowed).
        :param embedding: Vector encoding of the text
        :param id_hash_keys: Generate the document id from a custom list of strings.
                             If you want ensure you don't have duplicate documents in your DocumentStore but texts are
                             not unique, you can provide custom strings here that will be used (e.g. ["filename_xy", "text_of_doc"].
        """

        if content is None:
            raise ValueError(f"Can't create 'Document': Mandatory 'content' field is None")

        self.content = content
        self.content_type = content_type
        self.score = score
        self.meta = meta or {}

        if embedding is not None:
            embedding = np.asarray(embedding)
        self.embedding = embedding

        # Create a unique ID (either new one, or one from user input)
        if id:
            self.id: str = str(id)
        else:
            self.id: str = self._get_id(id_hash_keys)

    def _get_id(self, id_hash_keys):
        final_hash_key = ":".join(id_hash_keys) if id_hash_keys else self.content
        return '{:02x}'.format(mmh3.hash128(final_hash_key, signed=False))

    def to_dict(self, field_map={}) -> Dict:
        """
        Convert Document to dict. An optional field_map can be supplied to change the names of the keys in the
        resulting dict. This way you can work with standardized Document objects in Haystack, but adjust the format that
        they are serialized / stored in other places (e.g. elasticsearch)
        Example:
        | doc = Document(content="some text", content_type="text")
        | doc.to_dict(field_map={"custom_content_field": "content"})
        | >>> {"custom_content_field": "some text", content_type": "text"}

        :param field_map: Dict with keys being the custom target keys and values being the standard Document attributes
        :return: dict with content of the Document
        """
        inv_field_map = {v: k for k, v in field_map.items()}
        _doc: Dict[str, str] = {}
        for k, v in self.__dict__.items():
            k = k if k not in inv_field_map else inv_field_map[k]
            _doc[k] = v
        return _doc

    @classmethod
    def from_dict(cls, dict, field_map={}):
        """
        Create Document from dict. An optional field_map can be supplied to adjust for custom names of the keys in the
        input dict. This way you can work with standardized Document objects in Haystack, but adjust the format that
        they are serialized / stored in other places (e.g. elasticsearch)
        Example:
        | my_dict = {"custom_content_field": "some text", content_type": "text"}
        | Document.from_dict(my_dict, field_map={"custom_content_field": "content"})

        :param field_map: Dict with keys being the custom target keys and values being the standard Document attributes
        :return: dict with content of the Document
        """

        _doc = dict.copy()
        init_args = ["content", "content_type", "id", "score", "question", "meta", "embedding"]
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

    def to_json(self, field_map={}) -> str:
        d = self.to_dict(field_map=field_map)
        j = json.dumps(d, cls=NumpyEncoder)
        return j

    @classmethod
    def from_json(cls, data: str, field_map={}):
        d = json.loads(data)
        return cls.from_dict(d, field_map=field_map)

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                getattr(other, 'content', None) == self.content and
                getattr(other, 'content_type', None) == self.content_type and
                getattr(other, 'id', None) == self.id and
                getattr(other, 'score', None) == self.score and
                getattr(other, 'meta', None) == self.meta and
                np.array_equal(getattr(other, 'embedding', None), self.embedding) and
                getattr(other, 'id_hash_keys', None) == self.id_hash_keys)

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return f"content: {self.content[:100]} {'[...]' if len(self.content) > 100 else ''}"

    def __lt__(self, other):
        """ Enable sorting of Documents by score """
        return self.score < other.score


@dataclass
class Span:
    start: int
    end: int
    """
    Defining a sequence of characters (Text span) or cells (Table span) via start and end index. 
    For extractive QA: Character where answer starts/ends  
    For TableQA: Cell where the answer starts/ends (counted from top left to bottom right of table)
    
    :param start: Position where the span starts
    :param end:  Position where the spand ends
    """

@dataclass_json
@dataclass
class Answer:
    answer: str
    type: Literal["generative", "extractive", "other"] = "extractive"
    score: Optional[float] = None
    context: Optional[Union[str, pd.DataFrame]] = None
    offsets_in_document: Optional[List[Span]] = None
    offsets_in_context: Optional[List[Span]] = None
    document_id: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

    """
    The fundamental object in Haystack to represent any type of Answers (e.g. extractive QA, generative QA or TableQA).
    For example, it's used within some Nodes like the Reader, but also in the REST API.

    :param answer: The answer string. If there's no possible answer (aka "no_answer" or "is_impossible) this will be an empty string.
    :param type: One of ("generative", "extractive", "other"): Whether this answer comes from an extractive model 
                 (i.e. we can locate an exact answer string in one of the documents) or from a generative model 
                 (i.e. no pointer to a specific document, no offsets ...). 
    :param score: The relevance score of the Answer determined by a model (e.g. Reader or Generator).
                  In the range of [0,1], where 1 means extremely relevant.
    :param context: The related content that was used to create the answer (i.e. a text passage, part of a table, image ...)
    :param offsets_in_document: List of `Span` objects with start and end positions of the answer **in the
                                document** (as stored in the document store).
                                For extractive QA: Character where answer starts => `Answer.offsets_in_document[0].start 
                                For TableQA: Cell where the answer starts (counted from top left to bottom right of table) => `Answer.offsets_in_document[0].start
                                (Note that in TableQA there can be multiple cell ranges that are relevant for the answer, thus there can be multiple `Spans` here) 
    :param offsets_in_context: List of `Span` objects with start and end positions of the answer **in the
                                context** (i.e. the surrounding text/table of a certain window size).
                                For extractive QA: Character where answer starts => `Answer.offsets_in_document[0].start 
                                For TableQA: Cell where the answer starts (counted from top left to bottom right of table) => `Answer.offsets_in_document[0].start
                                (Note that in TableQA there can be multiple cell ranges that are relevant for the answer, thus there can be multiple `Spans` here) 
    :param document_id: ID of the document that the answer was located it (if any)
    :param meta: Dict that can be used to associate any kind of custom meta data with the answer. 
                 In extractive QA, this will carry the meta data of the document where the answer was found.
    """

    def __post_init__(self):
        # In case offsets are passed as dicts rather than Span objects we convert them here
        # For example, this is used when instantiating an object via from_json()
        if self.offsets_in_document is not None:
            self.offsets_in_document = [Span(**e) if isinstance(e, dict) else e for e in self.offsets_in_document]
        if self.offsets_in_context is not None:
            self.offsets_in_context = [Span(**e) if isinstance(e, dict) else e for e in self.offsets_in_context]

        if self.meta is None:
            self.meta = {}

    def __lt__(self, other):
        """ Enable sorting of Answers by score """
        return self.score < other.score

    def __str__(self):
        return f"answer: {self.answer} \nscore: {self.score} \ncontext: {self.context}"

    #TODO: switch to manual serialization instead of dataclass_json as it seems to break autocomplete of IDE in some cases
    # def to_json(self):
    #     # usage of dataclass_json seems to break autocomplete in the IDE, so we implement the methods ourselves here
    #     j = json.dumps(asdict(self))
    #     return j
    #
    # @classmethod
    # def from_json(cls, data):
    #     d = json.loads(data)
    #     return cls(**d)


@dataclass_json
@dataclass
class Label:
    id: str
    query: str
    document: Document
    is_correct_answer: bool
    is_correct_document: bool
    origin: Literal["user-feedback", "gold-label"]
    answer: Optional[Answer] = None
    no_answer: Optional[bool] = None
    pipeline_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    meta: Optional[dict] = None

    # We use a custom init here as we want some custom logic. The annotations above are however still needed in order
    # to use some dataclass magic like "asdict()". See https://www.python.org/dev/peps/pep-0557/#custom-init-method
    def __init__(self,
                 query: str,
                 document: Document,
                 is_correct_answer: bool,
                 is_correct_document: bool,
                 origin: Literal["user-feedback", "gold-label"],
                 answer: Optional[Answer],
                 id: Optional[str] = None,
                 no_answer: Optional[bool] = None,
                 pipeline_id: Optional[str] = None,
                 created_at: Optional[str] = None,
                 updated_at: Optional[str] = None,
                 meta: Optional[dict] = None
                 ):
        """
        Object used to represent label/feedback in a standardized way within Haystack.
        This includes labels from dataset like SQuAD, annotations from labeling tools,
        or, user-feedback from the Haystack REST API.

        :param query: the question (or query) for finding answers.
        :param document:
        :param answer: the answer object.
        :param is_correct_answer: whether the sample is positive or negative.
        :param is_correct_document: in case of negative sample(is_correct_answer is False), there could be two cases;
                                    incorrect answer but correct document & incorrect document. This flag denotes if
                                    the returned document was correct.
        :param origin: the source for the labels. It can be used to later for filtering.
        :param id: Unique ID used within the DocumentStore. If not supplied, a uuid will be generated automatically.
        :param no_answer: whether the question in unanswerable.
        :param pipeline_id: pipeline identifier (any str) that was involved for generating this label (in-case of user feedback).
        :param created_at: Timestamp of creation with format yyyy-MM-dd HH:mm:ss.
                           Generate in Python via time.strftime("%Y-%m-%d %H:%M:%S").
        :param created_at: Timestamp of update with format yyyy-MM-dd HH:mm:ss.
                           Generate in Python via time.strftime("%Y-%m-%d %H:%M:%S")
        :param meta: Meta fields like "annotator_name" in the form of a custom dict (any keys and values allowed).
        """

        # Create a unique ID (either new one, or one from user input)
        if id:
            self.id = str(id)
        else:
            self.id = str(uuid4())

        if created_at is None:
            created_at = time.strftime("%Y-%m-%d %H:%M:%S")
        self.created_at = created_at

        self.updated_at = updated_at
        self.query = query
        self.answer = answer
        self.document = document
        self.is_correct_answer = is_correct_answer
        self.is_correct_document = is_correct_document
        self.origin = origin

        # Remove
        # self.document_id = document_id
        # self.offset_start_in_doc = offset_start_in_doc

        # If an Answer is provided we need to make sure that it's consistent with the `no_answer` value
        # TODO: reassess if we want to enforce Span.start=0 and Span.end=0 for no_answer=True
        if self.answer is not None:
            if no_answer == True:
                if self.answer.answer != ""  or self.answer.context:
                    raise ValueError(f"Got no_answer == True while there seems to be an possible Answer: {self.answer}")
            elif no_answer == False:
                if self.answer.answer == "":
                    raise ValueError(f"Got no_answer == False while there seems to be no possible Answer: {self.answer}")
            else:
                # Automatically infer no_answer from Answer object
                if self.answer.answer == "" or self.answer.answer is None:
                    no_answer = True
                else:
                    no_answer = False
        self.no_answer = no_answer

        # TODO autofill answer.document_id if Document is provided

        self.pipeline_id = pipeline_id
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
                getattr(other, 'query', None) == self.query and
                getattr(other, 'answer', None) == self.answer and
                getattr(other, 'is_correct_answer', None) == self.is_correct_answer and
                getattr(other, 'is_correct_document', None) == self.is_correct_document and
                getattr(other, 'origin', None) == self.origin and
                getattr(other, 'document', None) == self.document and
                getattr(other, 'no_answer', None) == self.no_answer and
                getattr(other, 'pipeline_id', None) == self.pipeline_id)

    def __hash__(self):
        return hash(self.query +
                    str(self.answer) +
                    str(self.is_correct_answer) +
                    str(self.is_correct_document) +
                    str(self.origin) +
                    str(self.document) +
                    str(self.no_answer) +
                    str(self.pipeline_id)
                    )

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())


@dataclass
class MultiLabel:
    def __init__(self,
                 labels: List[Label],
                 drop_negative_labels=False,
                 drop_no_answers=False
                 ):
        """
        There are often multiple `Labels` associated with a single query. For example, there can be multiple annotated
        answers for one question or multiple documents contain the information you want for a query.
        This class is "syntactic sugar" that simplifies the work with such a list of related Labels.
        It stored the original labels in MultiLabel.labels and provides additional aggregated attributes that are
        automatically created at init time. For example, MultiLabel.no_answer allows you to easily access if any of the
        underlying Labels provided a text answer and therefore demonstrates that there is indeed a possible answer.

        :param labels: A list lof labels that belong to a similar query and shall be "grouped" together
        :param drop_negative_labels: Whether to drop negative labels from that group (e.g. thumbs down feedback from UI)
        :param drop_no_answers: Whether to drop labels that specify the answer is impossible
        """

        # drop duplicate labels and remove negative labels if needed.
        labels = list(set(labels))
        if drop_negative_labels:
            is_positive_label = lambda l: (l.is_correct_answer and l.is_correct_document) or \
                                          (l.answer is None and l.is_correct_document)
            labels = [l for l in labels if is_positive_label(l)]

        if drop_no_answers:
            labels = [l for l in labels if l.no_answer == False]

        self.labels = labels

        self.query = self._aggregate_labels(key="query", must_be_single_value=True)[0]
        # answer strings as this is mostly relevant in usage
        self.answers = [l.answer.answer for l in self.labels if l.answer is not None]
        # Currently no_answer is only true if all labels are "no_answers", we could later introduce a param here to let
        # users decided which aggregation logic they want
        self.no_answer = False not in [l.no_answer for l in self.labels]
        self.document_ids = [l.document.id for l in self.labels]

    def _aggregate_labels(self, key, must_be_single_value=True) -> List[Any]:
        unique_values = set([getattr(l, key) for l in self.labels])
        if must_be_single_value and len(unique_values) > 1:
                raise ValueError(f"Tried to combine attribute '{key}' of Labels, but found multiple different values: {unique_values}")
        else:
            return list(unique_values)

    @classmethod
    def from_dict(cls, dict):
        return cls(**dict)

    def to_dict(self):
        return self.__dict__

    def __repr__(self):
        return str(self.to_dict())

    def __str__(self):
        return str(self.to_dict())


class InMemoryLogger(io.TextIOBase):
    """
    Implementation of a logger that keeps track
    of the log lines in a list called `logs`,
    from where they can be accessed freely.
    """

    def __init__(self, *args):
        io.TextIOBase.__init__(self, *args)
        self.logs = []

    def write(self, x):
        self.logs.append(x)


def record_debug_logs(func: Callable, node_name: str, logs: bool) -> Callable:
    """
    Captures the debug logs of the wrapped function and
    saves them in the `_debug` key of the output dictionary.
    If `logs` is True, dumps the same logs to the console as well.

    Used in `BaseComponent.__getattribute__()` to wrap `run()` functions.
    This makes sure that every implementation of `run()` by a subclass will
    be automagically decorated with this method when requested.

    :param func: the function to decorate (must be an implementation of
                 `BaseComponent.run()`).
    :param logs: whether the captured logs should also be displayed
                 in the console during the execution of the pipeline.
    """
    @wraps(func)
    def inner(*args, **kwargs) -> Tuple[Dict[str, Any], str]:

        with InMemoryLogger() as logs_container:
            logger = logging.getLogger()

            # Adds a handler that stores the logs in a variable
            handler = logging.StreamHandler(logs_container)
            handler.setLevel(logger.level or logging.DEBUG)
            logger.addHandler(handler)

            # Add a handler that prints log messages in the console
            # to the specified level for the node
            if logs:
                handler_console = logging.StreamHandler()
                handler_console.setLevel(logging.DEBUG)
                formatter = logging.Formatter(f'[{node_name} logs] %(message)s')
                handler_console.setFormatter(formatter)
                logger.addHandler(handler_console)

            output, stream = func(*args, **kwargs)

            if not "_debug" in output.keys():
                output["_debug"] = {}
            output["_debug"]["logs"] = logs_container.logs

            # Remove both handlers
            logger.removeHandler(handler)
            if logs:
                logger.removeHandler(handler_console)

            return output, stream

    return inner


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

    def __getattribute__(self, name):
        """
        This modified `__getattribute__` method automagically decorates
        every `BaseComponent.run()` implementation with the
        `record_debug_logs` decorator defined above.

        This decorator makes the function collect its debug logs into a
        `_debug` key of the output dictionary.

        The logs collection is not always performed. Before applying the decorator,
        it checks for an instance attribute called `debug` to know
        whether it should or not. The decorator is applied if the attribute is
        defined and True.

        In addition, the value of the instance attribute `debug_logs` is
        passed to the decorator. If it's True, it will print the
        logs in the console as well.
        """
        if name == "run" and self.debug:
            func = getattr(type(self), "run")
            return record_debug_logs(func=func, node_name=self.__class__.__name__, logs=self.debug_logs).__get__(self)
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        """
        Ensures that `debug` and `debug_logs` are always defined.
        """
        if name in ["debug", "debug_logs"]:
            return None
        raise AttributeError(name)

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
        meta: Optional[dict] = None
    ) -> Tuple[Dict, str]:
        """
        Method that will be executed when the node in the graph is called.

        The argument that are passed can vary between different types of nodes
        (e.g. retriever nodes expect different args than a reader node)


        See an example for an implementation in haystack/reader/base/BaseReader.py
        :return:
        """
        pass

    def _dispatch_run(self, **kwargs) -> Tuple[Dict, str]:
        """
        The Pipelines call this method which in turn executes the run() method of Component.

        It takes care of the following:
          - inspect run() signature to validate if all necessary arguments are available
          - pop `debug` and `debug_logs` and sets them on the instance to control debug output
          - call run() with the corresponding arguments and gather output
          - collate `_debug` information if present
          - merge component output with the preceding output and pass it on to the subsequent Component in the Pipeline
        """
        arguments = deepcopy(kwargs)
        params = arguments.get("params") or {}

        run_signature_args = inspect.signature(self.run).parameters.keys()

        run_params: Dict[str, Any] = {}
        for key, value in params.items():
            if key == self.name:  # targeted params for this node
                if isinstance(value, dict):

                    # Extract debug attributes
                    if "debug" in value.keys():
                        self.debug = value.pop("debug")
                    if "debug_logs" in value.keys():
                        self.debug_logs = value.pop("debug_logs")

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

        # Collect debug information
        current_debug = output.get("_debug", {})
        if self.debug:
            current_debug["input"] = {**run_inputs, **run_params}
            if self.debug:
                current_debug["input"]["debug"] = self.debug
            if self.debug_logs:
                current_debug["input"]["debug_logs"] = self.debug_logs
            filtered_output = {key: value for key, value in output.items() if key != "_debug"} # Exclude _debug to avoid recursion
            current_debug["output"] = filtered_output

        # append _debug information from nodes
        all_debug = arguments.get("_debug", {})
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


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)