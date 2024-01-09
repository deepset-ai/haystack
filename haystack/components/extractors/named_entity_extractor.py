from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import Any, Dict, List, Optional, Union

from ... import ComponentError, DeserializationError, Document, component, default_from_dict, default_to_dict
from ...lazy_imports import LazyImport

with LazyImport(message="Run 'pip install transformers[torch]'") as transformers_import:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from transformers import Pipeline as HfPipeline
    from transformers import pipeline

with LazyImport(message="Run 'pip install spacy'") as spacy_import:
    import spacy
    from spacy import Language as SpacyPipeline


class _BackendEnumMeta(EnumMeta):
    """
    Metaclass for fine-grained error handling of backend enums.
    """

    def __call__(cls, value, names=None, *, module=None, qualname=None, type=None, start=1):
        if names is None:
            try:
                return EnumMeta.__call__(cls, value, names, module=module, qualname=qualname, type=type, start=start)
            except ValueError:
                supported_backends = ", ".join(sorted(v.value for v in cls))  # pylint: disable=not-an-iterable
                raise ComponentError(
                    f"Invalid backend `{value}` for named entity extractor. "
                    f"Supported backends: {supported_backends}"
                )
        else:
            return EnumMeta.__call__(  # pylint: disable=too-many-function-args
                cls, value, names, module, qualname, type, start
            )


class NamedEntityExtractorBackend(Enum, metaclass=_BackendEnumMeta):
    """
    NLP backend to use for Named Entity Recognition.
    """

    #: Hugging Face.
    #:
    #: Uses an Hugging Face model and pipeline.
    HUGGING_FACE = "hugging_face"

    #: spaCy.
    #:
    #: Uses a spaCy model and pipeline.
    SPACY = "spacy"


@dataclass
class NamedEntityAnnotation:
    """
    Describes a single NER annotation.

    :param entity:
        Entity label.
    :param start:
        Start index of the entity in the document.
    :param end:
        End index of the entity in the document.
    :param score:
        Score calculated by the model.
    """

    entity: str
    start: int
    end: int
    score: Optional[float] = None


@component
class NamedEntityExtractor:
    _METADATA_KEY = "named_entities"

    def __init__(
        self,
        *,
        backend: Union[str, NamedEntityExtractorBackend],
        model_name_or_path: str,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
        device_id: int = -1,
    ) -> None:
        """
        Construct a Named Entity extractor component.

        :param backend:
            Backend to use for NER.
        :param model_name_or_path:
            Name of the model or a path to the model on
            the local disk.

            Dependent on the backend.
        :param pipeline_kwargs:
            Keyword arguments passed to the pipeline. The
            pipeline can override these arguments.

            Dependent on the backend.
        :param device_id:
            Identifier of the device on which the backend
            is executed.

            To execute on the CPU, pass a value of `-1`.
            To execute on the GPU, pass the GPU identifier.
        """

        if isinstance(backend, str):
            backend = NamedEntityExtractorBackend(backend)

        self._backend: _NerBackend
        if backend == NamedEntityExtractorBackend.HUGGING_FACE:
            self._backend = _HfBackend(
                model_name_or_path=model_name_or_path, device_id=device_id, pipeline_kwargs=pipeline_kwargs
            )
        elif backend == NamedEntityExtractorBackend.SPACY:
            self._backend = _SpacyBackend(
                model_name_or_path=model_name_or_path, device_id=device_id, pipeline_kwargs=pipeline_kwargs
            )
        else:
            raise ComponentError(f"Unknown NER backend '{type(backend).__name__}' for extractor")

    def warm_up(self):
        try:
            self._backend.initialize()
        except Exception as e:
            raise ComponentError(
                f"Named entity extractor with backend '{self._backend.type} failed to initialize."
            ) from e

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], batch_size: int = 1) -> Dict[str, Any]:
        texts = [doc.content if doc.content is not None else "" for doc in documents]
        annotations = self._backend.annotate(texts, batch_size=batch_size)

        if len(annotations) != len(documents):
            raise ComponentError(
                "NER backend did not return the correct number of annotations; "
                f"got {len(annotations)} but expected {len(documents)}"
            )

        for doc, doc_annotations in zip(documents, annotations):
            doc.meta[self._METADATA_KEY] = doc_annotations

        return {"documents": documents}

    def to_dict(self) -> Dict[str, Any]:
        return default_to_dict(
            self,
            backend=self._backend.type,
            model_name_or_path=self._backend.model_name,
            device_id=self._backend.device_id,
            pipeline_kwargs=self._backend._pipeline_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NamedEntityExtractor":
        try:
            return default_from_dict(cls, data)
        except Exception as e:
            raise DeserializationError(f"Couldn't deserialize {cls.__name__} instance") from e

    @property
    def initialized(self) -> bool:
        """
        Returns if the extractor is ready to annotate text.
        """
        return self._backend.initialized

    @classmethod
    def get_stored_annotations(cls, document: Document) -> Optional[List[NamedEntityAnnotation]]:
        """
        Returns the document's named entity annotations stored
        in its metadata, if any.

        :param document:
            Document whose annotations are to be fetched.
        :returns:
            The stored annotations.
        """

        return document.meta.get(cls._METADATA_KEY)


class _NerBackend(ABC):
    """
    Base class for NER backends.
    """

    def __init__(self, type: NamedEntityExtractorBackend, pipeline_kwargs: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()

        self._type = type
        self._pipeline_kwargs = pipeline_kwargs if pipeline_kwargs is not None else {}

    @abstractmethod
    def initialize(self):
        """
        Initializes the backend. This would usually
        entail loading models, pipelines, etc.
        """

    @property
    @abstractmethod
    def initialized(self) -> bool:
        """
        Returns if the backend has been initialized, i.e,
        ready to annotate text.
        """

    @abstractmethod
    def annotate(self, texts: List[str], *, batch_size: int = 1) -> List[List[NamedEntityAnnotation]]:
        """
        Predict annotations for a collection of documents.

        :param texts:
            Raw texts to be annotated.
        :param batch_size:
            Size of text batches that are
            passed to the model.
        :returns:
            NER annotations.
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Returns the model name or path on the local disk.
        """

    @property
    @abstractmethod
    def device_id(self) -> int:
        """
        Returns the identifier of the device on which
        the backend's model is loaded.
        """

    @property
    def type(self) -> NamedEntityExtractorBackend:
        """
        Returns the type of the backend.
        """
        return self._type


class _HfBackend(_NerBackend):
    """
    Hugging Face backend for NER.
    """

    def __init__(
        self, *, model_name_or_path: str, device_id: int, pipeline_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Construct a Hugging Face NER backend.

        :param model_name_or_path:
            Name of the model or a path to the Hugging Face
            model on the local disk.
        :param device_id:
            Identifier of the device on which the backend
            is executed.

            To execute on the CPU, pass a value of `-1`.
            To execute on the GPU, pass the GPU identifier.
        :param pipeline_kwargs:
            Keyword arguments passed to the pipeline. The
            pipeline can override these arguments.
        """
        super().__init__(NamedEntityExtractorBackend.HUGGING_FACE, pipeline_kwargs)

        transformers_import.check()

        self._model_name_or_path = model_name_or_path
        self._device_id = device_id

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForTokenClassification] = None
        self.pipeline: Optional[HfPipeline] = None

    def initialize(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self._model_name_or_path)

        pipeline_params = {
            "task": "ner",
            "model": self.model,
            "tokenizer": self.tokenizer,
            "aggregation_strategy": "simple",
            "device": self.device_id,
        }
        pipeline_params.update({k: v for k, v in self._pipeline_kwargs.items() if k not in pipeline_params})
        self.pipeline = pipeline(**pipeline_params)

    def annotate(self, texts: List[str], *, batch_size: int = 1) -> List[List[NamedEntityAnnotation]]:
        if not self.initialized:
            raise ComponentError("Hugging Face NER backend was not initialized - Did you call `warm_up()`?")

        assert self.pipeline is not None
        outputs = self.pipeline(texts, batch_size=batch_size)
        return [
            [
                NamedEntityAnnotation(
                    entity=annotation["entity"] if "entity" in annotation else annotation["entity_group"],
                    start=annotation["start"],
                    end=annotation["end"],
                    score=annotation["score"],
                )
                for annotation in annotations
            ]
            for annotations in outputs
        ]

    @property
    def initialized(self) -> bool:
        return self.tokenizer is not None and self.model is not None or self.pipeline is not None

    @property
    def model_name(self) -> str:
        return self._model_name_or_path

    @property
    def device_id(self) -> int:
        return self._device_id


class _SpacyBackend(_NerBackend):
    """
    spaCy backend for NER.
    """

    def __init__(
        self, *, model_name_or_path: str, device_id: int, pipeline_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Construct a spaCy NER backend.

        :param model_name_or_path:
            Name of the model or a path to the spaCy
            model on the local disk.
        :param device_id:
            Identifier of the device on which the backend
            is executed.

            To execute on the CPU, pass a value of `-1`.
            To execute on the GPU, pass the GPU identifier.
        :param pipeline_kwargs:
            Keyword arguments passed to the pipeline. The
            pipeline can override these arguments.
        """
        super().__init__(NamedEntityExtractorBackend.SPACY, pipeline_kwargs)

        spacy_import.check()

        self._model_name_or_path = model_name_or_path
        self._device_id = device_id

        self.pipeline: Optional[SpacyPipeline] = None

    def initialize(self):
        # We need to initialize the model on the GPU if needed.
        with self._select_device():
            self.pipeline = spacy.load(self._model_name_or_path)

        if not self.pipeline.has_pipe("ner"):
            raise ComponentError(f"spaCy pipeline '{self._model_name_or_path}' does not contain an NER component")

        # Disable unnecessary pipes.
        pipes_to_keep = ("ner", "tok2vec", "transformer", "curated_transformer")
        for name in self.pipeline.pipe_names:
            if name not in pipes_to_keep:
                self.pipeline.disable_pipe(name)

        self._pipeline_kwargs = {k: v for k, v in self._pipeline_kwargs.items() if k not in ("texts", "batch_size")}

    def annotate(self, texts: List[str], *, batch_size: int = 1) -> List[List[NamedEntityAnnotation]]:
        if not self.initialized:
            raise ComponentError("spaCy NER backend was not initialized - Did you call `warm_up()`?")

        assert self.pipeline is not None
        with self._select_device():
            outputs = list(self.pipeline.pipe(texts=texts, batch_size=batch_size, **self._pipeline_kwargs))

        return [
            [
                NamedEntityAnnotation(entity=entity.label_, start=entity.start_char, end=entity.end_char)
                for entity in doc.ents
            ]
            for doc in outputs
        ]

    @property
    def initialized(self) -> bool:
        return self.pipeline is not None

    @property
    def model_name(self) -> str:
        return self._model_name_or_path

    @property
    def device_id(self) -> int:
        return self._device_id

    @contextmanager
    def _select_device(self):
        """
        Context manager used to run spaCy models on a specific
        GPU in a scoped manner.
        """

        # TODO: This won't restore the active device.
        # Since there are no opaque API functions to determine
        # the active device in spaCy/Thinc, we can't do much
        # about it as a consumer unless we start poking into their
        # internals.
        try:
            if self._device_id >= 0:
                spacy.require_gpu(self._device_id)
            yield
        finally:
            if self._device_id >= 0:
                spacy.require_cpu()
