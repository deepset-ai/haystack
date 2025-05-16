# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from haystack import ComponentError, DeserializationError, Document, component, default_from_dict, default_to_dict
from haystack.lazy_imports import LazyImport
from haystack.utils.auth import Secret, deserialize_secrets_inplace
from haystack.utils.device import ComponentDevice
from haystack.utils.hf import deserialize_hf_model_kwargs, resolve_hf_pipeline_kwargs, serialize_hf_model_kwargs

with LazyImport(message="Run 'pip install \"transformers[torch]\"'") as transformers_import:
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    from transformers import Pipeline as HfPipeline

with LazyImport(message="Run 'pip install spacy'") as spacy_import:
    import spacy  # pylint: disable=import-error
    from spacy import Language as SpacyPipeline  # pylint: disable=import-error


class NamedEntityExtractorBackend(Enum):
    """
    NLP backend to use for Named Entity Recognition.
    """

    #: Uses an Hugging Face model and pipeline.
    HUGGING_FACE = "hugging_face"

    #: Uses a spaCy model and pipeline.
    SPACY = "spacy"

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(string: str) -> "NamedEntityExtractorBackend":
        """
        Convert a string to a NamedEntityExtractorBackend enum.
        """
        enum_map = {e.value: e for e in NamedEntityExtractorBackend}
        mode = enum_map.get(string)
        if mode is None:
            msg = (
                f"Invalid backend '{string}' for named entity extractor. "
                f"Supported backends are: {list(enum_map.keys())}"
            )
            raise ComponentError(msg)
        return mode


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
    """
    Annotates named entities in a collection of documents.

    The component supports two backends: Hugging Face and spaCy. The
    former can be used with any sequence classification model from the
    [Hugging Face model hub](https://huggingface.co/models), while the
    latter can be used with any [spaCy model](https://spacy.io/models)
    that contains an NER component. Annotations are stored as metadata
    in the documents.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.extractors.named_entity_extractor import NamedEntityExtractor

    documents = [
        Document(content="I'm Merlin, the happy pig!"),
        Document(content="My name is Clara and I live in Berkeley, California."),
    ]
    extractor = NamedEntityExtractor(backend="hugging_face", model="dslim/bert-base-NER")
    extractor.warm_up()
    results = extractor.run(documents=documents)["documents"]
    annotations = [NamedEntityExtractor.get_stored_annotations(doc) for doc in results]
    print(annotations)
    ```
    """

    _METADATA_KEY = "named_entities"

    def __init__(
        self,
        *,
        backend: Union[str, NamedEntityExtractorBackend],
        model: str,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
    ) -> None:
        """
        Create a Named Entity extractor component.

        :param backend:
            Backend to use for NER.
        :param model:
            Name of the model or a path to the model on
            the local disk. Dependent on the backend.
        :param pipeline_kwargs:
            Keyword arguments passed to the pipeline. The
            pipeline can override these arguments. Dependent on the backend.
        :param device:
            The device on which the model is loaded. If `None`,
            the default device is automatically selected. If a
            device/device map is specified in `pipeline_kwargs`,
            it overrides this parameter (only applicable to the
            HuggingFace backend).
        :param token:
            The API token to download private models from Hugging Face.
        """

        if isinstance(backend, str):
            backend = NamedEntityExtractorBackend.from_str(backend)

        self._backend: _NerBackend
        self._warmed_up: bool = False
        self.token = token
        device = ComponentDevice.resolve_device(device)

        if backend == NamedEntityExtractorBackend.HUGGING_FACE:
            pipeline_kwargs = resolve_hf_pipeline_kwargs(
                huggingface_pipeline_kwargs=pipeline_kwargs or {},
                model=model,
                task="ner",
                supported_tasks=["ner"],
                device=device,
                token=token,
            )

            self._backend = _HfBackend(model_name_or_path=model, device=device, pipeline_kwargs=pipeline_kwargs)
        elif backend == NamedEntityExtractorBackend.SPACY:
            self._backend = _SpacyBackend(model_name_or_path=model, device=device, pipeline_kwargs=pipeline_kwargs)
        else:
            raise ComponentError(f"Unknown NER backend '{type(backend).__name__}' for extractor")

    def warm_up(self):
        """
        Initialize the component.

        :raises ComponentError:
            If the backend fails to initialize successfully.
        """
        if self._warmed_up:
            return

        try:
            self._backend.initialize()
            self._warmed_up = True
        except Exception as e:
            raise ComponentError(
                f"Named entity extractor with backend '{self._backend.type}' failed to initialize."
            ) from e

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], batch_size: int = 1) -> Dict[str, Any]:
        """
        Annotate named entities in each document and store the annotations in the document's metadata.

        :param documents:
            Documents to process.
        :param batch_size:
            Batch size used for processing the documents.
        :returns:
            Processed documents.
        :raises ComponentError:
            If the backend fails to process a document.
        """
        if not self._warmed_up:
            msg = "The component NamedEntityExtractor was not warmed up. Call warm_up() before running the component."
            raise RuntimeError(msg)

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
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialization_dict = default_to_dict(
            self,
            backend=self._backend.type.name,
            model=self._backend.model_name,
            device=self._backend.device.to_dict(),
            pipeline_kwargs=self._backend._pipeline_kwargs,
            token=self.token.to_dict() if self.token else None,
        )

        hf_pipeline_kwargs = serialization_dict["init_parameters"]["pipeline_kwargs"]
        hf_pipeline_kwargs.pop("token", None)

        serialize_hf_model_kwargs(hf_pipeline_kwargs)
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NamedEntityExtractor":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        try:
            deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
            init_params = data.get("init_parameters", {})
            if init_params.get("device") is not None:
                init_params["device"] = ComponentDevice.from_dict(init_params["device"])
            init_params["backend"] = NamedEntityExtractorBackend[init_params["backend"]]

            hf_pipeline_kwargs = init_params.get("pipeline_kwargs", {})
            deserialize_hf_model_kwargs(hf_pipeline_kwargs)
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
        Returns the document's named entity annotations stored in its metadata, if any.

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

    def __init__(
        self,
        _type: NamedEntityExtractorBackend,
        device: ComponentDevice,
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self._type = _type
        self._device = device
        self._pipeline_kwargs = pipeline_kwargs if pipeline_kwargs is not None else {}

    @abstractmethod
    def initialize(self):
        """
        Initializes the backend. This would usually entail loading models, pipelines, and so on.
        """

    @property
    @abstractmethod
    def initialized(self) -> bool:
        """
        Returns if the backend has been initialized, for example, ready to annotate text.
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
    def device(self) -> ComponentDevice:
        """
        The device on which the backend's model is loaded.

        :returns:
            The device on which the backend's model is loaded.
        """
        return self._device

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
        self, *, model_name_or_path: str, device: ComponentDevice, pipeline_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Construct a Hugging Face NER backend.

        :param model_name_or_path:
            Name of the model or a path to the Hugging Face
            model on the local disk.
        :param device:
            The device on which the model is loaded. If `None`,
            the default device is automatically selected.

            If a device/device map is specified in `pipeline_kwargs`,
            it overrides this parameter.
        :param pipeline_kwargs:
            Keyword arguments passed to the pipeline. The
            pipeline can override these arguments.
        """
        super().__init__(NamedEntityExtractorBackend.HUGGING_FACE, device, pipeline_kwargs)

        transformers_import.check()

        self._model_name_or_path = model_name_or_path
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForTokenClassification] = None
        self.pipeline: Optional[HfPipeline] = None

    def initialize(self):
        token = self._pipeline_kwargs.get("token", None)
        self.tokenizer = AutoTokenizer.from_pretrained(self._model_name_or_path, token=token)
        self.model = AutoModelForTokenClassification.from_pretrained(self._model_name_or_path, token=token)

        pipeline_params: Dict[str, Any] = {
            "task": "ner",
            "model": self.model,
            "tokenizer": self.tokenizer,
            "aggregation_strategy": "simple",
        }
        pipeline_params.update({k: v for k, v in self._pipeline_kwargs.items() if k not in pipeline_params})
        self.device.update_hf_kwargs(pipeline_params, overwrite=False)
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


class _SpacyBackend(_NerBackend):
    """
    spaCy backend for NER.
    """

    def __init__(
        self, *, model_name_or_path: str, device: ComponentDevice, pipeline_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Construct a spaCy NER backend.

        :param model_name_or_path:
            Name of the model or a path to the spaCy
            model on the local disk.
        :param device:
            The device on which the model is loaded. If `None`,
            the default device is automatically selected.
        :param pipeline_kwargs:
            Keyword arguments passed to the pipeline. The
            pipeline can override these arguments.
        """
        super().__init__(NamedEntityExtractorBackend.SPACY, device, pipeline_kwargs)

        spacy_import.check()

        self._model_name_or_path = model_name_or_path
        self.pipeline: Optional[SpacyPipeline] = None

        if self.device.has_multiple_devices:
            raise ValueError("spaCy backend for named entity extractor only supports inference on single devices")

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

    @contextmanager
    def _select_device(self):
        """
        Context manager used to run spaCy models on a specific GPU in a scoped manner.
        """

        # TODO: This won't restore the active device.
        # Since there are no opaque API functions to determine
        # the active device in spaCy/Thinc, we can't do much
        # about it as a consumer unless we start poking into their
        # internals.
        device_id = self._device.to_spacy()
        try:
            if device_id >= 0:
                spacy.require_gpu(device_id)
            yield
        finally:
            if device_id >= 0:
                spacy.require_cpu()
