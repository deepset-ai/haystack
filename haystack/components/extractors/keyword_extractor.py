import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import Any, Dict, List, Optional, Union

from haystack import ComponentError, DeserializationError, Document, component, default_from_dict, default_to_dict
from haystack.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install git+https://github.com/LIAAD/yake'") as yake_import:
    import yake

with LazyImport(message="Run 'pip install keybert'") as keybert_import:
    from keybert import KeyBERT


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
                    f"Invalid backend `{value}` for keyword extractor. " f"Supported backends: {supported_backends}"
                )
        else:
            return EnumMeta.__call__(  # pylint: disable=too-many-function-args
                cls, value, names, module, qualname, type, start
            )


class KeywordExtractorBackend(Enum, metaclass=_BackendEnumMeta):
    """
    Enumeration of supported NLP backends for keyword extraction.
    """

    #: Uses YAKE model
    YAKE = "yake"

    #: Uses KeyBERT model
    KEYBERT = "keybert"


@dataclass
class KeywordAnnotation:
    """
    Describes a single keyword annotation.

    :param keyword:
        keyword label.
    :param positions:
        List of start indices of the keyword occurrences in the document.
    :param score:
        Score calculated by the model.
    """

    keyword: str
    positions: List[int]
    score: Optional[float] = None


@component
class KeywordExtractor:
    """
    Annotates keywords in a collection of documents using specified NLP backends.
    The component supports two backends: YAKE and KeyBERT. The former can be used
    with custom parameters supported by YAKE, which are detailed in the
    [YAKE GitHub repository](https://github.com/LIAAD/yake/tree/master).
    The latter can be used with any model from the collection of sentence transformers,
    available at [Sentence Transformers Pretrained Models](https://www.sbert.net/docs/pretrained_models.html),
    and allows for the use of custom parameters specific to KeyBERT.
    Annotations are stored as metadata in the documents.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.extractors.keyword_extractor import KeywordExtractor

    documents = [
        Document(content="I'm Merlin, the happy pig!"),
        Document(content="My name is Clara and I live in Berkeley, California."),
    ]
    extractor = KeywordExtractor(backend="keybert", model="all-MiniLM-L6-v2")
    extractor.warm_up()
    results = extractor.run(documents=documents)["documents"]
    annotations = [KeywordExtractor.get_stored_annotations(doc) for doc in results]
    print(annotations)
    ```
    """

    _METADATA_KEY = "extracted_keywords"

    def __init__(self, *, backend: Union[str, KeywordExtractorBackend], **kwargs) -> None:
        """
        Create a Keyword Extractor component.

        :param backend:
            Backend to use for keyword extraction.
        :param **kwargs:
            Optional keyword arguments passed to the backend.

        """

        if isinstance(backend, str):
            backend = KeywordExtractorBackend(backend)

        self._backend: _KeywordExtractorBackend
        self._kwargs = kwargs

        if backend == KeywordExtractorBackend.YAKE:
            yake_config = YakeConfig(**kwargs)
            self._backend = _YakeBackend(yake_config)

        elif backend == KeywordExtractorBackend.KEYBERT:
            keybert_config = KeyBERTConfig(**kwargs)
            self._backend = _KeyBERTBackend(keybert_config)
        else:
            raise ComponentError(f"Unknown KeywordExtractor backend '{type(backend).__name__}' for extractor")

    def warm_up(self):
        """
        Initialize the component.

        :raises ComponentError:
            If the backend fails to initialize successfully.
        """
        try:
            self._backend.initialize()
        except Exception as e:
            raise ComponentError(f"Keyword extractor with backend '{self._backend.type} failed to initialize.") from e

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Annotate keywords in each document and store
        the annotations in the document's metadata.

        :param documents:
            Documents to process
        :returns:
            Processed documents.
        :raises ComponentError:
            If the backend fails to process a document.
        """
        texts = [doc.content if doc.content is not None else "" for doc in documents]
        annotations = self._backend.annotate(texts)

        if len(annotations) != len(documents):
            raise ComponentError(
                "KeywordExtractor backend did not return the correct number of annotations; "
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
        return default_to_dict(self, backend=self._backend.type, **self._kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeywordExtractor":
        """
        Deserializes the component from a dictionary.
        :param data: Dictionary to deserialize from.
        :returns: Deserialized component.
        """
        try:
            return default_from_dict(cls, data)
        except Exception as e:
            raise DeserializationError(f"Couldn't deserialize {cls.__name__} instance") from e

    @property
    def initialized(self) -> bool:
        """
        Returns if the extractor is ready to annotate text.
        """
        return self._backend.initialized if self._backend else False

    @classmethod
    def get_stored_annotations(cls, document: Document) -> Optional[List[KeywordAnnotation]]:
        """
        Returns the document's keyword annotations stored in its metadata, if any.
        :param document: Document whose annotations are to be fetched.
        :returns: The stored annotations or None if no annotations exist.
        """
        return document.meta.get(cls._METADATA_KEY, None)


class _KeywordExtractorBackend(ABC):
    """
    Base class for keyword extraction backends.
    """

    def __init__(self, type: KeywordExtractorBackend, **kwargs) -> None:
        super().__init__()

        self._type = type
        self._model = None

    @abstractmethod
    def initialize(self):
        """
        Initializes the backend. This would usually
        entail loading models, setting parameters, etc.
        """

    @property
    def initialized(self) -> bool:
        """
        Returns if the backend has been initialized, i.e,
        ready to annotate keywords in the text.
        """
        return self._model is not None

    @abstractmethod
    def annotate(self, texts: List[str]) -> List[List[KeywordAnnotation]]:
        """
        Predict keyword annotations for a collection of documents.
        :param texts:
            Raw texts to be annotated.
        :returns:
            Keyword annotations.
        """

    @property
    def type(self) -> KeywordExtractorBackend:
        """
        Returns the type of the backend.
        """
        return self._type


def initialize_config(config, kwargs):
    """
    Initializes a configuration object with values provided in a dictionary.
    This function dynamically sets attributes on the config object based on
    the provided kwargs dictionary.

    :param config: A config data class instance (YAKEConfig or KeyBERTConfig).
    :param kwargs: A dictionary of parameters intended to update the config object.

    :raises ComponentError: If a key in `kwargs` does not correspond to an attribute
                            in `config`'s annotations.

    """
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            valid_keys = list(config.__annotations__.keys())
            raise ComponentError(f"Invalid parameter '{key}'. Allowed parameters are: {', '.join(valid_keys)}")


@dataclass
class YakeConfig:
    """
    Configuration settings for the YAKE keyword extraction backend.

    :param lan:
        The language of the text to be processed.
    :param n:
        The maximum size of the n-grams to consider for keyword extraction.
    :param dedupLim:
        The threshold for keyword deduplication, controlling how much keyword overlap is allowed.
    :param dedupFunc:
        The algorithm used for deduplication.
    :param windowsSize:
        The size of the window within which to consider words as candidate keywords.
    :param top:
        The maximum number of keywords to extract from the text.
    """

    lan: str = "en"
    n: int = 3
    dedupLim: float = 0.9
    dedupFunc: str = "seqm"
    windowsSize: int = 1
    top: int = 20

    def __init__(self, **kwargs):
        initialize_config(self, kwargs)


class _YakeBackend(_KeywordExtractorBackend):
    """
    YAKE based backend for keyword extraction.
    """

    def __init__(self, config: YakeConfig) -> None:
        """
        Construct a Keyword Extractor backend using YAKE.
        :param config:
            A configuration object containing parameters for the YAKE keyword extractor.
        """
        super().__init__(KeywordExtractorBackend.YAKE)

        self.config = config

    def initialize(self):
        """
        Initializes the Yake keyword extractor.
        """
        self._model = yake.KeywordExtractor(
            lan=self.config.lan,
            n=self.config.n,
            dedupLim=self.config.dedupLim,
            dedupFunc=self.config.dedupFunc,
            windowsSize=self.config.windowsSize,
            top=self.config.top,
        )

    def annotate(self, texts: List[str]) -> List[List[KeywordAnnotation]]:
        if not self.initialized:
            raise ComponentError("YAKE keyword extractor was not initialized - Did you call `initialize()`?")
        assert self._model is not None

        return [
            [
                KeywordAnnotation(
                    keyword=keyword, positions=[m.start() for m in re.finditer(re.escape(keyword), text)], score=score
                )
                for keyword, score in self._model.extract_keywords(text)
            ]
            for text in texts
        ]


@dataclass
class KeyBERTConfig:
    """
    Configuration settings for the KeyBERT keyword extraction backend.

    :param model:
        The model name or a path to a model used by KeyBERT, which is based on sentence-transformers.
    :param keyphrase_ngram_range:
        A tuple indicating the range of n-grams to consider for keyword extraction.
    :param num_of_keywords:
        The number of top keywords to extract from each document.
    :param stop_words:
        Stop words to be excluded during keyword extraction. Can be a list or 'english'.
    :param use_maxsum:
        Whether to use the Max Sum Similarity for diverse keyword extraction.
    :param use_mmr:
        Whether to use Maximal Marginal Relevance to balance relevance and diversity.
    :param diversity:
        Controls the diversity of returned keywords when using MMR or Max Sum.
    :param nr_candidates:
        Number of candidate keywords to consider for diversity calculation.
    """

    model: str = "all-MiniLM-L6-v2"
    keyphrase_ngram_range: tuple = (1, 3)
    num_of_keywords: int = 10
    stop_words: str = "english"
    use_maxsum: bool = False
    use_mmr: bool = False
    diversity: float = 0.5
    nr_candidates: int = 20

    def __init__(self, **kwargs):
        initialize_config(self, kwargs)


class _KeyBERTBackend(_KeywordExtractorBackend):
    """
    KeyBERT backend for keyword extraction.
    """

    def __init__(self, config: KeyBERTConfig) -> None:
        """
        Construct a KeyBERT keyword extraction backend.

        :param config: Configuration object containing parameters for the KeyBERT keyword extractor.
        """
        super().__init__(KeywordExtractorBackend.KEYBERT)
        self.config = config

    def initialize(self):
        """
        Initializes the KeyBERT model.
        """
        self._model = KeyBERT(model=self.config.model)

    def annotate(self, texts: List[str]) -> List[List[KeywordAnnotation]]:
        """
        Extracts keywords from the provided texts using the initialized KeyBERT model.

        :param texts: A list of text documents from which to extract keywords.
        :returns: A list of lists containing keyword annotations for each text document.

        :raises:
            ComponentError: If the keyword extractor has not been initialized.
        """

        if not self.initialized:
            raise ComponentError("KeyBERT backend was not initialized - Did you call `initialize()`?")
        assert self._model is not None
        return [
            [
                KeywordAnnotation(
                    keyword=keyword,
                    positions=[m.start() for m in re.finditer(re.escape(keyword), text.lower())],
                    score=score,
                )
                for keyword, score in self._model.extract_keywords(
                    text,
                    keyphrase_ngram_range=self.config.keyphrase_ngram_range,
                    stop_words=self.config.stop_words,
                    use_maxsum=self.config.use_maxsum,
                    use_mmr=self.config.use_mmr,
                    diversity=self.config.diversity,
                    nr_candidates=self.config.nr_candidates,
                    top_n=self.config.num_of_keywords,
                )
            ]
            for text in texts
        ]

    @property
    def model_name(self) -> str:
        """
        Return the model name or path used for the KeyBERT model.
        """
        return self.config.model
