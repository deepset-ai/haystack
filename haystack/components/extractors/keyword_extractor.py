import concurrent.futures
import inspect
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import Any, Dict, List, Optional, Union

from tqdm import tqdm

from haystack import ComponentError, DeserializationError, Document, component, default_from_dict, default_to_dict
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, Secret

with LazyImport(message="Run 'pip install yake'") as yake_import:
    import yake
    from yake.highlight import TextHighlighter
with LazyImport(message="Run 'pip install keybert'") as keybert_import:
    from keybert import KeyBERT
with LazyImport(message="Run 'pip install \"sentence-transformers>=2.2.0\"'") as sentence_transformers_import:
    from sentence_transformers import SentenceTransformer


class _BackendkwEnumMeta(EnumMeta):
    """
    Metaclass for fine-grained error handling of backend enums.
    """

    def __init__(cls, clsname, bases, clsdict):
        supported_backends = {v.value: v for v in cls if isinstance(v, Enum)}
        cls._supported_backends = supported_backends
        super().__init__(clsname, bases, clsdict)

    def __call__(cls, value, *args, **kwargs):
        if (
            isinstance(value, Enum)
            and value not in cls._supported_backends.values()
            or isinstance(value, str)
            and value not in cls._supported_backends.keys()
        ):
            raise ComponentError(
                f"Invalid backend `{value}` for keyword extractor. "
                f"Supported backends: {cls._supported_backends.keys()}"
            )
        return super().__call__(value, *args, **kwargs)


class KeywordsExtractorBackend(Enum, metaclass=_BackendkwEnumMeta):
    """
    Enum for specifying the backend to use for keyword extraction.
    """

    #: Uses keyBert with sentence-transfomer model.
    SENTENCETRANSFORMER = "sentence_transformer"

    #: Uses the yake package.
    YAKE = "yake"


@dataclass
class KeyWordsSelection:
    """Represents a keyword selection.

    Attributes:
        entity (str): The keyword entity.
        start (int, optional): The start position of the keyword in the text.
        end (int, optional): The end position of the keyword in the text.
        score (float, optional): The score associated with the keyword.
    """

    entity: str
    score: Optional[float] = None


@dataclass
class HighlightedText:
    """Represents a highlighted text.

    Attributes:
        text (str): The highlighted text.
    """

    text: str = Optional[str]


@component
class KeywordsExtractor:
    """
    Extracts keywords from a list of documents using different backends.
    The component supports the following backends:
    - YAKE: Uses the yake package to extract keywords.
    - sentence_transformer: Uses the keyBert package and sentence_transformer model to extract keywords.

    Usage example:
    ```python
    from pprint import pprint
    from haystack.dataclasses import Document
    from haystack.components.extractors import KeywordsExtractor, KeywordsExtractorBackend

    documents = [
        Document(content="Supervised learning is the machine learning task"),
        Document(content="A supervised learning algorithm analyzes the training data."),
    ]

    extractor = KeywordsExtractor(backend=KeywordsExtractorBackend.YAKE)
    extractor.warm_up()
    result = extractor.run(documents)["documents"]

    keywords = [KeywordsExtractor.get_stored_annotations(doc) for doc in result]
    pprint(keywords)
    ```
    """

    _METADATA_KEYWORDS_KEY = "keywords"
    _METADATA_HIGHLIGHTS_KEY = "highlight"

    def __init__(
        self,
        *,
        backend: Union[str, KeywordsExtractorBackend],
        backend_kwargs: Optional[Dict[str, Any]] = None,
        top_n: int = 3,
        max_ngram_size: int = 3,
    ) -> None:
        """
        create keyword extractor component.
        :param backend:
            The backend to use for keyword extraction.
            It can be either a string representing the backend name or an instance of `KeywordsExtractorBackend`.
        :param backend_kwargs:
            Additional keyword arguments to pass to the backend.
        :param top_n:
            The number of keywords to extract from each document. Defaults to 3.
        :param max_ngram_size:
            The maximum size of n-grams to consider when extracting keywords.
            If backend accepts a range for n-grams, by default, this will be used as both the upper and lower bound.
            This is also needed for highliting keywords. Defaults to 3.

        :raises ComponentError:
            If an unknown keyword backend is provided.
        """
        if isinstance(backend, str):
            backend = KeywordsExtractorBackend(backend)
        # Ignore the backend_kwargs if it is not a dictionary
        backend_kwargs = backend_kwargs if isinstance(backend_kwargs, dict) else {}
        selected_backend = _SupportedBackendModel.get_backend(backend.value)

        self._backend: _KWExtracorBackend = selected_backend(
            type=backend, backend_kwargs=backend_kwargs, top_n=top_n, max_ngram_size=max_ngram_size
        )

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document], concurrent_workers=10) -> Dict[str, Any]:
        """
        Run the keyword extractor component on a list of documents.

        :param documents:
            A list of Document objects to extract keywords from.
        Raises:
            ComponentError: If the keyword extractor backend does not return the correct number of annotations.
        :returns:
            Dict[str, Any]: A dictionary containing the extracted keywords for each document.
        """
        result = defaultdict(list)
        #
        doc: Document
        texts = {doc.id: doc.content if doc.content is not None else "" for doc in documents}

        with tqdm(total=len(texts)) as pbar, concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrent_workers
        ) as executor:
            future_to_dataset = {executor.submit(self._backend.extract, doc): doc_id for doc_id, doc in texts.items()}
            for future in concurrent.futures.as_completed(future_to_dataset):
                doc_id = future_to_dataset[future]
                result[doc_id] = (future.result(), self._backend.highlight(texts[doc_id]))

                pbar.update(1)

        for doc in documents:
            doc.meta[self._METADATA_KEYWORDS_KEY] = result[doc.id][0]
            doc.meta[self._METADATA_HIGHLIGHTS_KEY] = result[doc.id][1]

        return {"documents": documents}

    def __iter__(self):
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """

        dict_ = default_to_dict(self, backend=self._backend._type, backend_kwargs=self._backend._backend_kwargs)
        for key, value in dict_.items():
            yield key, value

    def warm_up(self):
        """
        Initializes the keyword extractor.

        Raises:
            ComponentError: If the keyword extractor fails to initialize.
        """
        try:
            self._backend.initialize()
        except Exception as e:
            raise ComponentError(f"Keywords extractor with backend '{self.type} failed to initialize.") from e

    # Method's just kept for the sake of compatibility with the rest of the codebase
    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return dict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KeywordsExtractor":
        """Deserialize a KeywordsExtractor instance from a dictionary.

        :param data:
            A dictionary containing the serialized representation of a KeywordsExtractor instance.

        Raises:
            DeserializationError: If there is an error during deserialization.

        :returns:
            KeywordsExtractor: A deserialized KeywordsExtractor instance.
        """
        try:
            return default_from_dict(cls, data)
        except Exception as e:
            raise DeserializationError(f"Couldn't deserialize {cls.__name__} instance") from e

    @property
    def initialized(self) -> bool:
        """
        Returns if the extractor is ready to extract keywords.
        """
        return self._backend.initialized

    @property
    def type(self) -> KeywordsExtractorBackend:
        """
        Returns the type of the backend.
        """
        return self._backend._type

    @property
    def top_n(self) -> KeywordsExtractorBackend:
        """
        Returns the type of the backend.
        """
        return self._backend._top_n

    @classmethod
    def get_stored_annotations(cls, document: Document) -> Optional[Dict[str, Union[List[KeyWordsSelection], str]]]:
        """
        Returns the document's keywords stored
        in its metadata, if any.

        :param document:
            Document whose annotations are to be fetched.
        :returns:
            a dictionary containing the keywords and highlights.
        """

        return {
            cls._METADATA_KEYWORDS_KEY: document.meta.get(cls._METADATA_KEYWORDS_KEY),
            cls._METADATA_HIGHLIGHTS_KEY: document.meta.get(cls._METADATA_HIGHLIGHTS_KEY),
        }


class _KWExtracorBackend(ABC):
    """
    This class represents the abstract base class for keyword extractor backends.

    :param data:
        ABC (type): The base class for defining abstract base classes.
    """

    def __init__(
        self, type: KeywordsExtractorBackend, top_n: int, max_ngram_size: int, backend_kwargs: Optional[Dict[str, Any]]
    ) -> None:
        super().__init__()
        self._type = type
        self._top_n = top_n
        self._max_ngram_size = max_ngram_size
        self._backend_kwargs = backend_kwargs
        self._keywords: dict = {}

    @property
    @abstractmethod
    def initialized(self) -> bool:
        """
        Returns if the backend has been initialized, i.e, ready to extract keywords.
        """

    @abstractmethod
    def extract(self, texts: Dict[str, str], max_workers: int) -> List[KeyWordsSelection]:
        """
        Annotates a multiple texts and returns a list of keywords for each text.

        :param texts:
            A dictionary including document_id as key and text as value.
        :param max_workers:
            The maximum number of workers to use for parallel processing.

        :returns:
            list of keywords
        """

    @abstractmethod
    def highlight(self, text: str, keywords: List[str]) -> HighlightedText:
        """
        Highlights the keywords in the text.

        :param text:
            The text to highlight.
        :param keywords:
            The keywords to highlight.

        :returns:
            The highlighted text.
        """

    @abstractmethod
    def initialize(self):
        """
        Initializes the backend.
        This would usually entail loading models, etc.
        """


class _YakeBackend(_KWExtracorBackend):
    """It uses yake package for extracting keywords for documents."""

    def __init__(
        self, type: KeywordsExtractorBackend, top_n: int, max_ngram_size: int, backend_kwargs: Optional[Dict[str, Any]]
    ) -> None:
        """
        Initialize the Yake KeywordExtractor.

        :param top_n:
            The number of top keywords to extract. Defaults to 3.
        :param backend_kwargs:
            Additional keyword arguments to pass to the backend. Defaults to None.
        """
        super().__init__(type, top_n, max_ngram_size, backend_kwargs)
        yake_import.check()

    def initialize(self):
        pass

    @property
    def initialized(self) -> bool:
        return True

    def extract(self, text: str) -> List[KeyWordsSelection]:
        "extract keywords from a list of documents using Yake backend."
        extractor = yake.KeywordExtractor(top=self._top_n, n=self._max_ngram_size, **self._backend_kwargs)

        self._keywords = extractor.extract_keywords(text)
        return [KeyWordsSelection(entity=record[0], score=record[1]) for record in self._keywords]

    def highlight(self, text: str) -> HighlightedText:
        """
        Highlights the keywords in the text.

        :param text:
            The text to highlight.
        :param keywords:
            The keywords to highlight.

        :returns:
            The highlighted text.
        """
        highlighter = TextHighlighter(max_ngram_size=self._max_ngram_size)

        return HighlightedText(highlighter.highlight(text, self._keywords))


class _KeyBertBackend(_KWExtracorBackend):
    """It uses KeyBert package for extracting keywords for documents."""

    def __init__(
        self, type: KeywordsExtractorBackend, top_n: int, max_ngram_size: int, backend_kwargs: Optional[Dict[str, Any]]
    ) -> None:
        """
        Initialize the KeyBert KeywordExtractor.

        :param top_n:
            The number of top keywords to extract. Defaults to 3.
        :param backend_kwargs:
            Additional keyword arguments to pass to the backend. Defaults to {}.
            If you want to pass arguments to the SentenceTransformer model, you can pass them here.
        """
        # check required imports
        keybert_import.check()

        # separate the KeyBert parameters and the model parameters
        _model_name = None
        _model_param = {}
        # Fetch the parameters that are accepted by KeyBERT.extract_keywords
        keybert_param = inspect.signature(KeyBERT.extract_keywords).parameters
        keybert_param = [param.name for param in keybert_param.values()]
        _backend_kwargs = {}
        for key in backend_kwargs.keys():
            if key in ["model"]:
                _model_name = backend_kwargs[key]

            elif key in keybert_param:
                _backend_kwargs[key] = backend_kwargs[key]
            else:
                _model_param[key] = backend_kwargs[key]
        # If keyphrase_ngram_range is not provided, set it based on the max_ngram_size
        if backend_kwargs.get("keyphrase_ngram_range") is None:
            _backend_kwargs["keyphrase_ngram_range"] = (max_ngram_size, max_ngram_size)

        super().__init__(type, top_n, max_ngram_size, backend_kwargs=_backend_kwargs)

        # Initialize the KeyBert model
        self._keybert_model: _KeyBertModel = _SupportedBackendModel.get_model_type(self._type.value)(
            _model_param, model_name=_model_name
        )

    def initialize(self):
        """This method initializes sentence transformer model and then ataches KeyBert to it."""
        if not self._keybert_model._initialized:
            self._keybert_model._initialize()

    @property
    def initialized(self) -> bool:
        return self._keybert_model._initialized

    @property
    def model_name(self):
        return self._keybert_model._model_name

    def extract(self, text: str) -> List[KeyWordsSelection]:
        "extract keywords from a list of documents using KeyBert backend."
        if not self.initialized:
            raise ComponentError(f"{self._keybert_model} was not initialized - Did you call `warm_up()`?")
        extractor = KeyBERT(self._keybert_model._model)
        self._keywords = extractor.extract_keywords(text, top_n=self._top_n, **self._backend_kwargs)
        return [KeyWordsSelection(entity=record[0], score=record[1]) for record in self._keywords]

    def highlight(self, text: str) -> HighlightedText:
        return HighlightedText("")


class _KeyBertModel(ABC):
    """This class represents the abstract base class for different KeyBert models."""

    @abstractmethod
    def __init__(
        self,
        _model_name: Optional[str] = None,
        _model_kwargs: Optional[dict] = {},
        _device: Optional[ComponentDevice] = None,
        _token: Optional[Secret] = None,
    ) -> None:
        super().__init__()
        self._model_name = _model_name
        self._model_kwargs = _model_kwargs
        self._device = _device
        self._token = _token
        self._model = None

    @abstractmethod
    def _initialize(self):
        pass

    @property
    def _initialized(self) -> bool:
        return self._model is not None


class _SentenceTransformerModel(_KeyBertModel):
    """A class representing a Sentence Transformer model.

    This class extends the `_KeyBertModel` class and provides functionality to initialize and use a Sentence Transformer model.

    :param model_kwargs:
        A dictionary containing the keyword arguments to be passed to the Sentence Transformer model.
    :param model_name:
        The name or path of the Sentence Transformer model. If not provided, a default model will be used.
    """

    def __init__(self, model_kwargs: dict, model_name: Optional[str] = None) -> None:
        sentence_transformers_import.check()
        super().__init__(model_name or model_kwargs.pop("model_name_or_path", "all-MiniLM-L6-v2"))
        self._model: Optional[SentenceTransformer] = None

        self._token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False)

        for key in model_kwargs.keys():
            if key == "device":
                self._device = model_kwargs[key]
            elif key == "token":
                self._token = model_kwargs[key]
            else:
                self._model_kwargs[key] = model_kwargs[key]
        self._device = ComponentDevice.resolve_device(self._device)

    def _initialize(self):
        """This method initializes the Sentence Transformer model and attaches KeyBert to it."""
        if self._model is None:
            self._model = SentenceTransformer(
                model_name_or_path=self._model_name,
                device=self._device.to_torch_str(),
                use_auth_token=self._token.resolve_value() if self._token else None,
                **self._model_kwargs,
            )


class _SupportedBackendModel:
    """
    This class is used to get the supported models for KeyBert.
    """

    _models_dict = {KeywordsExtractorBackend.SENTENCETRANSFORMER.value: _SentenceTransformerModel}
    _backend_dict = {
        KeywordsExtractorBackend.SENTENCETRANSFORMER.value: _KeyBertBackend,
        KeywordsExtractorBackend.YAKE.value: _YakeBackend,
    }

    @staticmethod
    def get_model_type(key: str) -> _KeyBertModel:
        return _SupportedBackendModel._models_dict.get(key, None)

    @staticmethod
    def get_backend(key: str) -> _KWExtracorBackend:
        return _SupportedBackendModel._backend_dict[key]
