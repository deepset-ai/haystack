import concurrent.futures
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, EnumMeta
from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm

from haystack import ComponentError, DeserializationError, Document, component, default_from_dict, default_to_dict
from haystack.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install yake'") as yake_import:
    import yake
    from yake.highlight import TextHighlighter


class _BackendkwEnumMeta(EnumMeta):
    """
    Metaclass for fine-grained error handling of backend enums.
    """

    def __call__(cls, value, names=None, *, module=None, qualname=None, type=None, start=1):
        if names is None:
            try:
                return EnumMeta.__call__(cls, value, names, module=module, qualname=qualname, type=type, start=start)
            except ValueError:
                supported_backends = ", ".join(sorted(v.value for v in cls))
                raise ComponentError(
                    f"Invalid backend `{value}` for keyword extractor. " f"Supported backends: {supported_backends}"
                )
        else:
            return EnumMeta.__call__(  # pylint: disable=too-many-function-args
                cls, value, names, module, qualname, type, start
            )


class KeywordsExtractorBackend(Enum, metaclass=_BackendkwEnumMeta):
    """
    Enum for specifying the backend to use for keyword extraction.
    """

    #: Uses keyBert with sentence-transfomer model.
    KEYBERT = "keybert"

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

    text: str


@component
class KeywordsExtractor:
    """
    Extracts keywords from a list of documents using different backends.
    The component supports the following backends:
    - YAKE: Uses the yake package to extract keywords.

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

    :param backend:
        The backend to use for keyword extraction.
        It can be either a string representing the backend name or an instance of `KeywordsExtractorBackend`.
    :param backend_kwargs:
        Additional keyword arguments to pass to the backend.
    :param top_n:
        The number of keywords to extract from each document. Defaults to 3.
    :param max_ngram_size:
        The maximum size of n-grams to consider when extracting keywords.
        This is also needed for highliting keywords. Defaults to 3.

    :raises ComponentError:
        If an unknown keyword backend is provided.
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
        Initialize the KeywordExtractor.

        :param data:
            backend (Union[str, KeywordsExtractorBackend]): The backend to use for keyword extraction.
                It can be either a string representing the backend name or an instance of `KeywordsExtractorBackend`.
        :param backend_kwargs:

        Raises:
            ComponentError: If an unknown keyword backend is provided.
        """
        if isinstance(backend, str):
            backend = KeywordsExtractorBackend(backend)

        if backend == KeywordsExtractorBackend.KEYBERT:
            raise ComponentError(f"'{type(backend).__name__}' is not ready yet")
        elif backend == KeywordsExtractorBackend.YAKE:
            self._backend = _YakeBackend(backend_kwargs=backend_kwargs, top_n=top_n, max_ngram_size=max_ngram_size)
        else:
            raise ComponentError(f"Unknown keyword backend '{type(backend).__name__}' for extractor")

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

    # It's just kept the method for the sake of compatibility with the rest of the codebase
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
        self._backend_kwargs = backend_kwargs if isinstance(backend_kwargs, dict) else {}
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

    def __init__(self, *, top_n: int, max_ngram_size: int, backend_kwargs: Optional[Dict[str, Any]]) -> None:
        """
        Initialize the Yake KeywordExtractor.

        :param top_n:
            The number of top keywords to extract. Defaults to 3.
        :param backend_kwargs:
            Additional keyword arguments to pass to the backend. Defaults to None.
        """
        super().__init__(KeywordsExtractorBackend.YAKE, top_n, max_ngram_size, backend_kwargs)
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
