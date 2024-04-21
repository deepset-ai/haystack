- Title: keywords extractor component
- Decision driver: Hadis
- Start Date: 2024-04-21
- Proposal PR: (fill in after opening the PR)
- Github Issue or Discussion: [issue 7083](https://github.com/deepset-ai/haystack/issues/7083)

# Summary

The proposal aims to enhance the keyword extraction component in Haystack by introducing multiple backend options, parallel processing capabilities, and robust error handling. This enhancement will provide users with flexible, scalable, and accurate methods for extracting keywords from text data and highlight them.


# Basic example

```python
from haystack import Document
from haystack.components.extractors import KeywordsExtractor, KeywordsExtractorBackend

# Initialize the KeywordsExtractor with the YAKE backend
extractor = KeywordsExtractor(backend="sentence_transformer")
extractor.warm_up()

# Define documents
documents = [
    Document(content="Supervised learning is the machine learning task"),
    Document(content="A supervised learning algorithm analyzes the training data."),
]

# Extract keywords from documents
result = extractor.run(documents)["documents"]

# Retrieve keywords for each document
keywords = [KeywordsExtractor.get_stored_annotations(doc) for doc in result]
print(keywords)
```

# Motivation

Keyword extraction is a fundamental task in natural language processing, supporting various use cases such as document summarization, information retrieval, and content analysis. There are different approach to extract keywords including unsupervised approaches (TF.IDF, YAKE) and using pretrained models (such as sentence-transformer). The goal of this proposal is to introduce a component that allow user to extract keyword with its preferred method and add it to his pipeline.

# Detailed design

The provided code defines a Python class KeywordsExtractor that is used to extract keywords from a list of documents. This class supports two backends for keyword extraction: YAKE and sentence_transformer. sentence_transformer uses KeyBert package in background (more detail in [implementation datial](#implementation-detail))). User could use any sentence_transformer model to extract keywords, as default, it uses `all-MiniLM-L6-v2`.
For sake of backward compatibility with different approaches, user could pass any acceptable parameter from Yake, KeyBert and sentence-transfommer as `backend_kwargs` . For example, in the following code `model` is accepted parameter from `KeyBert` and `model_name_or_path` is compatible with `sentence-transformer`.

Both `yake` and `keyBert` support highlighting keywords. Having a highlighted text could give a good value to the end user. That's why Instead of returning the position of keywords, `KeywordExtractor` returns `keywords` and `highlights` as metadata for **each document**:

````python
metadata: {
"keywords": [KeyWordsSelection(entity="machine learning task", score=0.026288458458349206)],
"highlight": HighlightedText(
    text=(
        "Supervised learning is the <kw>machine learning task</kw> of learning afunction that maps an "
        "input to an output based on example input-output pairs."
    )
),
},
````

```python
extractor = KeywordsExtractor(backend="sentence_transformer", backend_kwargs={"model":"paraphrase-multilingual-MiniLM-L12-v2"})

extractor = KeywordsExtractor(backend="sentence_transformer", backend_kwargs={"model_name_or_path":"paraphrase-multilingual-MiniLM-L12-v2"})
```

## Implementation detail

*You can skip this section if you are primarily interested in user experience.*
`KeyWordExtractor` support different backends (at the moment 2). User must define the backend when he intiates `KeywordsExtractor`. allowed backends is defined in `KeywordsExtractorBackend` and validated when user instantiates `KeyWordExtractor`

```python
class KeywordsExtractorBackend(Enum, metaclass=_BackendkwEnumMeta):

    SENTENCETRANSFORMER = "sentence_transformer"
    YAKE = "yake"

```

Instead of considering KeyBert as a backend name *sentence_transformer* is chosen because, then it gives Haystack to add other Embedding models, such as Spacy, Gensim, etc, as a supported method.

Each backend has its own Class for extracting & highlighting keywords:

- `_YakeBackend(_KWExtracorBackend)`
- `_KeyBertBackend(_KWExtracorBackend)`

For being able to add different models to `KeyBert`, the model should be loaded first and then feed to the KeyBert. For example [[ref](https://github.com/MaartenGr/KeyBERT?tab=readme-ov-file#25-embedding-models)]:

```python
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=sentence_model)
```

This brings the challenge of how impelent the component to be easy to improve. For tackling this challenge, a new 'ABC' class introduced. Any new model would be inheritated from this class and all specific implementation such as importing required package and loading the model would be handled here:

```python
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
```

`KeywordsExtractor` get chosen backend and model with `_SupportedBackendModel` class. The purpose of this class is for better maintainability and easy way to add new method without need to modify `KeywordsExtractor`.

```python
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
```

This way for supporting new Embedding model we only need to:

- Implement a new sub class of `_KeyBertModel`
- Add desired name to `KeywordsExtractorBackend`
- update `_models_dict` and `_backend_dict` in `_SupportedBackendModel`

Following picture shows class diagram for `KeyWordExtractor`

# Drawbacks

This is a new component and it uses regular packages which are used in some other component, so it doesn't have breaking change. Also implementation is compatible Haystack code base.

# Alternatives

# Adoption strategy

No adaptation strategy is needed. The implementation doesn't have any breaking change.

# How we teach this

*How to use the component* must be added to the documentation. However, structure design and usage follow same pattern with `NamedEntityExtractor`. So it would be easier for users and developers to use or maintain any extractor component in extroctor module.

# Unresolved questions
