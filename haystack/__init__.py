# pylint: disable=wrong-import-position
# Logging is not configured here on purpose, see https://github.com/deepset-ai/haystack/issues/2485

import sys
from importlib import metadata

__version__: str = str(metadata.version("farm-haystack"))

from generalimport import generalimport, MissingOptionalDependency, FakeModule

generalimport(
    # "pydantic", # Required for all dataclasses
    "pandas",
    "numpy",
    "requests",  # Used in dcdocumentstore, probably can be removed with some work
    "transformers",  # Used in feature_extraction.py, can be removed too with some work
    "tenacity",  # Probably needed because it's a decorator, to be evaluated
    "PIL",  # something tricky going on with transformers :(
    "yaml",
    "torch",
    "protobuf",
    "nltk",
    "rank_bm25",
    "sklearn",
    "dill",
    "tqdm",
    "networkx",
    "mmh3",
    "quantulum3",
    "posthog",
    "azure",
    "huggingface_hub",
    "tenacity",
    "sseclient",
    "boilerpy3",
    "more_itertools",
    "docx",
    "langdetect",
    "tika",
    "sentence_transformers",
    "elasticsearch",
    "tiktoken",
    "jsonschema",
    "canals",
    "events",
    "sqlalchemy",
    "psycopg2",
    "faiss",
    "pymilvus",
    "weaviate",
    "pinecone",
    "SPARQLWrapper",
    "rdflib",
    "opensearchpy",
    "whisper",
    "beir",
    "selenium",
    "webdriver_manager",
    "beautifulsoup4",
    "markdown",
    "frontmatter",
    "magic",
    "fitz",
    "pytesseract",
    "pdf2image",
    "onnxruntime",
    "onnxruntime_tools",
    "scipy",
    "rapidfuzz",
    "seqeval",
    "mlflow",
    "ray",
    "ray",
    "aiorwlock",
)


#
# Temporary patch: remove once https://github.com/ManderaGeneral/generalimport/pull/25 is merged
# and the updated generalimport is released.
#
def is_imported(module_name: str) -> bool:
    """
    Returns True if the module was actually imported, False, if generalimport mocked it.
    """
    try:
        return not isinstance(sys.modules.get(module_name), FakeModule)
    except MissingOptionalDependency as exc:
        # isinstance() raises MissingOptionalDependency: fake module
        pass
    return False


from haystack.schema import Document, Answer, Label, MultiLabel, Span, EvaluationResult
from haystack.nodes.base import BaseComponent
from haystack.pipelines.base import Pipeline
from haystack.environment import set_pytorch_secure_model_loading


# Enables torch's secure model loading through setting an env var.
# Does not use torch.
set_pytorch_secure_model_loading()
