# pylint: disable=wrong-import-position
# Logging is not configured here on purpose, see https://github.com/deepset-ai/haystack/issues/2485

import sys
from importlib import metadata

__version__: str = str(metadata.version("farm-haystack"))

from generalimport import generalimport, MissingOptionalDependency, FakeModule

generalimport(
    # "pydantic", # Required for all dataclasses
    # "tenacity",  # Probably needed because it's a decorator, to be evaluated
    # "pandas",
    "aiorwlock",
    "azure",
    "beautifulsoup4",
    "beir",
    "boilerpy3",
    "canals",
    "dill",
    "docx",
    "elasticsearch",
    "events",
    "faiss",
    "fitz",
    "frontmatter",
    "huggingface_hub",
    "jsonschema",
    "langdetect",
    "magic",
    "markdown",
    "mlflow",
    "more_itertools",
    "networkx",
    "nltk",
    "numpy",
    "onnxruntime",
    "onnxruntime_tools",
    "opensearchpy",
    "pdf2image",
    "PIL",
    "pinecone",
    "posthog",
    "protobuf",
    "psycopg2",
    "pytesseract",
    "quantulum3",
    "rank_bm25",
    "rapidfuzz",
    "ray",
    "rdflib",
    "requests",
    "scipy",
    "selenium",
    "sentence_transformers",
    "seqeval",
    "sklearn",
    "SPARQLWrapper",
    "sqlalchemy",
    "sseclient",
    "tenacity",
    "tika",
    "tiktoken",
    "tokenizers",
    "torch",
    "tqdm",
    "transformers",
    "weaviate",
    "webdriver_manager",
    "whisper",
    "yaml",
)


# TODO: remove this function once this PR is merged and released by generalimport:
# https://github.com/ManderaGeneral/generalimport/pull/25
def is_imported(module_name: str) -> bool:
    """
    Returns True if the module was actually imported, False, if generalimport mocked it.
    """
    module = sys.modules.get(module_name)
    try:
        return bool(module) and not isinstance(module, FakeModule)
    except MissingOptionalDependency:
        # isinstance() raises MissingOptionalDependency: fake module
        pass
    return False


from haystack.schema import Document, Answer, Label, MultiLabel, Span, EvaluationResult, TableCell
from haystack.nodes.base import BaseComponent
from haystack.pipelines.base import Pipeline
from haystack.environment import set_pytorch_secure_model_loading
from haystack.mmh3 import hash128


# Enables torch's secure model loading through setting an env var.
# Does not use torch.
set_pytorch_secure_model_loading()
