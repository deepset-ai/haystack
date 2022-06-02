# pylint: disable=wrong-import-position,wrong-import-order

from typing import Union
from types import ModuleType

try:
    from importlib import metadata
except (ModuleNotFoundError, ImportError):
    # Python <= 3.7
    import importlib_metadata as metadata  # type: ignore

__version__: str = str(metadata.version("farm-haystack"))


# This configuration must be done before any import to apply to all submodules
import logging

logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", datefmt="%m/%d/%Y %H:%M:%S", level=logging.WARNING
)
logging.getLogger("haystack").setLevel(logging.INFO)

import pandas as pd

from haystack.schema import Document, Answer, Label, MultiLabel, Span, EvaluationResult
from haystack.nodes.base import BaseComponent
from haystack.pipelines.base import Pipeline


pd.options.display.max_colwidth = 80


# ###########################################
# Enable old style imports (temporary)
import sys

logger = logging.getLogger(__name__)

# Wrapper emitting a warning on import
def DeprecatedModule(mod, deprecated_attributes=None, is_module_deprecated=True):
    """
    Return a wrapped object that warns about deprecated accesses at import
    """

    class DeprecationWrapper:
        warned = []

        def __getattr__(self, attr):
            is_a_deprecated_attr = deprecated_attributes and attr in deprecated_attributes
            is_a_deprecated_module = is_module_deprecated and attr not in ["__path__", "__spec__", "__name__"]
            warning_already_emitted = attr in self.warned
            attribute_exists = getattr(mod, attr) is not None

            if (is_a_deprecated_attr or is_a_deprecated_module) and not warning_already_emitted and attribute_exists:
                logger.warn(
                    f"Object '{attr}' is imported through a deprecated path. Please check out the docs for the new import path."
                )
                self.warned.append(attr)
            return getattr(mod, attr)

    return DeprecationWrapper()


# All modules to be aliased need to be imported here

# This self-import is used to monkey-patch, keep for now
import haystack  # pylint: disable=import-self
from haystack import pipelines
from haystack.nodes import (
    connector,
    document_classifier,
    extractor,
    file_converter,
    answer_generator as generator,
    preprocessor,
    question_generator,
    ranker,
    reader,
    retriever,
    summarizer,
    translator,
    label_generator,
)

# Note that we ignore the ImportError here because if the user did not install
# the correct dependency group for a document store, we don't need to setup
# import warnings for that, so the import here is useless and should fail silently.

document_stores: Union[ModuleType, None] = None
try:
    from haystack import document_stores
except ImportError:
    pass

graph_retriever: Union[ModuleType, None] = None
try:
    from haystack.nodes.retriever import text2sparql as graph_retriever
except ImportError:
    pass

knowledge_graph: Union[ModuleType, None] = None
try:
    from haystack.document_stores import graphdb as knowledge_graph
except ImportError:
    pass

from haystack.modeling.evaluation import eval
from haystack.nodes.other import JoinDocuments, Docs2Answers, JoinAnswers, RouteDocuments
from haystack.nodes.query_classifier import SklearnQueryClassifier, TransformersQueryClassifier
from haystack.nodes.file_classifier import FileTypeClassifier
from haystack.utils import preprocessing
import haystack.modeling.utils as modeling_utils
from haystack.utils import cleaning

# For the alias to work as an importable module (like `from haystack import reader`),
# modules need to be set as attributes of their parent model.
# To make chain imports work (`from haystack.reader import FARMReader`) the module
# needs to be also present in sys.modules with its complete import path.
if knowledge_graph:
    setattr(knowledge_graph, "graphdb", DeprecatedModule(knowledge_graph))
    sys.modules["haystack.knowledge_graph.graphdb"] = DeprecatedModule(knowledge_graph)

setattr(preprocessor, "utils", DeprecatedModule(preprocessing))
setattr(preprocessor, "cleaning", DeprecatedModule(cleaning))
sys.modules["haystack.preprocessor.utils"] = DeprecatedModule(preprocessing)
sys.modules["haystack.preprocessor.cleaning"] = DeprecatedModule(cleaning)

setattr(haystack, "document_store", DeprecatedModule(document_stores))
setattr(haystack, "connector", DeprecatedModule(connector))
setattr(haystack, "generator", DeprecatedModule(generator))
setattr(haystack, "document_classifier", DeprecatedModule(document_classifier))
setattr(haystack, "extractor", DeprecatedModule(extractor))
setattr(haystack, "eval", DeprecatedModule(eval))
setattr(haystack, "file_converter", DeprecatedModule(file_converter, deprecated_attributes=["FileTypeClassifier"]))
setattr(haystack, "knowledge_graph", DeprecatedModule(knowledge_graph, deprecated_attributes=["graphdb"]))
setattr(
    haystack,
    "pipeline",
    DeprecatedModule(
        pipelines,
        deprecated_attributes=[
            "JoinDocuments",
            "Docs2Answers",
            "SklearnQueryClassifier",
            "TransformersQueryClassifier",
        ],
    ),
)
setattr(haystack, "preprocessor", DeprecatedModule(preprocessor, deprecated_attributes=["utils", "cleaning"]))
setattr(haystack, "question_generator", DeprecatedModule(question_generator))
setattr(haystack, "ranker", DeprecatedModule(ranker))
setattr(haystack, "reader", DeprecatedModule(reader))
setattr(haystack, "retriever", DeprecatedModule(retriever))
setattr(haystack, "summarizer", DeprecatedModule(summarizer))
setattr(haystack, "translator", DeprecatedModule(translator))
sys.modules["haystack.document_store"] = DeprecatedModule(document_stores)
sys.modules["haystack.connector"] = DeprecatedModule(connector)
sys.modules["haystack.generator"] = DeprecatedModule(generator)
sys.modules["haystack.document_classifier"] = DeprecatedModule(document_classifier)
sys.modules["haystack.extractor"] = DeprecatedModule(extractor)
sys.modules["haystack.eval"] = DeprecatedModule(eval)
sys.modules["haystack.file_converter"] = DeprecatedModule(file_converter)
sys.modules["haystack.knowledge_graph"] = DeprecatedModule(knowledge_graph)
sys.modules["haystack.pipeline"] = DeprecatedModule(pipelines)
sys.modules["haystack.preprocessor"] = DeprecatedModule(preprocessor, deprecated_attributes=["utils", "cleaning"])
sys.modules["haystack.question_generator"] = DeprecatedModule(question_generator)
sys.modules["haystack.ranker"] = DeprecatedModule(ranker)
sys.modules["haystack.reader"] = DeprecatedModule(reader)
sys.modules["haystack.retriever"] = DeprecatedModule(retriever)
sys.modules["haystack.summarizer"] = DeprecatedModule(summarizer)
sys.modules["haystack.translator"] = DeprecatedModule(translator)
if graph_retriever:
    setattr(haystack, "graph_retriever", DeprecatedModule(graph_retriever))
    sys.modules["haystack.graph_retriever"] = DeprecatedModule(graph_retriever)

# To be imported from modules, classes need only to be set as attributes,
# they don't need to be present in sys.modules too.
# Adding them to sys.modules would enable `import haystack.pipelines.JoinDocuments`,
# which I believe it's a very rare import style.
setattr(file_converter, "FileTypeClassifier", FileTypeClassifier)
setattr(pipelines, "JoinDocuments", JoinDocuments)
setattr(pipelines, "Docs2Answers", Docs2Answers)
setattr(pipelines, "SklearnQueryClassifier", SklearnQueryClassifier)
setattr(pipelines, "TransformersQueryClassifier", TransformersQueryClassifier)

# This last line is used to throw the deprecation error for imports like `from haystack import connector`
deprecated_attributes = [
    "document_store",
    "connector",
    "generator",
    "document_classifier",
    "extractor",
    "eval",
    "file_converter",
    "knowledge_graph",
    "pipeline",
    "preprocessor",
    "question_generator",
    "ranker",
    "reader",
    "retriever",
    "summarizer",
    "translator",
]
if graph_retriever:
    deprecated_attributes.append("graph_retriever")
sys.modules["haystack"] = DeprecatedModule(
    haystack, is_module_deprecated=False, deprecated_attributes=deprecated_attributes
)
