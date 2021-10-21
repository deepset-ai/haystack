import logging
from haystack import pipelines

# Configure the root logger t0 DEBUG to allow the "debug" flag to receive the logs
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# Then reconfigure the StreamHandler not to display anything below WARNING as default
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
root_logger.addHandler(stream_handler)

# Change log-levels before modules are loaded to avoid verbose log messages.
logging.getLogger('haystack.modeling').setLevel(logging.WARNING)
logging.getLogger('haystack.modeling.utils').setLevel(logging.INFO)
logging.getLogger('haystack.modeling.infer').setLevel(logging.INFO)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('haystack.modeling.evaluation.eval').setLevel(logging.INFO)
logging.getLogger('haystack.modeling.model.optimization').setLevel(logging.INFO)
logging.getLogger('faiss.loader').setLevel(logging.WARNING)

from haystack.schema import Document, Answer, Label, MultiLabel, Span
from haystack.nodes import BaseComponent, connector
from haystack.pipelines import Pipeline
from haystack._version import __version__

import pandas as pd
pd.options.display.max_colwidth = 80

logger = logging.getLogger(__name__)


# Old style imports
import sys

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
    translator
)
from haystack.nodes.retriever import text2sparql as graph_retriever
from haystack.document_store import (
    graphdb as knowledge_graph
)
from haystack.modeling.evaluation import eval
from haystack.utils import preprocessing 
from haystack.utils import cleaning 
# query_classifier ?

sys.modules["haystack.connector"] = connector
sys.modules["haystack.document_classifier"] = document_classifier
sys.modules["haystack.extractor"] = extractor
sys.modules["haystack.file_converter"] = file_converter
sys.modules["haystack.preprocessor"] = preprocessor
sys.modules["haystack.question_generator"] = question_generator
sys.modules["haystack.ranker"] = ranker
sys.modules["haystack.reader"] = reader
sys.modules["haystack.retriever"] = retriever
sys.modules["haystack.summarizer"] = summarizer
sys.modules["haystack.translator"] = translator
sys.modules["haystack.graph_retriever"] = graph_retriever
sys.modules["haystack.knowledge_graph"] = knowledge_graph
sys.modules["haystack.knowledge_graph.graphdb"] = knowledge_graph
sys.modules["haystack.eval"] = eval
sys.modules["haystack.pipeline"] = pipelines
sys.modules["haystack.preprocessor.utils"] = preprocessing
sys.modules["haystack.preprocessor.cleaning"] = cleaning
