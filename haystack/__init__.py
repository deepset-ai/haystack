import logging

# Change log-levels before modules are loaded to avoid verbose log messages.
logging.getLogger('farm').setLevel(logging.WARNING)
logging.getLogger('farm.utils').setLevel(logging.INFO)
logging.getLogger('farm.infer').setLevel(logging.INFO)
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('farm.eval').setLevel(logging.INFO)
logging.getLogger('farm.modeling.optimization').setLevel(logging.INFO)
logging.getLogger('faiss.loader').setLevel(logging.WARNING)

import pandas as pd
from haystack.schema import Document, Label, MultiLabel, BaseComponent
from haystack.pipeline import Pipeline
from haystack._version import __version__

pd.options.display.max_colwidth = 80

logger = logging.getLogger(__name__)
