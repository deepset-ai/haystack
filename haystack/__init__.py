# pylint: disable=wrong-import-position,wrong-import-order

from typing import Union
from types import ModuleType

try:
    from importlib import metadata
except (ModuleNotFoundError, ImportError):
    # Python <= 3.7
    import importlib_metadata as metadata  # type: ignore

__version__: str = str(metadata.version("farm-haystack"))


# Logging is not configured here on purpose, see https://github.com/deepset-ai/haystack/issues/2485
import logging

import pandas as pd

from haystack.schema import Document, Answer, Label, MultiLabel, Span, EvaluationResult
from haystack.nodes.base import BaseComponent
from haystack.pipelines.base import Pipeline


pd.options.display.max_colwidth = 80
