# pylint: disable=wrong-import-position
# Logging is not configured here on purpose, see https://github.com/deepset-ai/haystack/issues/2485

from importlib import metadata

__version__: str = str(metadata.version("farm-haystack"))


import haystack.silenceable_tqdm  # Needs to be imported first to wrap TQDM for all following modules
from haystack.schema import Document, Answer, Label, MultiLabel, Span, EvaluationResult, TableCell
from haystack.nodes.base import BaseComponent
from haystack.pipelines.base import Pipeline
from haystack.environment import set_pytorch_secure_model_loading
from haystack.mmh3 import hash128


# Enables torch's secure model loading through setting an env var.
# Does not use torch.
set_pytorch_secure_model_loading()
