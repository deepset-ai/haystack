from importlib import metadata

from canals import component
from canals.serialization import default_from_dict, default_to_dict
from canals.errors import DeserializationError, ComponentError
from haystack.preview.pipeline import Pipeline
from haystack.preview.dataclasses import Document, Answer, GeneratedAnswer, ExtractedAnswer

# haystack.preview is distributed as a separate package called `haystack-ai`.
# We want to keep all preview dependencies separate from the current Haystack version,
# so imports in haystack.preview must only import from haystack.preview.
# Since we need to access __version__ in haystack.preview without importing from
# haystack we must set it here too.
# When installing `haystack-ai` we want to use that package version though
# as `farm-haystack` might not be installed and cause this to fail.
try:
    __version__: str = str(metadata.version("haystack-ai"))
except metadata.PackageNotFoundError:
    __version__: str = str(metadata.version("farm-haystack"))
