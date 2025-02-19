# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

import haystack.logging
import haystack.tracing
from haystack.version import __version__

_import_structure = {
    "core.component": ["component"],
    "core.errors": ["ComponentError", "DeserializationError"],
    "core.pipeline": ["AsyncPipeline", "Pipeline", "PredefinedPipeline"],
    "core.serialization": ["default_from_dict", "default_to_dict"],
    "dataclasses": ["Answer", "Document", "ExtractedAnswer", "GeneratedAnswer"],
}

if TYPE_CHECKING:
    from .core.component import component
    from .core.errors import ComponentError, DeserializationError
    from .core.pipeline import AsyncPipeline, Pipeline, PredefinedPipeline
    from .core.serialization import default_from_dict, default_to_dict
    from .dataclasses import Answer, Document, ExtractedAnswer, GeneratedAnswer

else:
    sys.modules[__name__] = LazyImporter(
        name=__name__,
        module_file=__file__,
        import_structure=_import_structure,
        # we need to pass the some objects as extra objects since we do not want to import them lazily
        extra_objects={"__version__": __version__, "logging": haystack.logging, "tracing": haystack.tracing},
    )

# Initialize the logging configuration
# This is a no-op unless `structlog` is installed
haystack.logging.configure_logging()

# Same for tracing (no op if `opentelemetry` or `ddtrace` is not installed)
haystack.tracing.auto_enable_tracing()
