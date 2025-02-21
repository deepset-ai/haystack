# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

# These imports need to be loaded eagerly:
# - they configure essential services (logging, tracing)
# - they define core classes which should be accessible through the haystack namespace
import haystack.logging
import haystack.tracing
from haystack.core.component import component
from haystack.version import __version__

_import_structure = {
    "core.errors": ["ComponentError", "DeserializationError"],
    "core.pipeline": ["AsyncPipeline", "Pipeline", "PredefinedPipeline"],
    "core.serialization": ["default_from_dict", "default_to_dict"],
    "dataclasses": ["Answer", "Document", "ExtractedAnswer", "GeneratedAnswer"],
}

if TYPE_CHECKING:
    from .core.errors import ComponentError, DeserializationError
    from .core.pipeline import AsyncPipeline, Pipeline, PredefinedPipeline
    from .core.serialization import default_from_dict, default_to_dict
    from .dataclasses import Answer, Document, ExtractedAnswer, GeneratedAnswer

else:
    sys.modules[__name__] = LazyImporter(
        name=__name__,
        module_file=__file__,
        import_structure=_import_structure,
        # These modules were imported eagerly above,
        # but must also be added to extra_objects so LazyImporter exposes them
        # through the haystack namespace.
        extra_objects={
            "__version__": __version__,
            "logging": haystack.logging,
            "tracing": haystack.tracing,
            "component": component,
            # haystack.core requires special handling:
            # - It has an empty __init__.py to avoid circular imports.
            # - For this reason, it does not play well with LazyImporter.
            # - We pass it directly in extra_objects to preserve the module reference.
            # - This preserves the ability to monkey-patch the module in tests.
            "core": haystack.core,
        },
    )

# Initialize the logging configuration
# This is a no-op unless `structlog` is installed
haystack.logging.configure_logging()

# Same for tracing (no op if `opentelemetry` or `ddtrace` is not installed)
haystack.tracing.auto_enable_tracing()
