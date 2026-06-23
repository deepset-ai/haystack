# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

# We avoid lazy imports here because:
# - they create potential static type checking issues which are hard to debug
# - they make this module more complicated and hard to maintain
# - they offer minimal performance gains in this case.

import haystack.logging

# Imported so the `haystack.tracing` namespace is available after `import haystack`.
import haystack.tracing  # noqa: F401
from haystack.core.component import component
from haystack.core.errors import ComponentError, DeserializationError
from haystack.core.pipeline import Pipeline
from haystack.core.serialization import default_from_dict, default_to_dict
from haystack.core.super_component.super_component import SuperComponent, super_component
from haystack.dataclasses import Answer, Document, ExtractedAnswer, GeneratedAnswer
from haystack.version import __version__  # noqa: F401

# Initialize the logging configuration
# This is a no-op unless `structlog` is installed
haystack.logging.configure_logging()

__all__ = [
    "Answer",
    "ComponentError",
    "DeserializationError",
    "Document",
    "ExtractedAnswer",
    "GeneratedAnswer",
    "Pipeline",
    "SuperComponent",
    "super_component",
    "component",
    "default_from_dict",
    "default_to_dict",
]
