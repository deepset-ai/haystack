# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from haystack.components.extractors.named_entity_extractor import (
    NamedEntityAnnotation,
    NamedEntityExtractor,
    NamedEntityExtractorBackend,
)

__all__ = ["NamedEntityExtractor", "NamedEntityExtractorBackend", "NamedEntityAnnotation"]
