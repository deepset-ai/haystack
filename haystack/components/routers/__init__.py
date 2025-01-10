# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from haystack.lazy_imports import lazy_dir, lazy_getattr

if TYPE_CHECKING:
    from haystack.components.routers.conditional_router import ConditionalRouter
    from haystack.components.routers.file_type_router import FileTypeRouter
    from haystack.components.routers.metadata_router import MetadataRouter
    from haystack.components.routers.text_language_router import TextLanguageRouter
    from haystack.components.routers.transformers_text_router import TransformersTextRouter
    from haystack.components.routers.zero_shot_text_router import TransformersZeroShotTextRouter


_lazy_imports = {
    "ConditionalRouter": "haystack.components.routers.conditional_router",
    "FileTypeRouter": "haystack.components.routers.file_type_router",
    "MetadataRouter": "haystack.components.routers.metadata_router",
    "TextLanguageRouter": "haystack.components.routers.text_language_router",
    "TransformersTextRouter": "haystack.components.routers.transformers_text_router",
    "TransformersZeroShotTextRouter": "haystack.components.routers.zero_shot_text_router",
}

__all__ = list(_lazy_imports.keys())


def __getattr__(name):
    return lazy_getattr(name, _lazy_imports, __name__)


def __dir__():
    return lazy_dir(_lazy_imports)
