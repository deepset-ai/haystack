# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from haystack.lazy_imports import lazy_dir, lazy_getattr

if TYPE_CHECKING:
    from haystack.components.generators.openai import (  # noqa: I001 (otherwise we end up with partial imports)
        OpenAIGenerator,
    )
    from haystack.components.generators.azure import AzureOpenAIGenerator
    from haystack.components.generators.hugging_face_local import HuggingFaceLocalGenerator
    from haystack.components.generators.hugging_face_api import HuggingFaceAPIGenerator
    from haystack.components.generators.openai_dalle import DALLEImageGenerator


_lazy_imports = {
    "OpenAIGenerator": "haystack.components.generators.openai",
    "AzureOpenAIGenerator": "haystack.components.generators.azure",
    "HuggingFaceLocalGenerator": "haystack.components.generators.hugging_face_local",
    "HuggingFaceAPIGenerator": "haystack.components.generators.hugging_face_api",
    "DALLEImageGenerator": "haystack.components.generators.openai_dalle",
}

__all__ = list(_lazy_imports.keys())


def __getattr__(name):
    return lazy_getattr(name, _lazy_imports, __name__)


def __dir__():
    return lazy_dir(_lazy_imports)
