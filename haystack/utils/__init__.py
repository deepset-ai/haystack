# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import TYPE_CHECKING

from haystack.lazy_imports import lazy_dir, lazy_getattr

from .expit import expit

if TYPE_CHECKING:
    from .auth import Secret, deserialize_secrets_inplace
    from .callable_serialization import deserialize_callable, serialize_callable
    from .device import ComponentDevice, Device, DeviceMap, DeviceType
    from .docstore_deserialization import deserialize_document_store_in_init_params_inplace
    from .filters import document_matches_filter, raise_on_invalid_filter_syntax
    from .jinja2_extensions import Jinja2TimeExtension
    from .jupyter import is_in_jupyter
    from .requests_utils import request_with_retry
    from .type_serialization import deserialize_type, serialize_type

_lazy_imports = {
    "Secret": "haystack.utils.auth",
    "deserialize_secrets_inplace": "haystack.utils.auth",
    "deserialize_callable": "haystack.utils.callable_serialization",
    "serialize_callable": "haystack.utils.callable_serialization",
    "ComponentDevice": "haystack.utils.device",
    "Device": "haystack.utils.device",
    "DeviceMap": "haystack.utils.device",
    "DeviceType": "haystack.utils.device",
    "deserialize_document_store_in_init_params_inplace": "haystack.utils.docstore_deserialization",
    "document_matches_filter": "haystack.utils.filters",
    "raise_on_invalid_filter_syntax": "haystack.utils.filters",
    "Jinja2TimeExtension": "haystack.utils.jinja2_extensions",
    "is_in_jupyter": "haystack.utils.jupyter",
    "request_with_retry": "haystack.utils.requests_utils",
    "deserialize_type": "haystack.utils.type_serialization",
    "serialize_type": "haystack.utils.type_serialization",
}

__all__ = list(_lazy_imports.keys())


def __getattr__(name):
    return lazy_getattr(name, _lazy_imports, __name__)


def __dir__():
    return lazy_dir(_lazy_imports)
