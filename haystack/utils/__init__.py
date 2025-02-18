# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    ".utils.device": ["ComponentDevice", "Device", "DeviceMap", "DeviceType"],
    ".utils.auth": ["Secret", "deserialize_secrets_inplace"],
    ".utils.callable_serialization": ["deserialize_callable", "serialize_callable"],
    ".utils.docstore_deserialization": ["deserialize_document_store_in_init_params_inplace"],
    ".utils.expit": ["expit"],
    ".utils.filters": ["document_matches_filter", "raise_on_invalid_filter_syntax"],
    ".utils.type_serialization": ["deserialize_type", "serialize_type"],
    ".utils.jinja2_extensions": ["Jinja2TimeExtension"],
    ".utils.jupyter": ["is_in_jupyter"],
    ".utils.requests_utils": ["request_with_retry"],
    ".utils.utils": ["expand_page_range"],
}

if TYPE_CHECKING:
    from .auth import Secret, deserialize_secrets_inplace
    from .callable_serialization import deserialize_callable, serialize_callable
    from .device import ComponentDevice, Device, DeviceMap, DeviceType
    from .docstore_deserialization import deserialize_document_store_in_init_params_inplace
    from .expit import expit
    from .filters import document_matches_filter, raise_on_invalid_filter_syntax
    from .jinja2_extensions import Jinja2TimeExtension
    from .jupyter import is_in_jupyter
    from .requests_utils import request_with_retry
    from .type_serialization import deserialize_type, serialize_type
    from .utils import expand_page_range
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
