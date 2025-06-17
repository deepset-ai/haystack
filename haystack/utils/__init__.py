# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import sys
from typing import TYPE_CHECKING

from lazy_imports import LazyImporter

_import_structure = {
    "auth": ["Secret", "deserialize_secrets_inplace"],
    "azure": ["default_azure_ad_token_provider"],
    "base_serialization": ["_deserialize_value_with_schema", "_serialize_value_with_schema"],
    "callable_serialization": ["deserialize_callable", "serialize_callable"],
    "device": ["ComponentDevice", "Device", "DeviceMap", "DeviceType"],
    "deserialization": ["deserialize_document_store_in_init_params_inplace", "deserialize_chatgenerator_inplace"],
    "filters": ["document_matches_filter", "raise_on_invalid_filter_syntax"],
    "jinja2_extensions": ["Jinja2TimeExtension"],
    "jupyter": ["is_in_jupyter"],
    "misc": ["expit", "expand_page_range"],
    "requests_utils": ["request_with_retry", "async_request_with_retry"],
    "type_serialization": ["deserialize_type", "serialize_type"],
}

if TYPE_CHECKING:
    from .auth import Secret as Secret
    from .auth import deserialize_secrets_inplace as deserialize_secrets_inplace
    from .azure import default_azure_ad_token_provider as default_azure_ad_token_provider
    from .base_serialization import _deserialize_value_with_schema as _deserialize_value_with_schema
    from .base_serialization import _serialize_value_with_schema as _serialize_value_with_schema
    from .callable_serialization import deserialize_callable as deserialize_callable
    from .callable_serialization import serialize_callable as serialize_callable
    from .deserialization import deserialize_chatgenerator_inplace as deserialize_chatgenerator_inplace
    from .deserialization import (
        deserialize_document_store_in_init_params_inplace as deserialize_document_store_in_init_params_inplace,
    )
    from .device import ComponentDevice as ComponentDevice
    from .device import Device as Device
    from .device import DeviceMap as DeviceMap
    from .device import DeviceType as DeviceType
    from .filters import document_matches_filter as document_matches_filter
    from .filters import raise_on_invalid_filter_syntax as raise_on_invalid_filter_syntax
    from .jinja2_extensions import Jinja2TimeExtension as Jinja2TimeExtension
    from .jupyter import is_in_jupyter as is_in_jupyter
    from .misc import expand_page_range as expand_page_range
    from .misc import expit as expit
    from .requests_utils import async_request_with_retry as async_request_with_retry
    from .requests_utils import request_with_retry as request_with_retry
    from .type_serialization import deserialize_type as deserialize_type
    from .type_serialization import serialize_type as serialize_type
else:
    sys.modules[__name__] = LazyImporter(name=__name__, module_file=__file__, import_structure=_import_structure)
