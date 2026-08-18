# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

import pytest

from haystack.components.generators.chat import MockChatGenerator
from haystack.components.generators.chat.utils import (
    _HAYSTACK_GENERATION_PARAMETERS,
    _convert_haystack_generation_kwargs,
)


class MappedMockChatGenerator(MockChatGenerator):
    _HAYSTACK_TO_PROVIDER_GENERATION_KWARGS: ClassVar[dict[str, str]] = {"max_output_tokens": "provider_max_tokens"}


class TestConvertHaystackGenerationKwargs:
    def test_haystack_generation_parameters(self) -> None:
        assert {"max_output_tokens"} == _HAYSTACK_GENERATION_PARAMETERS

    def test_conversion(self) -> None:
        converted = _convert_haystack_generation_kwargs(MappedMockChatGenerator(), {"max_output_tokens": 100})
        assert converted == {"provider_max_tokens": 100}

    def test_no_mapping(self) -> None:
        assert _convert_haystack_generation_kwargs(MockChatGenerator(), {"max_output_tokens": 100}) == {}

    def test_invalid_parameter(self) -> None:
        with pytest.raises(ValueError, match="Unknown Haystack generation parameter\\(s\\): max_tokens"):
            _convert_haystack_generation_kwargs(MockChatGenerator(), {"max_tokens": 100})
