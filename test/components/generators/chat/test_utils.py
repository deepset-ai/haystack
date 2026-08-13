# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.components.generators.chat import MockChatGenerator, OpenAIChatGenerator, OpenAIResponsesChatGenerator
from haystack.components.generators.chat.utils import (
    _HAYSTACK_GENERATION_PARAMETERS,
    _convert_haystack_generation_kwargs,
)


class TestConvertHaystackGenerationKwargs:
    def test_haystack_generation_parameters(self) -> None:
        assert {"max_output_tokens", "temperature", "top_p"} == _HAYSTACK_GENERATION_PARAMETERS

    def test_openai_kwargs(self) -> None:
        converted = _convert_haystack_generation_kwargs(
            OpenAIChatGenerator.__new__(OpenAIChatGenerator),
            {"max_output_tokens": 100, "temperature": 0.2, "top_p": 0.9},
        )
        assert converted == {"max_completion_tokens": 100, "temperature": 0.2, "top_p": 0.9}

    def test_openai_responses_kwargs(self) -> None:
        converted = _convert_haystack_generation_kwargs(
            OpenAIResponsesChatGenerator.__new__(OpenAIResponsesChatGenerator),
            {"max_output_tokens": 100, "temperature": 0.2, "top_p": 0.9},
        )
        assert converted == {"max_output_tokens": 100, "temperature": 0.2, "top_p": 0.9}

    def test_no_mapping(self) -> None:
        assert _convert_haystack_generation_kwargs(MockChatGenerator(), {"max_output_tokens": 100}) == {}

    def test_invalid_parameter(self) -> None:
        with pytest.raises(ValueError, match="Unknown Haystack generation parameter\\(s\\): max_tokens"):
            _convert_haystack_generation_kwargs(MockChatGenerator(), {"max_tokens": 100})
