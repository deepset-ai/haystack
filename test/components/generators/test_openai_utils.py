# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack.dataclasses import ChatMessage
from haystack.components.generators.openai_utils import _convert_message_to_openai_format


def test_convert_message_to_openai_format():
    message = ChatMessage.from_system("You are good assistant")
    assert _convert_message_to_openai_format(message) == {"role": "system", "content": "You are good assistant"}

    message = ChatMessage.from_user("I have a question")
    assert _convert_message_to_openai_format(message) == {"role": "user", "content": "I have a question"}
