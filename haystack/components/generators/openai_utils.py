# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

from haystack.dataclasses import ChatMessage


def _convert_message_to_openai_format(message: ChatMessage) -> Dict[str, str]:
    """
    Convert a message to the format expected by OpenAI's Chat API.

    See the [API reference](https://platform.openai.com/docs/api-reference/chat/create) for details.

    :returns: A dictionary with the following key:
        - `role`
        - `content`
        - `name` (optional)
    """
    openai_msg = {"role": message.role.value, "content": message.content}
    if message.name:
        openai_msg["name"] = message.name

    return openai_msg
