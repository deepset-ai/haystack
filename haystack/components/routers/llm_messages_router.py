# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Dict, List, Optional, Union

from haystack import component
from haystack.components.generators.chat.types import ChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole


@component
class LLMMessagesRouter:
    """
    Routes Chat Messages to different connections, using a generative Language Model to perform classification.

    ### Usage example
    ```python
        from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
        from haystack.components.routers.llm_messages_router import LLMMessagesRouter
        from haystack.dataclasses import ChatMessage

        # initialize a Chat Generator with a generative model for moderation
        chat_generator = HuggingFaceAPIChatGenerator(
            api_type="serverless_inference_api",
            api_params={"model": "meta-llama/Llama-Guard-4-12B", "provider": "groq"},
        )

        router = LLMMessagesRouter(chat_generator=chat_generator,
                                   output_names=["unsafe", "safe"],
                                   output_patterns=["unsafe", "safe"])


        print(router.run([ChatMessage.from_user("How to rob a bank?")]))

        # {
        #     'router_text': 'unsafe\nS2',
        #     'unsafe': [
        #         ChatMessage(
        #             _role=<ChatRole.USER: 'user'>,
        #             _content=[TextContent(text='How to rob a bank?')],
        #             _name=None,
        #             _meta={}
        #         )
        #     ]
        # }
    """

    def __init__(
        self,
        chat_generator: ChatGenerator,
        output_names: List[str],
        output_patterns: List[str],
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the LLMMessagesRouter component.

        :param chat_generator: a ChatGenerator instance which represents the LLM.
        :param output_names: list of names of the output connections of the router. These names can be used to connect
            the router to other components.
        :param output_patterns: list of regular expressions to be matched against the output of the LLM. Each of them
            corresponds to an output name. Matching is executed in the order of the output_patterns list.
        :param system_prompt: system prompt to customize the behavior of the LLM.

        :return: a LLMMessagesRouter instance.

        :raises ValueError: if output_names and output_patterns are not non-empty lists of the same length.
        """
        if not output_names or not output_patterns or len(output_names) != len(output_patterns):
            raise ValueError("output_names and output_patterns must be non-empty lists of the same length")

        self._chat_generator = chat_generator
        self._system_prompt = system_prompt
        self._output_names = output_names
        self._output_patterns = output_patterns

        self._compiled_patterns = [re.compile(pattern) for pattern in output_patterns]
        self._is_warmed_up = False

        component.set_output_types(
            self, **{"router_text": str, **dict.fromkeys(output_names + ["unmatched"], List[ChatMessage])}
        )

    def warm_up(self):
        """
        Warm up the underlying LLM.
        """
        if not self._is_warmed_up:
            if hasattr(self._chat_generator, "warm_up"):
                self._chat_generator.warm_up()
            self._is_warmed_up = True

    def run(self, messages: List[ChatMessage]) -> Dict[str, Union[str, List[ChatMessage]]]:
        """
        Use the LLM to classify the messages and route them to the appropriate output connection.

        :param messages: list of ChatMessages to route. Only user and assistant messages are supported.

        :returns: A dictionary with the following keys:
            - "router_text": the text output of the LLM (for debugging purposes).
            - "unmatched": the messages that did not match any of the output patterns.
            - other keys are the output names, and the values are the messages that matched the corresponding output
            pattern.

        :raises ValueError: if messages is an empty list.
        :raises RuntimeError: if the component is not warmed up and the ChatGenerator has a warm_up method.
        """
        if not messages:
            raise ValueError("messages must be a non-empty list.")
        if not all(message.is_from(ChatRole.USER) or message.is_from(ChatRole.ASSISTANT) for message in messages):
            msg = (
                "messages must contain only user and assistant messages. To customize the behavior of the "
                "chat_generator, you can use the `system_prompt` parameter."
            )
            raise ValueError(msg)

        if not self._is_warmed_up and hasattr(self._chat_generator, "warm_up"):
            raise RuntimeError("The component is not warmed up. Please call the `warm_up` method first.")

        messages_for_inference = []
        if self._system_prompt:
            messages_for_inference.append(ChatMessage.from_system(self._system_prompt))
        messages_for_inference.extend(messages)

        llm_response = self._chat_generator.run(messages=messages_for_inference)["replies"][0].text

        output = {"router_text": llm_response}

        for output_name, pattern in zip(self._output_names, self._compiled_patterns):
            if pattern.search(llm_response):
                output[output_name] = messages
                break
        else:
            output["unmatched"] = messages

        return output
