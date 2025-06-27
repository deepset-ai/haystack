# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.serialization import component_to_dict
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.utils import deserialize_chatgenerator_inplace


@component
class LLMMessagesRouter:
    """
    Routes Chat Messages to different connections using a generative Language Model to perform classification.

    This component can be used with general-purpose LLMs and with specialized LLMs for moderation like Llama Guard.

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
    #     'chat_generator_text': 'unsafe\nS2',
    #     'unsafe': [
    #         ChatMessage(
    #             _role=<ChatRole.USER: 'user'>,
    #             _content=[TextContent(text='How to rob a bank?')],
    #             _name=None,
    #             _meta={}
    #         )
    #     ]
    # }
    ```
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

        :param chat_generator: A ChatGenerator instance which represents the LLM.
        :param output_names: A list of output connection names. These can be used to connect the router to other
            components.
        :param output_patterns: A list of regular expressions to be matched against the output of the LLM. Each pattern
            corresponds to an output name. Patterns are evaluated in order.
            When using moderation models, refer to the model card to understand the expected outputs.
        :param system_prompt: An optional system prompt to customize the behavior of the LLM.
            For moderation models, refer to the model card for supported customization options.

        :raises ValueError: If output_names and output_patterns are not non-empty lists of the same length.
        """
        if not output_names or not output_patterns or len(output_names) != len(output_patterns):
            raise ValueError("`output_names` and `output_patterns` must be non-empty lists of the same length")

        self._chat_generator = chat_generator
        self._system_prompt = system_prompt
        self._output_names = output_names
        self._output_patterns = output_patterns

        self._compiled_patterns = [re.compile(pattern) for pattern in output_patterns]
        self._is_warmed_up = False

        component.set_output_types(
            self, **{"chat_generator_text": str, **dict.fromkeys(output_names + ["unmatched"], List[ChatMessage])}
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
        Classify the messages based on LLM output and route them to the appropriate output connection.

        :param messages: A list of ChatMessages to be routed. Only user and assistant messages are supported.

        :returns: A dictionary with the following keys:
            - "chat_generator_text": The text output of the LLM, useful for debugging.
            - "output_names": Each contains the list of messages that matched the corresponding pattern.
            - "unmatched": The messages that did not match any of the output patterns.

        :raises ValueError: If messages is an empty list or contains messages with unsupported roles.
        :raises RuntimeError: If the component is not warmed up and the ChatGenerator has a warm_up method.
        """
        if not messages:
            raise ValueError("`messages` must be a non-empty list.")
        if not all(message.is_from(ChatRole.USER) or message.is_from(ChatRole.ASSISTANT) for message in messages):
            msg = (
                "`messages` must contain only user and assistant messages. To customize the behavior of the "
                "`chat_generator`, you can use the `system_prompt` parameter."
            )
            raise ValueError(msg)

        if not self._is_warmed_up and hasattr(self._chat_generator, "warm_up"):
            raise RuntimeError("The component is not warmed up. Please call the `warm_up` method first.")

        messages_for_inference = []
        if self._system_prompt:
            messages_for_inference.append(ChatMessage.from_system(self._system_prompt))
        messages_for_inference.extend(messages)

        chat_generator_text = self._chat_generator.run(messages=messages_for_inference)["replies"][0].text

        output = {"chat_generator_text": chat_generator_text}

        for output_name, pattern in zip(self._output_names, self._compiled_patterns):
            if pattern.search(chat_generator_text):
                output[output_name] = messages
                break
        else:
            output["unmatched"] = messages

        return output

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        return default_to_dict(
            self,
            chat_generator=component_to_dict(obj=self._chat_generator, name="chat_generator"),
            output_names=self._output_names,
            output_patterns=self._output_patterns,
            system_prompt=self._system_prompt,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMMessagesRouter":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        if data["init_parameters"].get("chat_generator"):
            deserialize_chatgenerator_inplace(data["init_parameters"], key="chat_generator")

        return default_from_dict(cls, data)
