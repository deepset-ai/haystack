# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, override

from haystack import component, logging
from haystack.components.agents.agent import Agent
from haystack.components.generators.chat.types import ChatGenerator
from haystack.core.serialization import component_to_dict, default_from_dict, default_to_dict
from haystack.dataclasses import ChatMessage, StreamingCallbackT
from haystack.utils.callable_serialization import deserialize_callable, serialize_callable
from haystack.utils.deserialization import deserialize_component_inplace

logger = logging.getLogger(__name__)


@component
class LLM(Agent):
    """
    A text generation component powered by a large language model.

    The LLM component is a simplified version of the Agent that focuses solely on text generation
    without tool usage. It processes messages and returns a single response from the language model.

    ### Usage examples

    ```python
    from haystack.components.generators.chat import LLM
    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.dataclasses import ChatMessage

    llm = LLM(
        chat_generator=OpenAIChatGenerator(),
        system_prompt="You are a helpful translation assistant.",
        user_prompt=\"\"\"{% message role="user"%}
    Summarize the following document: {{ document }}
    {% endmessage %}\"\"\",
        required_variables=["document"],
    )

    result = llm.run(document="The weather is lovely today and the sun is shining. ")
    print(result["last_message"].text)
    ```
    """

    def __init__(
        self,
        *,
        chat_generator: ChatGenerator,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        required_variables: list[str] | Literal["*"] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
    ) -> None:
        """
        Initialize the LLM component.

        :param chat_generator: An instance of the chat generator that the LLM should use.
        :param system_prompt: System prompt for the LLM.
        :param user_prompt: User prompt for the LLM. If provided this is appended to the messages provided at runtime.
        :param required_variables:
            List variables that must be provided as input to user_prompt.
            If a variable listed as required is not provided, an exception is raised.
            If set to `"*"`, all variables found in the prompt are required. Optional.
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
        """
        super().__init__(
            chat_generator=chat_generator,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            required_variables=required_variables,
            streaming_callback=streaming_callback,
        )

    @override
    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the LLM component to a dictionary.

        :return: Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            chat_generator=component_to_dict(obj=self.chat_generator, name="chat_generator"),
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
            required_variables=self.required_variables,
            streaming_callback=serialize_callable(self.streaming_callback) if self.streaming_callback else None,
        )

    @classmethod
    @override
    def from_dict(cls, data: dict[str, Any]) -> "LLM":
        """
        Deserialize the LLM from a dictionary.

        :param data: Dictionary to deserialize from.
        :return: Deserialized LLM instance.
        """
        init_params = data.get("init_parameters", {})

        deserialize_component_inplace(init_params, key="chat_generator")

        if init_params.get("streaming_callback") is not None:
            init_params["streaming_callback"] = deserialize_callable(init_params["streaming_callback"])

        return default_from_dict(cls, data)

    @override
    def run(
        self,
        messages: list[ChatMessage] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        *,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Process messages and generate a response from the language model.

        :param messages: List of Haystack ChatMessage objects to process.
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
        :param system_prompt: System prompt for the LLM. If provided, it overrides the default system prompt.
        :param user_prompt: User prompt for the LLM. If provided, it overrides the default user prompt and is
            appended to the messages provided at runtime.
        :param kwargs: Additional keyword arguments. These are used to fill template variables in the `user_prompt`
            (the keys must match template variable names). You can also pass `generation_kwargs` to override
            generation parameters for the underlying chat generator.
        :returns:
            A dictionary with the following keys:
            - "messages": List of all messages exchanged during the LLM's run.
            - "last_message": The last message exchanged during the LLM's run.
        """
        return super().run(
            messages=messages,
            streaming_callback=streaming_callback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs,
        )

    @override
    async def run_async(
        self,
        messages: list[ChatMessage] | None = None,
        streaming_callback: StreamingCallbackT | None = None,
        *,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Asynchronously process messages and generate a response from the language model.

        :param messages: List of Haystack ChatMessage objects to process.
        :param streaming_callback: An asynchronous callback that will be invoked when a response is streamed
            from the LLM.
        :param system_prompt: System prompt for the LLM. If provided, it overrides the default system prompt.
        :param user_prompt: User prompt for the LLM. If provided, it overrides the default user prompt and is
            appended to the messages provided at runtime.
        :param kwargs: Additional keyword arguments. These are used to fill template variables in the `user_prompt`
            (the keys must match template variable names). You can also pass `generation_kwargs` to override
            generation parameters for the underlying chat generator.
        :returns:
            A dictionary with the following keys:
            - "messages": List of all messages exchanged during the LLM's run.
            - "last_message": The last message exchanged during the LLM's run.
        """
        return await super().run_async(
            messages=messages,
            streaming_callback=streaming_callback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs,
        )
