# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal

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
        required_variables: list[str] | Literal["*"] = "*",
        streaming_callback: StreamingCallbackT | None = None,
    ) -> None:
        """
        Initialize the LLM component.

        :param chat_generator: An instance of the chat generator that the LLM should use.
        :param system_prompt: System prompt for the LLM.
        :param user_prompt: User prompt for the LLM. This prompt is appended to the messages provided at
            runtime. If it contains Jinja2 template variables (e.g., `{{ variable_name }}`), they become
            inputs to the component. If omitted or if there are no template variables, `messages` must be
            provided at runtime instead.
        :param required_variables:
            Variables that must be provided as input to user_prompt.
            If a variable listed as required is not provided, an exception is raised.
            If set to `"*"`, all variables found in the prompt are required. Defaults to `"*"`.
            Only relevant when `user_prompt` contains template variables.
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
        :raises ValueError: If user_prompt contains template variables but required_variables is an empty list.
        """
        super(LLM, self).__init__(  # noqa: UP008
            chat_generator=chat_generator,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            required_variables=required_variables,
            streaming_callback=streaming_callback,
        )
        if self._user_chat_prompt_builder is None or len(self._user_chat_prompt_builder.variables) == 0:
            # This means user_prompt is empty or has no template variables.
            # To ensure properly scheduling we then require messages to be passed at runtime.
            component.set_input_type(self, "messages", list[ChatMessage])
        else:
            # user prompt was provided with variables
            if isinstance(required_variables, list) and len(required_variables) == 0:
                raise ValueError(
                    "required_variables must not be empty. Set it to '*' to require all variables, "
                    "or provide a non-empty list of variable names."
                )
            component.set_input_type(self, "messages", list[ChatMessage], None)

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

    def run(  # type: ignore[override]  # `messages` is in **kwargs to allow dynamic required/optional status
        self,
        *,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Process messages and generate a response from the language model.

        :param messages: Optional list of ChatMessage objects to prepend to the conversation. Whether this is
            required or optional depends on the `user_prompt` configuration: if `user_prompt` has no template
            variables, `messages` must be provided. Passed via `**kwargs`.
        :param streaming_callback: A callback that will be invoked when a response is streamed from the LLM.
        :param generation_kwargs: Additional keyword arguments for the underlying chat generator. These parameters
            will override the parameters passed during component initialization.
        :param system_prompt: System prompt for the LLM. If provided, it overrides the default system prompt.
        :param user_prompt: User prompt for the LLM. If provided, it overrides the default user prompt and is
            appended to the messages provided at runtime.
        :param kwargs: Additional keyword arguments. These are used to fill template variables in the `user_prompt`
            (the keys must match template variable names).
        :returns:
            A dictionary with the following keys:
            - "messages": List of all messages exchanged during the LLM's run.
            - "last_message": The last message exchanged during the LLM's run.
        """
        # `messages` is intentionally omitted from the signature so the framework can treat it as required
        # or optional depending on init configuration. See __init__ for details.
        messages = kwargs.pop("messages", None)
        return super(LLM, self).run(  # noqa: UP008
            messages=messages or [],
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs,
        )

    async def run_async(  # type: ignore[override]  # `messages` is in **kwargs to allow dynamic required/optional status
        self,
        *,
        streaming_callback: StreamingCallbackT | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Asynchronously process messages and generate a response from the language model.

        :param messages: Optional list of ChatMessage objects to prepend to the conversation. Whether this is
            required or optional depends on the `user_prompt` configuration: if `user_prompt` has no template
            variables, `messages` must be provided. Passed via `**kwargs`.
        :param streaming_callback: An asynchronous callback that will be invoked when a response is streamed
            from the LLM.
        :param generation_kwargs: Additional keyword arguments for the underlying chat generator. These parameters
            will override the parameters passed during component initialization.
        :param system_prompt: System prompt for the LLM. If provided, it overrides the default system prompt.
        :param user_prompt: User prompt for the LLM. If provided, it overrides the default user prompt and is
            appended to the messages provided at runtime.
        :param kwargs: Additional keyword arguments. These are used to fill template variables in the `user_prompt`
            (the keys must match template variable names).
        :returns:
            A dictionary with the following keys:
            - "messages": List of all messages exchanged during the LLM's run.
            - "last_message": The last message exchanged during the LLM's run.
        """
        # `messages` is intentionally omitted from the signature so the framework can treat it as required
        # or optional depending on init configuration. See __init__ for details.
        messages = kwargs.pop("messages", None)
        return await super(LLM, self).run_async(  # noqa: UP008
            messages=messages or [],
            streaming_callback=streaming_callback,
            generation_kwargs=generation_kwargs,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs,
        )
