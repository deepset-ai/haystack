# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable
from haystack.utils.http_client import init_http_client

logger = logging.getLogger(__name__)


@component
class OpenAIGenerator:
    """
    Generates text using OpenAI's large language models (LLMs).

    It works with the gpt-4 and o-series models and supports streaming responses
    from OpenAI API. It uses strings as input and output.

    You can customize how the text is generated by passing parameters to the
    OpenAI API. Use the `**generation_kwargs` argument when you initialize
    the component or when you run it. Any parameter that works with
    `openai.ChatCompletion.create` will work here too.


    For details on OpenAI API parameters, see
    [OpenAI documentation](https://platform.openai.com/docs/api-reference/chat).

    ### Usage example

    ```python
    from haystack.components.generators import OpenAIGenerator
    client = OpenAIGenerator()
    response = client.run("What's Natural Language Processing? Be brief.")
    print(response)

    >> {'replies': ['Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on
    >> the interaction between computers and human language. It involves enabling computers to understand, interpret,
    >> and respond to natural human language in a way that is both meaningful and useful.'], 'meta': [{'model':
    >> 'gpt-4o-mini', 'index': 0, 'finish_reason': 'stop', 'usage': {'prompt_tokens': 16,
    >> 'completion_tokens': 49, 'total_tokens': 65}}]}
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        api_key: Secret = Secret.from_env_var("OPENAI_API_KEY"),
        model: str = "gpt-4o-mini",
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        system_prompt: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        http_client_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of OpenAIGenerator. Unless specified otherwise in `model`, uses OpenAI's gpt-4o-mini

        By setting the 'OPENAI_TIMEOUT' and 'OPENAI_MAX_RETRIES' you can change the timeout and max_retries parameters
        in the OpenAI client.

        :param api_key: The OpenAI API key to connect to OpenAI.
        :param model: The name of the model to use.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url: An optional base URL.
        :param organization: The Organization ID, defaults to `None`.
        :param system_prompt: The system prompt to use for text generation. If not provided, the system prompt is
        omitted, and the default system prompt of the model is used.
        :param generation_kwargs: Other parameters to use for the model. These parameters are all sent directly to
            the OpenAI endpoint. See OpenAI [documentation](https://platform.openai.com/docs/api-reference/chat) for
            more details.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So, 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `n`: How many completions to generate for each prompt. For example, if the LLM gets 3 prompts and n is 2,
                it will generate two completions for each of the three prompts, ending up with 6 completions in total.
            - `stop`: One or more sequences after which the LLM should stop generating tokens.
            - `presence_penalty`: What penalty to apply if a token is already present at all. Bigger values mean
                the model will be less likely to repeat the same token in the text.
            - `frequency_penalty`: What penalty to apply if a token has already been generated in the text.
                Bigger values mean the model will be less likely to repeat the same token in the text.
            - `logit_bias`: Add a logit bias to specific tokens. The keys of the dictionary are tokens, and the
                values are the bias to add to that token.
        :param timeout:
            Timeout for OpenAI Client calls, if not set it is inferred from the `OPENAI_TIMEOUT` environment variable
            or set to 30.
        :param max_retries:
            Maximum retries to establish contact with OpenAI if it returns an internal error, if not set it is inferred
            from the `OPENAI_MAX_RETRIES` environment variable or set to 5.
        :param http_client_kwargs:
            A dictionary of keyword arguments to configure a custom `httpx.Client`or `httpx.AsyncClient`.
            For more information, see the [HTTPX documentation](https://www.python-httpx.org/api/#client).
        """
        self.api_key = api_key
        self.model = model
        self.generation_kwargs = generation_kwargs or {}
        self.system_prompt = system_prompt
        self.streaming_callback = streaming_callback

        self.api_base_url = api_base_url
        self.organization = organization
        self.http_client_kwargs = http_client_kwargs

        if timeout is None:
            timeout = float(os.environ.get("OPENAI_TIMEOUT", "30.0"))
        if max_retries is None:
            max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))

        self.client = OpenAI(
            api_key=api_key.resolve_value(),
            organization=organization,
            base_url=api_base_url,
            timeout=timeout,
            max_retries=max_retries,
            http_client=init_http_client(self.http_client_kwargs, async_client=False),
        )

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            The serialized component as a dictionary.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            organization=self.organization,
            generation_kwargs=self.generation_kwargs,
            system_prompt=self.system_prompt,
            api_key=self.api_key.to_dict(),
            http_client_kwargs=self.http_client_kwargs,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAIGenerator":
        """
        Deserialize this component from a dictionary.

        :param data:
            The dictionary representation of this component.
        :returns:
            The deserialized component instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_key"])
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param prompt:
            The string prompt to use for text generation.
        :param system_prompt:
            The system prompt to use for text generation. If this run time system prompt is omitted, the system
            prompt, if defined at initialisation time, is used.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param generation_kwargs:
            Additional keyword arguments for text generation. These parameters will potentially override the parameters
            passed in the `__init__` method. For more details on the parameters supported by the OpenAI API, refer to
            the OpenAI [documentation](https://platform.openai.com/docs/api-reference/chat/create).
        :returns:
            A list of strings containing the generated responses and a list of dictionaries containing the metadata
        for each response.
        """
        message = ChatMessage.from_user(prompt)
        if system_prompt is not None:
            messages = [ChatMessage.from_system(system_prompt), message]
        elif self.system_prompt:
            messages = [ChatMessage.from_system(self.system_prompt), message]
        else:
            messages = [message]

        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        # check if streaming_callback is passed
        streaming_callback = streaming_callback or self.streaming_callback

        # adapt ChatMessage(s) to the format expected by the OpenAI API
        openai_formatted_messages = [message.to_openai_dict_format() for message in messages]

        completion: Union[Stream[ChatCompletionChunk], ChatCompletion] = self.client.chat.completions.create(
            model=self.model,
            messages=openai_formatted_messages,  # type: ignore
            stream=streaming_callback is not None,
            **generation_kwargs,
        )

        completions: List[ChatMessage] = []
        if isinstance(completion, Stream):
            num_responses = generation_kwargs.pop("n", 1)
            if num_responses > 1:
                raise ValueError("Cannot stream multiple responses, please set n=1.")
            chunks: List[StreamingChunk] = []
            completion_chunk: Optional[ChatCompletionChunk] = None

            # pylint: disable=not-an-iterable
            for completion_chunk in completion:
                if completion_chunk.choices and streaming_callback:
                    chunk_delta: StreamingChunk = self._build_chunk(completion_chunk)
                    chunks.append(chunk_delta)
                    streaming_callback(chunk_delta)  # invoke callback with the chunk_delta
            # Makes type checkers happy
            assert completion_chunk is not None
            completions = [self._create_message_from_chunks(completion_chunk, chunks)]
        elif isinstance(completion, ChatCompletion):
            completions = [self._build_message(completion, choice) for choice in completion.choices]

        # before returning, do post-processing of the completions
        for response in completions:
            self._check_finish_reason(response)

        return {"replies": [message.text for message in completions], "meta": [message.meta for message in completions]}

    @staticmethod
    def _create_message_from_chunks(
        completion_chunk: ChatCompletionChunk, streamed_chunks: List[StreamingChunk]
    ) -> ChatMessage:
        """
        Creates a single ChatMessage from the streamed chunks. Some data is retrieved from the completion chunk.
        """
        complete_response = ChatMessage.from_assistant("".join([chunk.content for chunk in streamed_chunks]))
        finish_reason = streamed_chunks[-1].meta["finish_reason"]
        complete_response.meta.update(
            {
                "model": completion_chunk.model,
                "index": 0,
                "finish_reason": finish_reason,
                "completion_start_time": streamed_chunks[0].meta.get("received_at"),  # first chunk received
                "usage": dict(completion_chunk.usage or {}),
            }
        )
        return complete_response

    @staticmethod
    def _build_message(completion: Any, choice: Any) -> ChatMessage:
        """
        Converts the response from the OpenAI API to a ChatMessage.

        :param completion:
            The completion returned by the OpenAI API.
        :param choice:
            The choice returned by the OpenAI API.
        :returns:
            The ChatMessage.
        """
        # function or tools calls are not going to happen in non-chat generation
        # as users can not send ChatMessage with function or tools calls
        chat_message = ChatMessage.from_assistant(choice.message.content or "")
        chat_message.meta.update(
            {
                "model": completion.model,
                "index": choice.index,
                "finish_reason": choice.finish_reason,
                "usage": dict(completion.usage),
            }
        )
        return chat_message

    @staticmethod
    def _build_chunk(chunk: Any) -> StreamingChunk:
        """
        Converts the response from the OpenAI API to a StreamingChunk.

        :param chunk:
            The chunk returned by the OpenAI API.
        :returns:
            The StreamingChunk.
        """
        choice = chunk.choices[0]
        content = choice.delta.content or ""
        chunk_message = StreamingChunk(content)
        chunk_message.meta.update(
            {
                "model": chunk.model,
                "index": choice.index,
                "finish_reason": choice.finish_reason,
                "received_at": datetime.now().isoformat(),
            }
        )
        return chunk_message

    @staticmethod
    def _check_finish_reason(message: ChatMessage) -> None:
        """
        Check the `finish_reason` returned with the OpenAI completions.

        If the `finish_reason` is `length`, log a warning to the user.

        :param message:
            The message returned by the LLM.
        """
        if message.meta["finish_reason"] == "length":
            logger.warning(
                "The completion for index {index} has been truncated before reaching a natural stopping point. "
                "Increase the max_tokens parameter to allow for longer completions.",
                index=message.meta["index"],
                finish_reason=message.meta["finish_reason"],
            )
        if message.meta["finish_reason"] == "content_filter":
            logger.warning(
                "The completion for index {index} has been truncated due to the content filter.",
                index=message.meta["index"],
                finish_reason=message.meta["finish_reason"],
            )
