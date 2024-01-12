import dataclasses
import logging
import warnings
from typing import Optional, List, Callable, Dict, Any, Union

from openai import OpenAI, Stream
from openai.types.chat import ChatCompletionChunk, ChatCompletion

from haystack import component, default_from_dict, default_to_dict
from haystack.components.generators.utils import serialize_callback_handler, deserialize_callback_handler
from haystack.dataclasses import StreamingChunk, ChatMessage

logger = logging.getLogger(__name__)


@component
class OpenAIGenerator:
    """
    Enables text generation using OpenAI's large language models (LLMs). It supports gpt-4 and gpt-3.5-turbo
    family of models.

    Users can pass any text generation parameters valid for the `openai.ChatCompletion.create` method
    directly to this component via the `**generation_kwargs` parameter in __init__ or the `**generation_kwargs`
    parameter in `run` method.

    For more details on the parameters supported by the OpenAI API, refer to the OpenAI
    [documentation](https://platform.openai.com/docs/api-reference/chat).

    ```python
    from haystack.components.generators import OpenAIGenerator
    client = OpenAIGenerator()
    response = client.run("What's Natural Language Processing? Be brief.")
    print(response)

    >> {'replies': ['Natural Language Processing (NLP) is a branch of artificial intelligence that focuses on
    >> the interaction between computers and human language. It involves enabling computers to understand, interpret,
    >> and respond to natural human language in a way that is both meaningful and useful.'], 'meta': [{'model':
    >> 'gpt-3.5-turbo-0613', 'index': 0, 'finish_reason': 'stop', 'usage': {'prompt_tokens': 16,
    >> 'completion_tokens': 49, 'total_tokens': 65}}]}
    ```

     Key Features and Compatibility:
         - **Primary Compatibility**: Designed to work seamlessly with gpt-4, gpt-3.5-turbo family of models.
         - **Streaming Support**: Supports streaming responses from the OpenAI API.
         - **Customizability**: Supports all parameters supported by the OpenAI API.

     Input and Output Format:
         - **String Format**: This component uses the strings for both input and output.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        system_prompt: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Creates an instance of OpenAIGenerator. Unless specified otherwise in the `model`, this is for OpenAI's
        GPT-3.5 model.

        :param api_key: The OpenAI API key. It can be explicitly provided or automatically read from the
            environment variable OPENAI_API_KEY (recommended).
        :param model: The name of the model to use.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function accepts StreamingChunk as an argument.
        :param api_base_url: An optional base URL.
        :param organization: The Organization ID, defaults to `None`. See
        [production best practices](https://platform.openai.com/docs/guides/production-best-practices/setting-up-your-organization).
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
        """
        self.model = model
        self.generation_kwargs = generation_kwargs or {}
        self.system_prompt = system_prompt
        self.streaming_callback = streaming_callback

        self.api_base_url = api_base_url
        self.organization = organization
        self.client = OpenAI(api_key=api_key, organization=organization, base_url=api_base_url)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        :return: The serialized component as a dictionary.
        """
        callback_name = serialize_callback_handler(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.model,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            generation_kwargs=self.generation_kwargs,
            system_prompt=self.system_prompt,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenAIGenerator":
        """
        Deserialize this component from a dictionary.
        :param data: The dictionary representation of this component.
        :return: The deserialized component instance.
        """
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callback_handler(serialized_callback_handler)
        return default_from_dict(cls, data)

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param prompt: The string prompt to use for text generation.
        :param generation_kwargs: Additional keyword arguments for text generation. These parameters will
        potentially override the parameters passed in the __init__ method.
        For more details on the parameters supported by the OpenAI API, refer to the
        OpenAI [documentation](https://platform.openai.com/docs/api-reference/chat/create).
        :return: A list of strings containing the generated responses and a list of dictionaries containing the metadata
        for each response.
        """
        message = ChatMessage.from_user(prompt)
        if self.system_prompt:
            messages = [ChatMessage.from_system(self.system_prompt), message]
        else:
            messages = [message]

        # update generation kwargs by merging with the generation kwargs passed to the run method
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        # adapt ChatMessage(s) to the format expected by the OpenAI API
        openai_formatted_messages = self._convert_to_openai_format(messages)

        completion: Union[Stream[ChatCompletionChunk], ChatCompletion] = self.client.chat.completions.create(
            model=self.model,
            messages=openai_formatted_messages,  # type: ignore
            stream=self.streaming_callback is not None,
            **generation_kwargs,
        )

        completions: List[ChatMessage] = []
        if isinstance(completion, Stream):
            num_responses = generation_kwargs.pop("n", 1)
            if num_responses > 1:
                raise ValueError("Cannot stream multiple responses, please set n=1.")
            chunks: List[StreamingChunk] = []
            chunk = None

            # pylint: disable=not-an-iterable
            for chunk in completion:
                if chunk.choices and self.streaming_callback:
                    chunk_delta: StreamingChunk = self._build_chunk(chunk)
                    chunks.append(chunk_delta)
                    self.streaming_callback(chunk_delta)  # invoke callback with the chunk_delta
            completions = [self._connect_chunks(chunk, chunks)]
        elif isinstance(completion, ChatCompletion):
            completions = [self._build_message(completion, choice) for choice in completion.choices]

        # before returning, do post-processing of the completions
        for response in completions:
            self._check_finish_reason(response)

        return {
            "replies": [message.content for message in completions],
            "meta": [message.meta for message in completions],
        }

    def _convert_to_openai_format(self, messages: List[ChatMessage]) -> List[Dict[str, Any]]:
        """
        Converts the list of ChatMessage to the list of messages in the format expected by the OpenAI API.
        :param messages: The list of ChatMessage.
        :return: The list of messages in the format expected by the OpenAI API.
        """
        openai_chat_message_format = {"role", "content", "name"}
        openai_formatted_messages = []
        for m in messages:
            message_dict = dataclasses.asdict(m)
            filtered_message = {k: v for k, v in message_dict.items() if k in openai_chat_message_format and v}
            openai_formatted_messages.append(filtered_message)
        return openai_formatted_messages

    def _connect_chunks(self, chunk: Any, chunks: List[StreamingChunk]) -> ChatMessage:
        """
        Connects the streaming chunks into a single ChatMessage.
        """
        complete_response = ChatMessage.from_assistant("".join([chunk.content for chunk in chunks]))
        complete_response.meta.update(
            {
                "model": chunk.model,
                "index": 0,
                "finish_reason": chunk.choices[0].finish_reason,
                "usage": {},  # we don't have usage data for streaming responses
            }
        )
        return complete_response

    def _build_message(self, completion: Any, choice: Any) -> ChatMessage:
        """
        Converts the response from the OpenAI API to a ChatMessage.
        :param completion: The completion returned by the OpenAI API.
        :param choice: The choice returned by the OpenAI API.
        :return: The ChatMessage.
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

    def _build_chunk(self, chunk: Any) -> StreamingChunk:
        """
        Converts the response from the OpenAI API to a StreamingChunk.
        :param chunk: The chunk returned by the OpenAI API.
        :param choice: The choice returned by the OpenAI API.
        :return: The StreamingChunk.
        """
        # function or tools calls are not going to happen in non-chat generation
        # as users can not send ChatMessage with function or tools calls
        choice = chunk.choices[0]
        content = choice.delta.content or ""
        chunk_message = StreamingChunk(content)
        chunk_message.meta.update({"model": chunk.model, "index": choice.index, "finish_reason": choice.finish_reason})
        return chunk_message

    def _check_finish_reason(self, message: ChatMessage) -> None:
        """
        Check the `finish_reason` returned with the OpenAI completions.
        If the `finish_reason` is `length`, log a warning to the user.
        :param message: The message returned by the LLM.
        """
        if message.meta["finish_reason"] == "length":
            logger.warning(
                "The completion for index %s has been truncated before reaching a natural stopping point. "
                "Increase the max_tokens parameter to allow for longer completions.",
                message.meta["index"],
            )
        if message.meta["finish_reason"] == "content_filter":
            logger.warning(
                "The completion for index %s has been truncated due to the content filter.", message.meta["index"]
            )


class GPTGenerator(OpenAIGenerator):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        api_base_url: Optional[str] = None,
        organization: Optional[str] = None,
        system_prompt: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        warnings.warn(
            "GPTGenerator is deprecated and will be removed in the next beta release. "
            "Please use OpenAIGenerator instead.",
            UserWarning,
            stacklevel=2,
        )
        super().__init__(
            api_key=api_key,
            model=model,
            streaming_callback=streaming_callback,
            api_base_url=api_base_url,
            organization=organization,
            system_prompt=system_prompt,
            generation_kwargs=generation_kwargs,
        )
