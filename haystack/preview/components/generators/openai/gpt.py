import dataclasses
import logging
import os
import sys
from collections import defaultdict
from typing import Optional, List, Callable, Dict, Any, Union

import openai
from openai.openai_object import OpenAIObject

from haystack.preview import component, default_from_dict, default_to_dict, DeserializationError
from haystack.preview.dataclasses.chat_message import ChatMessage

logger = logging.getLogger(__name__)


API_BASE_URL = "https://api.openai.com/v1"


def default_streaming_callback(chunk):
    """
    Default callback function for streaming responses from OpenAI API.
    Prints the tokens of the first completion to stdout as soon as they are received and returns the chunk unchanged.
    """
    print(chunk.content, flush=True, end="")
    return chunk


@component
class GPTGenerator:
    """
    LLM Generator compatible with GPT (ChatGPT) large language models.

    Queries the LLM using OpenAI's API. Invocations are made using OpenAI SDK ('openai' package)
    See [OpenAI GPT API](https://platform.openai.com/docs/guides/chat) for more details.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-3.5-turbo",
        system_prompt: Optional[str] = None,
        streaming_callback: Optional[Callable] = None,
        api_base_url: str = API_BASE_URL,
        **kwargs,
    ):
        """
        Creates an instance of GPTGenerator. Unless specified otherwise in the `model_name`, this is for OpenAI's GPT-3.5 model.

        :param api_key: The OpenAI API key. It can be explicitly provided or automatically read from the
            environment variable OPENAI_API_KEY (recommended).
        :param model_name: The name of the model to use.
        :param system_prompt: An additional message to be sent to the LLM at the beginning of each conversation.
            Typically, a conversation is formatted with a system message first, followed by alternating messages from
            the 'user' (the "queries") and the 'assistant' (the "responses"). The system message helps set the behavior
            of the assistant. For example, you can modify the personality of the assistant or provide specific
            instructions about how it should behave throughout the conversation.
        :param streaming_callback: A callback function that is called when a new token is received from the stream.
            The callback function should accept two parameters: the token received from the stream and **kwargs.
            The callback function should return the token to be sent to the stream. If the callback function is not
            provided, the token is printed to stdout.
        :param api_base_url: The OpenAI API Base url, defaults to `https://api.openai.com/v1`.
        :param kwargs: Other parameters to use for the model. These parameters are all sent directly to the OpenAI
            endpoint. See OpenAI [documentation](https://platform.openai.com/docs/api-reference/chat) for more details.
            Some of the supported parameters:
            - `max_tokens`: The maximum number of tokens the output text can have.
            - `temperature`: What sampling temperature to use. Higher values mean the model will take more risks.
                Try 0.9 for more creative applications, and 0 (argmax sampling) for ones with a well-defined answer.
            - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model
                considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens
                comprising the top 10% probability mass are considered.
            - `n`: How many completions to generate for each prompt. For example, if the LLM gets 3 prompts and n is 2,
                it will generate two completions for each of the three prompts, ending up with 6 completions in total.
            - `stop`: One or more sequences after which the LLM should stop generating tokens.
            - `presence_penalty`: What penalty to apply if a token is already present at all. Bigger values mean
                the model will be less likely to repeat the same token in the text.
            - `frequency_penalty`: What penalty to apply if a token has already been generated in the text.
                Bigger values mean the model will be less likely to repeat the same token in the text.
            - `logit_bias`: Add a logit bias to specific tokens. The keys of the dictionary are tokens and the
                values are the bias to add to that token.
        """
        # if the user does not provide the API key, check if it is set in the module client
        api_key = api_key or openai.api_key
        if api_key is None:
            try:
                api_key = os.environ["OPENAI_API_KEY"]
            except KeyError as e:
                raise ValueError(
                    "GPTGenerator expects an OpenAI API key. "
                    "Set the OPENAI_API_KEY environment variable (recommended) or pass it explicitly."
                ) from e
        openai.api_key = api_key

        self.model_name = model_name
        self.system_prompt = system_prompt
        self.model_parameters = kwargs
        self.streaming_callback = streaming_callback

        self.api_base_url = api_base_url
        openai.api_base = api_base_url

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        if self.streaming_callback:
            module = self.streaming_callback.__module__
            if module == "builtins":
                callback_name = self.streaming_callback.__name__
            else:
                callback_name = f"{module}.{self.streaming_callback.__name__}"
        else:
            callback_name = None

        return default_to_dict(
            self,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            streaming_callback=callback_name,
            api_base_url=self.api_base_url,
            **self.model_parameters,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GPTGenerator":
        """
        Deserialize this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        streaming_callback = None
        if "streaming_callback" in init_params and init_params["streaming_callback"]:
            parts = init_params["streaming_callback"].split(".")
            module_name = ".".join(parts[:-1])
            function_name = parts[-1]
            module = sys.modules.get(module_name, None)
            if not module:
                raise DeserializationError(f"Could not locate the module of the streaming callback: {module_name}")
            streaming_callback = getattr(module, function_name, None)
            if not streaming_callback:
                raise DeserializationError(f"Could not locate the streaming callback: {function_name}")
            data["init_parameters"]["streaming_callback"] = streaming_callback
        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(self, prompt: Union[str, List[ChatMessage]]):
        """
        Queries the LLM with the prompts to produce replies.

        :param prompt: The prompts to be sent to the generative model.
        """
        messages: List[ChatMessage] = []
        if isinstance(prompt, str):
            message = ChatMessage.from_user(prompt)
            messages = [ChatMessage.from_system(self.system_prompt), message] if self.system_prompt else [message]
        elif isinstance(prompt, list) and all(isinstance(message, ChatMessage) for message in prompt):
            messages = prompt
        else:
            raise ValueError(
                f"Invalid prompt. Expected either a string or a list of ChatMessage(s), but got {type(prompt)}"
            )
        openai_chat_message_format = ["role", "content", "name"]
        completion = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                dataclasses.asdict(
                    m, dict_factory=lambda obj: {k: v for k, v in obj if k in openai_chat_message_format and v}
                )
                for m in messages
            ],
            stream=self.streaming_callback is not None,
            **self.model_parameters,
        )

        completions: List[ChatMessage]
        if self.streaming_callback:
            # buckets for n responses
            chunk_buckets = defaultdict(list)
            for chunk in completion:
                if chunk.choices:
                    # we always get a chunk with a single choice.
                    # the index idx of a choice varies, idx < number of requested completions
                    chunk_delta: ChatMessage = self._build_chunk(chunk, chunk.choices[0])
                    index = int(chunk_delta.metadata["index"])
                    chunk_buckets[index].append(chunk_delta)
                    # invoke callback with the chunk_delta
                    self.streaming_callback(chunk_delta)
            completions = self._collect_chunks(chunk_buckets)
        else:
            completions = [self._build_message(completion, choice) for choice in completion.choices]

        # before returning, do post-processing of the completions
        for completion in completions:
            self._post_receive(completion)

        return {"replies": completions}

    def _build_message(self, completion: OpenAIObject, choice: OpenAIObject) -> ChatMessage:
        """
        Converts the response from the OpenAI API to a ChatMessage.
        """
        message: OpenAIObject = choice.message
        content = dict(message.function_call) if choice.finish_reason == "function_call" else message.content
        chat_message = ChatMessage.from_assistant(content)
        chat_message.metadata.update(
            {
                "model": completion.model,
                "index": choice.index,
                "finish_reason": choice.finish_reason,
                "usage": dict(completion.usage.items()),
            }
        )
        return chat_message

    def _build_chunk(self, chunk: OpenAIObject, choice: OpenAIObject) -> ChatMessage:
        """
        Converts the response from the OpenAI API to a ChatMessage.
        """
        has_content = bool(hasattr(choice.delta, "content") and choice.delta.content)
        if has_content:
            content = choice.delta.content
        elif hasattr(choice.delta, "function_call"):
            content = str(choice.delta.function_call)
        else:
            content = ""
        # TODO: these should perhaps be ChatMessageChunk objects
        chunk_message = ChatMessage.from_assistant(content)
        chunk_message.metadata.update(
            {"model": chunk.model, "index": choice.index, "finish_reason": choice.finish_reason}
        )
        return chunk_message

    def _collect_chunks(self, chunk_buckets: Dict[int, List[ChatMessage]]):
        content_list = ["".join([chunk.content for chunk in chunk_list]) for chunk_list in chunk_buckets.values()]
        replies: List[ChatMessage] = [
            # take metadata from the last chunk in the bucket
            ChatMessage.from_assistant(content, chunk_buckets[i][-1].metadata)
            for i, content in enumerate(content_list)
        ]
        return replies

    def _check_finish_reason(self, message: ChatMessage) -> None:
        """
        Check the `finish_reason` returned with the OpenAI completions.
        If the `finish_reason` is `length`, log a warning to the user.
        :param message: The message returned by the LLM.
        """
        if message.metadata["finish_reason"] == "length":
            logger.warning(
                "The completion for index %s has been truncated before reaching a natural stopping point. "
                "Increase the max_tokens parameter to allow for longer completions.",
                message.metadata["index"],
            )
        if message.metadata["finish_reason"] == "content_filter":
            logger.warning(
                "The completion for index %s has been truncated due to the content filter.", message.metadata["index"]
            )

    def _post_receive(self, message: ChatMessage) -> None:
        """
        Post-processing of the message received from the LLM.
        :param message: The message returned by the LLM.
        """
        self._check_finish_reason(message)
