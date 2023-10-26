import logging
from dataclasses import fields, asdict
from typing import Any, Dict, List, Optional, Union, Iterable, Callable
from urllib.parse import urlparse

from huggingface_hub import InferenceClient, HfApi
from huggingface_hub.inference._text_generation import (
    TextGenerationStreamResponse,
    TextGenerationParameters,
    TextGenerationResponse,
    Token,
)
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import AutoTokenizer

from haystack.preview import component, default_to_dict
from haystack.preview.dataclasses import ChatMessage, StreamingChunk

logger = logging.getLogger(__name__)


def check_generation_params(kwargs: Dict[str, Any], additional_params: Optional[List[str]] = None):
    """
    Check the provided generation parameters for validity.

    :param kwargs: A dictionary containing the generation parameters.
    :param additional_params: An optional list of strings representing additional parameters.
    :raises ValueError: If any unknown text generation parameters are provided.
    """
    if kwargs:
        accepted_params = {field.name for field in fields(TextGenerationParameters)}
        if additional_params:
            accepted_params.update(additional_params)
        unknown_params = set(kwargs.keys()) - accepted_params
        if unknown_params:
            raise ValueError(
                f"Unknown text generation parameters: {unknown_params}, please provide {accepted_params} only."
            )


def check_valid_model(model_id: str, token: Optional[str]):
    """
    Check if the provided model ID corresponds to a valid model on HuggingFace Hub.
    Also check if the model is a text generation model.

    :param model_id: A string representing the HuggingFace model ID.
    :param token: An optional string representing the authentication token.
    :raises ValueError: If the model is not found or is not a text generation model.
    """
    api = HfApi()
    try:
        model_info = api.model_info(model_id, token=token)
    except RepositoryNotFoundError as e:
        raise ValueError(
            f"Model {model_id} not found on HuggingFace Hub. Please provide a valid HuggingFace model_id."
        ) from e

    allowed_model = model_info.pipeline_tag in ["text-generation", "text2text-generation"]
    if not allowed_model:
        raise ValueError(f"Model {model_id} is not a text generation model. Please provide a text generation model.")


def convert_to_chat_message(response: TextGenerationResponse, model_id: Optional[str] = None) -> ChatMessage:
    """
    Convert a TextGenerationResponse instance to a ChatMessage instance.

    :param response: A TextGenerationResponse instance representing the text generation response.
    :param model_id: An optional string representing the HuggingFace model ID.
    :return: A ChatMessage instance representing the converted response.
    """
    message = ChatMessage.from_assistant(response.generated_text)
    # TODO add token usage to metadata, it is in res.details.tokens
    message.metadata.update(
        {"finish_reason": response.details.finish_reason.value, "index": 0, "model": model_id or "unknown"}
    )
    return message


@component
class HuggingFaceRemoteGenerator:
    """
    HuggingFaceRemoteGenerator inferences remote Hugging Face models for text generation. It is designed to work
    with any HuggingFace inference endpoint (https://huggingface.co/inference-endpoints) as well as models deployed
    with the Text Generation Inference (TGI) framework (https://github.com/huggingface/text-generation-inference)
    hosting non-chat models.

    It can also use TGI based non-chat models on the rate-limited Inference API (https://huggingface.co/inference-api).
    The list of available models can be viewed with the command:
    ```
    wget -qO- https://api-inference.huggingface.co/framework/text-generation-inference
    ```
    HuggingFaceRemoteGenerator uses strings for both input and output.
    """

    def __init__(
        self,
        model: str = "HuggingFaceH4/zephyr-7b-alpha",
        model_id: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        streaming_callback: Optional[Callable] = None,
    ):
        """
        Initialize the HuggingFaceRemoteGenerator instance.

        :param model: A string representing the model path or URL. Default is "HuggingFaceH4/zephyr-7b-alpha".
        :param model_id: An optional string representing the HuggingFace model ID.
        :param token: An optional string or boolean representing the authentication token.
        :param generation_kwargs: An optional dictionary containing generation parameters.
        :param stop_words: An optional list of strings representing the stop words.
        :param streaming_callback: An optional callable for handling streaming responses.
        """
        r = urlparse(model)
        is_url = all([r.scheme in ["http", "https"], r.netloc])
        if is_url:
            if not model_id:
                raise ValueError(
                    "If model is a URL, you must provide a HuggingFace model_id (e.g.mistralai/Mistral-7B-v0.1)"
                )
            check_valid_model(model_id, token)
        else:
            check_valid_model(model, token)

        generation_kwargs = generation_kwargs or {}
        check_generation_params(generation_kwargs, ["n"])
        generation_kwargs["stop_sequences"] = generation_kwargs.get("stop_sequences", []) + (stop_words or [])
        self.generation_kwargs = generation_kwargs

        self.client = InferenceClient(model, token=token)
        self.streaming_callback = streaming_callback

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        pass

    @component.output_types(replies=List[str])
    def run(self, prompt: str, **generation_kwargs):
        """
        Invoke the text generation inference for the given prompt and generation parameters.

        :param prompt: A string representing the prompt.
        :param generation_kwargs: Additional keyword arguments for text generation.
        :return: A list containing the generated responses as ChatMessage instances.
        """
        num_responses = generation_kwargs.get("n", 1)

        # check for unknown kwargs
        check_generation_params(generation_kwargs, ["n"])

        # update generation kwargs with invocation provided kwargs
        generation_kwargs = {**self.generation_kwargs, **generation_kwargs}

        if self.streaming_callback:
            if num_responses > 1:
                raise ValueError("Cannot stream multiple responses, please set n=1.")

            res: Iterable[TextGenerationStreamResponse] = self.client.text_generation(
                prompt, details=True, **generation_kwargs
            )
            chunks: List[ChatMessage] = []
            for chunk in res:
                token: Token = chunk.token
                if token.special:
                    continue
                chunk_metadata = {**asdict(token), **(asdict(chunk.details) if chunk.details else {})}
                chunk = StreamingChunk(token.text, chunk_metadata)
                self.streaming_callback(chunk)
            return {"replies": ["".join([chunk.content for chunk in chunks])]}
        else:
            responses: List[str] = []
            for _i in range(num_responses):
                res: TextGenerationResponse = self.client.text_generation(prompt, **generation_kwargs)
                responses.append(res.generated_text)
            return {"replies": [responses]}


class ChatHuggingFaceRemoteGenerator:
    """
    ChatHuggingFaceRemoteGenerator inferences remote Hugging Face chat models for text generation. It is designed to work
    with any HuggingFace inference endpoint (https://huggingface.co/inference-endpoints) as well as models deployed
    with the Text Generation Inference (TGI) framework (https://github.com/huggingface/text-generation-inference).
    It can also use TGI based models on the rate-limited tier called Inference API (https://huggingface.co/inference-api).
    The list of available models can be viewed with the command:
    ```
    wget -qO- https://api-inference.huggingface.co/framework/text-generation-inference | grep chat
    ```
    ChatHuggingFaceRemoteGenerator uses the ChatMessage format for both input and output, which is defined at
    https://github.com/openai/openai-python/blob/main/chatml.md. This format facilitates the representation of chat
    conversations in a structured manner, which is crucial for generating coherent and contextually relevant responses
    in a chat-based text generation scenario.

    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-2-13b-chat-hf",
        model_id: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        streaming_callback: Optional[Callable] = None,
    ):
        """
        Initialize the ChatHuggingFaceRemoteGenerator instance.

        :param model: A string representing the model path or URL. Default is "meta-llama/Llama-2-13b-chat-hf".
        :param model_id: An optional string representing the HuggingFace model ID.
        :param token: An optional string or boolean representing the authentication token.
        :param generation_kwargs: An optional dictionary containing generation parameters.
        :param stop_words: An optional list of strings representing the stop words.
        :param streaming_callback: An optional callable for handling streaming responses.
        """
        r = urlparse(model)
        is_url = all([r.scheme in ["http", "https"], r.netloc])
        if is_url:
            if not model_id:
                raise ValueError(
                    "If model is a URL, you must provide a HuggingFace model_id (e.g. meta-llama/Llama-2-7b-chat-hf)"
                )
            check_valid_model(model_id, token)
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        else:
            check_valid_model(model, token)
            self.tokenizer = AutoTokenizer.from_pretrained(model, token=token)

        generation_kwargs = generation_kwargs or {}
        check_generation_params(generation_kwargs, ["n"])

        generation_kwargs["stop_sequences"] = generation_kwargs.get("stop_sequences", []) + (stop_words or [])
        self.generation_kwargs = generation_kwargs

        self.client = InferenceClient(model, token=token)
        self.streaming_callback = streaming_callback

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.

        :return: A dictionary containing the telemetry data.
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize the component to a dictionary.

        :return: A dictionary representing the serialized component.
        """
        pass

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], **generation_kwargs):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param generation_kwargs: Additional keyword arguments for text generation.
        :return: A list containing the generated responses as ChatMessage instances.
        """
        num_responses = generation_kwargs.get("n", 1)

        # check for unknown kwargs
        check_generation_params(generation_kwargs, ["n"])

        # update generation kwargs with invocation provided kwargs
        generation_kwargs = {**self.generation_kwargs, **generation_kwargs}

        # apply chat template to messages to get string prompt
        prepared_prompt: str = self.tokenizer.apply_chat_template(messages, tokenize=False)

        if self.streaming_callback:
            if num_responses > 1:
                raise ValueError("Cannot stream multiple responses, please set n=1.")

            res: Iterable[TextGenerationStreamResponse] = self.client.text_generation(
                prepared_prompt, stream=True, details=True, **generation_kwargs
            )
            chunks: List[StreamingChunk] = []
            for chunk in res:
                token: Token = chunk.token
                if token.special:
                    continue
                chunk_metadata = {**asdict(token), **(asdict(chunk.details) if chunk.details else {})}
                chunk = StreamingChunk(token.text, chunk_metadata)
                self.streaming_callback(chunk)
                chunks.append(chunk)
            return {"replies": [ChatMessage.from_assistant("".join([chunk.content for chunk in chunks]))]}
        else:
            chat_messages: List[ChatMessage] = []
            for _i in range(num_responses):
                res: TextGenerationResponse = self.client.text_generation(
                    prepared_prompt, details=True, **generation_kwargs
                )
                chat_messages.append(convert_to_chat_message(res))
            return {"replies": [chat_messages]}
