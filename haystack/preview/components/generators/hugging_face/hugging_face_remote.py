import inspect
import logging
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Iterable, Callable
from urllib.parse import urlparse

from huggingface_hub import InferenceClient, HfApi
from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, TextGenerationResponse, Token
from huggingface_hub.utils import RepositoryNotFoundError
from transformers import AutoTokenizer

from haystack.preview import component, default_to_dict, default_from_dict, DeserializationError
from haystack.preview.dataclasses import ChatMessage, StreamingChunk

logger = logging.getLogger(__name__)


def _check_generation_params(kwargs: Dict[str, Any], additional_accepted_params: Optional[List[str]] = None):
    """
    Check the provided generation parameters for validity.

    :param kwargs: A dictionary containing the generation parameters.
    :param additional_accepted_params: An optional list of strings representing additional accepted parameters.
    :raises ValueError: If any unknown text generation parameters are provided.
    """
    if kwargs:
        accepted_params = {
            param
            for param in inspect.signature(InferenceClient.text_generation).parameters.keys()
            if param not in ["self", "prompt"]
        }
        if additional_accepted_params:
            accepted_params.update(additional_accepted_params)
        unknown_params = set(kwargs.keys()) - accepted_params
        if unknown_params:
            raise ValueError(
                f"Unknown text generation parameters: {unknown_params}, please provide {accepted_params} only."
            )


def _check_valid_model(model_id: str, token: Optional[str]) -> None:
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


def _serialize_callback_handler(streaming_callback: Callable[[StreamingChunk], None]) -> str:
    """
    Serialize the streaming callback handler.
    """
    module = streaming_callback.__module__
    if module == "builtins":
        callback_name = streaming_callback.__name__
    else:
        callback_name = f"{module}.{streaming_callback.__name__}"
    return callback_name


def _deserialize_callback_handler(callback_name: str) -> Optional[Callable[[StreamingChunk], None]]:
    """
    Deserialize the streaming callback handler.
    """
    parts = callback_name.split(".")
    module_name = ".".join(parts[:-1])
    function_name = parts[-1]
    module = sys.modules.get(module_name, None)
    if not module:
        raise DeserializationError(f"Could not locate the module of the streaming callback: {module_name}")
    streaming_callback = getattr(module, function_name, None)
    if not streaming_callback:
        raise DeserializationError(f"Could not locate the streaming callback: {function_name}")
    return streaming_callback


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
        token: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
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
            _check_valid_model(model_id, token)
        else:
            _check_valid_model(model, token)

        # handle generation kwargs
        generation_kwargs = generation_kwargs.copy() if generation_kwargs else {}
        _check_generation_params(generation_kwargs, ["n"])
        generation_kwargs.setdefault("stop_sequences", []).extend(stop_words or [])

        self.model_id = model_id if is_url and model_id else model
        self.token = token
        self.generation_kwargs = generation_kwargs
        self.client = InferenceClient(model, token=token)
        self.streaming_callback = streaming_callback
        self.tokenizer = None

    def warm_up(self) -> None:
        """
        Load the tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.token)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :return: A dictionary containing the serialized component.
        """
        callback_name = _serialize_callback_handler(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.client.model,
            model_id=self.model_id,
            token=self.token if not isinstance(self.token, str) else None,  # don't serialize valid tokens
            generation_kwargs=self.generation_kwargs,
            streaming_callback=callback_name,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceRemoteGenerator":
        """
        Deserialize this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = _deserialize_callback_handler(serialized_callback_handler)
        return default_from_dict(cls, data)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        # prefer model_id if available as model might be a URL and sensitive
        # Use model only if it is not a URL
        return {"model": self.model_id if self.model_id else self.client.model}

    @component.output_types(replies=List[str], metadata=List[Dict[str, Any]])
    def run(self, prompt: str, **generation_kwargs):
        """
        Invoke the text generation inference for the given prompt and generation parameters.

        :param prompt: A string representing the prompt.
        :param generation_kwargs: Additional keyword arguments for text generation.
        :return: A list containing the generated responses as ChatMessage instances.
        """
        # check generation kwargs given as parameters to override the default ones
        additional_params = ["n", "stop_words"]
        _check_generation_params(generation_kwargs, additional_params)

        # update generation kwargs by merging with the default ones
        generation_kwargs = {**self.generation_kwargs, **generation_kwargs}
        num_responses = generation_kwargs.pop("n", 1)
        generation_kwargs.setdefault("stop_sequences", []).extend(generation_kwargs.pop("stop_words", []))

        if self.tokenizer is None:
            raise RuntimeError("Please call warm_up() before running LLM inference.")

        prompt_token_count = len(self.tokenizer.encode(prompt, add_special_tokens=False))

        if self.streaming_callback:
            if num_responses > 1:
                raise ValueError("Cannot stream multiple responses, please set n=1.")

            return self._run_streaming(prompt, prompt_token_count, generation_kwargs)
        else:
            return self._run_non_streaming(prompt, prompt_token_count, num_responses, generation_kwargs)

    def _run_streaming(self, prompt: str, prompt_token_count: int, generation_kwargs: Dict[str, Any]):
        res_chunk: Iterable[TextGenerationStreamResponse] = self.client.text_generation(
            prompt, details=True, stream=True, **generation_kwargs
        )
        chunks: List[StreamingChunk] = []
        # pylint: disable=not-an-iterable
        for chunk in res_chunk:
            token: Token = chunk.token
            if token.special:
                continue
            chunk_metadata = {**asdict(token), **(asdict(chunk.details) if chunk.details else {})}
            stream_chunk = StreamingChunk(token.text, chunk_metadata)
            chunks.append(stream_chunk)
            self.streaming_callback(stream_chunk)  # type: ignore # guaranteed non-None by if statement above
        metadata = {
            "finish_reason": chunks[-1].metadata.get("finish_reason", None),
            "model": self.client.model,
            "usage": {
                "completion_tokens": chunks[-1].metadata.get("generated_tokens", 0),
                "prompt_tokens": prompt_token_count,
                "total_tokens": prompt_token_count + chunks[-1].metadata.get("generated_tokens", 0),
            },
        }
        return {"replies": ["".join([chunk.content for chunk in chunks])], "metadata": [metadata]}

    def _run_non_streaming(
        self, prompt: str, prompt_token_count: int, num_responses: int, generation_kwargs: Dict[str, Any]
    ):
        responses: List[str] = []
        all_metadata: List[Dict[str, Any]] = []
        for _i in range(num_responses):
            tgr: TextGenerationResponse = self.client.text_generation(prompt, details=True, **generation_kwargs)
            all_metadata.append(
                {
                    "model": self.client.model,
                    "index": _i,
                    "finish_reason": tgr.details.finish_reason.value,
                    "usage": {
                        "completion_tokens": len(tgr.details.tokens),
                        "prompt_tokens": prompt_token_count,
                        "total_tokens": prompt_token_count + len(tgr.details.tokens),
                    },
                }
            )
            responses.append(tgr.generated_text)
        return {"replies": responses, "metadata": all_metadata}


class ChatHuggingFaceRemoteGenerator:
    """
    ChatHuggingFaceRemoteGenerator inferences remote Hugging Face chat models for text generation. It is designed to
    work with any HuggingFace inference endpoint (https://huggingface.co/inference-endpoints) as well as models deployed
    with the Text Generation Inference (TGI) framework (https://github.com/huggingface/text-generation-inference).
    It can also use TGI models on the rate-limited tier called Inference API (https://huggingface.co/inference-api).
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
        token: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
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
            _check_valid_model(model_id, token)
        else:
            _check_valid_model(model, token)

        # handle generation kwargs
        generation_kwargs = generation_kwargs.copy() if generation_kwargs else {}
        _check_generation_params(generation_kwargs, ["n"])
        generation_kwargs.setdefault("stop_sequences", []).extend(stop_words or [])

        self.model_id = model_id if is_url and model_id else model
        self.token = token
        self.generation_kwargs = generation_kwargs
        self.client = InferenceClient(model, token=token)
        self.streaming_callback = streaming_callback
        self.tokenizer = None

    def warm_up(self) -> None:
        """
        Load the tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.token)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :return: A dictionary containing the serialized component.
        """
        callback_name = _serialize_callback_handler(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.client.model,
            model_id=self.model_id,
            token=self.token if not isinstance(self.token, str) else None,  # don't serialize valid tokens
            generation_kwargs=self.generation_kwargs,
            streaming_callback=callback_name,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceRemoteGenerator":
        """
        Deserialize this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = _deserialize_callback_handler(serialized_callback_handler)
        return default_from_dict(cls, data)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        # prefer model_id if available as model might be a URL and sensitive
        # Use model only if it is not a URL
        return {"model": self.model_id if self.model_id else self.client.model}

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], **generation_kwargs):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param generation_kwargs: Additional keyword arguments for text generation.
        :return: A list containing the generated responses as ChatMessage instances.
        """

        # check generation kwargs given as parameters to override the default ones
        additional_params = ["n", "stop_words"]
        _check_generation_params(generation_kwargs, additional_params)

        # update generation kwargs by merging with the default ones
        generation_kwargs = {**self.generation_kwargs, **generation_kwargs}
        num_responses = generation_kwargs.pop("n", 1)
        generation_kwargs.setdefault("stop_sequences", []).extend(generation_kwargs.pop("stop_words", []))

        if self.tokenizer is None:
            raise RuntimeError("Please call warm_up() before running LLM inference.")

        # apply chat template to messages to get string prompt
        prepared_prompt: str = self.tokenizer.apply_chat_template(messages, tokenize=False)
        prompt_token_count: int = len(self.tokenizer.encode(prepared_prompt, add_special_tokens=False))

        if self.streaming_callback:
            if num_responses > 1:
                raise ValueError("Cannot stream multiple responses, please set n=1.")

            return self._run_streaming(prepared_prompt, prompt_token_count, generation_kwargs)
        else:
            return self._run_non_streaming(prepared_prompt, prompt_token_count, num_responses, generation_kwargs)

    def _run_streaming(
        self, prepared_prompt: str, prompt_token_count: int, generation_kwargs: Dict[str, Any]
    ) -> Dict[str, List[ChatMessage]]:
        res: Iterable[TextGenerationStreamResponse] = self.client.text_generation(
            prepared_prompt, stream=True, details=True, **generation_kwargs
        )
        chunks: List[StreamingChunk] = []
        # pylint: disable=not-an-iterable
        for chunk in res:
            token: Token = chunk.token
            if token.special:
                continue
            chunk_metadata = {**asdict(token), **(asdict(chunk.details) if chunk.details else {})}
            stream_chunk = StreamingChunk(token.text, chunk_metadata)
            self.streaming_callback(stream_chunk)  # type: ignore # guaranteed non-None by if statement above
            chunks.append(stream_chunk)
        message = ChatMessage.from_assistant("".join([chunk.content for chunk in chunks]))
        message.metadata.update(
            {
                "finish_reason": chunks[-1].metadata.get("finish_reason", None),
                "model": self.client.model,
                "usage": {
                    "completion_tokens": chunks[-1].metadata.get("generated_tokens", 0),
                    "prompt_tokens": prompt_token_count,
                    "total_tokens": prompt_token_count + chunks[-1].metadata.get("generated_tokens", 0),
                },
            }
        )
        return {"replies": [message]}

    def _run_non_streaming(
        self, prepared_prompt: str, prompt_token_count: int, num_responses: int, generation_kwargs: Dict[str, Any]
    ) -> Dict[str, List[ChatMessage]]:
        chat_messages: List[ChatMessage] = []
        for _i in range(num_responses):
            tgr: TextGenerationResponse = self.client.text_generation(
                prepared_prompt, details=True, **generation_kwargs
            )
            message = ChatMessage.from_assistant(tgr.generated_text)
            message.metadata.update(
                {
                    "finish_reason": tgr.details.finish_reason.value,
                    "index": _i,
                    "model": self.client.model,
                    "usage": {
                        "completion_tokens": len(tgr.details.tokens),
                        "prompt_tokens": prompt_token_count,
                        "total_tokens": prompt_token_count + len(tgr.details.tokens),
                    },
                }
            )
            chat_messages.append(message)
        return {"replies": chat_messages}
