import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Iterable, Callable
from urllib.parse import urlparse

from huggingface_hub import InferenceClient
from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, TextGenerationResponse, Token
from transformers import AutoTokenizer

from haystack.preview import component, default_to_dict, default_from_dict
from haystack.preview.components.generators.hf_utils import check_valid_model, check_generation_params
from haystack.preview.components.generators.utils import serialize_callback_handler, deserialize_callback_handler
from haystack.preview.dataclasses import ChatMessage, StreamingChunk

logger = logging.getLogger(__name__)


class ChatHuggingFaceTGIGenerator:
    """
    ChatHuggingFaceTGIGenerator inferences remote Hugging Face chat models for text generation. It is designed to
    work with any HuggingFace inference endpoint (https://huggingface.co/inference-endpoints) as well as models deployed
    with the Text Generation Inference (TGI) framework (https://github.com/huggingface/text-generation-inference).
    It can also use TGI models on the rate-limited tier called Inference API (https://huggingface.co/inference-api).
    The list of available models can be viewed with the command:
    ```
    wget -qO- https://api-inference.huggingface.co/framework/text-generation-inference | grep chat
    ```
    ChatHuggingFaceTGIGenerator uses the ChatMessage format for both input and output, which is defined at
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
        Initialize the ChatHuggingFaceTGIGenerator instance.

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
        else:
            check_valid_model(model, token)

        # handle generation kwargs
        generation_kwargs = generation_kwargs.copy() if generation_kwargs else {}
        check_generation_params(generation_kwargs, ["n"])
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
        callback_name = serialize_callback_handler(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.client.model,
            model_id=self.model_id,
            token=self.token if not isinstance(self.token, str) else None,  # don't serialize valid tokens
            generation_kwargs=self.generation_kwargs,
            streaming_callback=callback_name,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChatHuggingFaceTGIGenerator":
        """
        Deserialize this component from a dictionary.
        """
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callback_handler(serialized_callback_handler)
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
        check_generation_params(generation_kwargs, additional_params)

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
