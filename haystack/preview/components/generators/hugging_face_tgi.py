import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Iterable, Callable
from urllib.parse import urlparse

from huggingface_hub import InferenceClient
from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, TextGenerationResponse, Token
from transformers import AutoTokenizer

from haystack.preview import component, default_to_dict, default_from_dict
from haystack.preview.components.generators.hf_utils import check_generation_params, check_valid_model
from haystack.preview.components.generators.utils import serialize_callback_handler, deserialize_callback_handler
from haystack.preview.dataclasses import StreamingChunk

logger = logging.getLogger(__name__)


@component
class HuggingFaceTGIGenerator:
    """
    Enables text generation using HuggingFace Hub hosted non-chat LLMs. This component is designed to seamlessly
    inference models deployed on the Text Generation Inference (TGI) backend.

     Key Features and Compatibility:
         - **Primary Compatibility**: Designed to work seamlessly with any non-chat model deployed using the TGI
           framework. For more information on TGI, visit https://github.com/huggingface/text-generation-inference.
         - **Hugging Face Inference Endpoints**: Supports inference of TGI chat LLMs deployed on Hugging Face
           inference endpoints. For more details refer to https://huggingface.co/inference-endpoints.
         - **Inference API Support**: Supports inference of TGI LLMs hosted on the rate-limited Inference
           API tier. Learn more about the Inference API at: https://huggingface.co/inference-api
           Discover available LLMs using the following command:
           ```
           wget -qO- https://api-inference.huggingface.co/framework/text-generation-inference
           ```
           And simply use the model ID as the model parameter for this component. You'll also need to provide a valid
           Hugging Face API token as the token parameter.
         - **Custom TGI Endpoints**: Supports inference of LLMs deployed on custom TGI endpoints. Anyone can
           deploy their own TGI endpoint using the TGI framework. For more details refer
           to https://huggingface.co/inference-endpoints.
     Input and Output Format:
         - **String Format**: This component uses the str format for structuring both input and output,
           ensuring coherent and contextually relevant responses in text generation scenarios.
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
        Initialize the HuggingFaceTGIGenerator instance.

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
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceTGIGenerator":
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

    @component.output_types(replies=List[str], metadata=List[Dict[str, Any]])
    def run(self, prompt: str, **generation_kwargs):
        """
        Invoke the text generation inference for the given prompt and generation parameters.

        :param prompt: A string representing the prompt.
        :param generation_kwargs: Additional keyword arguments for text generation.
        :return: A dictionary containing the generated replies and metadata. Both are lists of length n.
        Replies are strings and metadata are dictionaries.
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
