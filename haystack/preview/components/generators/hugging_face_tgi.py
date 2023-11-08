import logging
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Iterable, Callable
from urllib.parse import urlparse

from haystack.preview import component, default_to_dict, default_from_dict
from haystack.preview.components.generators.utils import serialize_callback_handler, deserialize_callback_handler
from haystack.preview.dataclasses import StreamingChunk
from haystack.preview.components.generators.hf_utils import check_generation_params, check_valid_model
from haystack.preview.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install transformers'") as transformers_import:
    from huggingface_hub import InferenceClient
    from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, TextGenerationResponse, Token
    from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


@component
class HuggingFaceTGIGenerator:
    """
    Enables text generation using HuggingFace Hub hosted non-chat LLMs. This component is designed to seamlessly
    inference models deployed on the Text Generation Inference (TGI) backend.

    You can use this component for LLMs hosted on Hugging Face inference endpoints, the rate-limited
    Inference API tier:

    ```python
    from haystack.preview.components.generators import HuggingFaceTGIGenerator
    client = HuggingFaceTGIGenerator(model="mistralai/Mistral-7B-v0.1", token="<your-token>")
    client.warm_up()
    response = client.run("What's Natural Language Processing?", max_new_tokens=120)
    print(response)
    ```

    Or for LLMs hosted on paid https://huggingface.co/inference-endpoints endpoint, and/or your own custom TGI endpoint.
    In these two cases, you'll need to provide the URL of the endpoint as well as a valid token:

    ```python
    from haystack.preview.components.generators import HuggingFaceTGIGenerator
    client = HuggingFaceTGIGenerator(model="mistralai/Mistral-7B-v0.1",
                                     url="<your-tgi-endpoint-url>",
                                     token="<your-token>")
    client.warm_up()
    response = client.run("What's Natural Language Processing?", max_new_tokens=120)
    print(response)
    ```


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
        model: str = "mistralai/Mistral-7B-v0.1",
        url: Optional[str] = None,
        token: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Initialize the HuggingFaceTGIGenerator instance.

        :param model: A string representing the model id on HF Hub. Default is "mistralai/Mistral-7B-v0.1".
        :param url: An optional string representing the URL of the TGI endpoint.
        :param token: The HuggingFace token to use as HTTP bearer authorization
            You can find your HF token at https://huggingface.co/settings/tokens
        :param generation_kwargs: A dictionary containing keyword arguments to customize text generation.
            Some examples: `max_new_tokens`, `temperature`, `top_k`, `top_p`,...
            See Hugging Face's documentation for more information at:
            https://huggingface.co/docs/huggingface_hub/v0.18.0.rc0/en/package_reference/inference_client#huggingface_hub.inference._text_generation.TextGenerationParameters
        :param stop_words: An optional list of strings representing the stop words.
        :param streaming_callback: An optional callable for handling streaming responses.
        """
        transformers_import.check()

        if url:
            r = urlparse(url)
            is_valid_url = all([r.scheme in ["http", "https"], r.netloc])
            if not is_valid_url:
                raise ValueError(f"Invalid TGI endpoint URL provided: {url}")

        check_valid_model(model, token)

        # handle generation kwargs setup
        generation_kwargs = generation_kwargs.copy() if generation_kwargs else {}
        check_generation_params(generation_kwargs, ["n"])
        generation_kwargs["stop_sequences"] = generation_kwargs.get("stop_sequences", [])
        generation_kwargs["stop_sequences"].extend(stop_words or [])

        self.model = model
        self.url = url
        self.token = token
        self.generation_kwargs = generation_kwargs
        self.client = InferenceClient(url or model, token=token)
        self.streaming_callback = streaming_callback
        self.tokenizer = None

    def warm_up(self) -> None:
        """
        Load the tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, token=self.token)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :return: A dictionary containing the serialized component.
        """
        callback_name = serialize_callback_handler(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.model,
            url=self.url,
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
        # Don't send URL as it is sensitive information
        return {"model": self.model}

    @component.output_types(replies=List[str], metadata=List[Dict[str, Any]])
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
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
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        num_responses = generation_kwargs.pop("n", 1)
        generation_kwargs.setdefault("stop_sequences", []).extend(generation_kwargs.pop("stop_words", []))

        if self.tokenizer is None:
            raise RuntimeError("Please call warm_up() before running LLM inference.")

        prompt_token_count = len(self.tokenizer.encode(prompt, add_special_tokens=False))

        if self.streaming_callback:
            if num_responses > 1:
                raise ValueError("Cannot stream multiple responses, please set n=1.")

            return self._run_streaming(prompt, prompt_token_count, generation_kwargs)

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
            self.streaming_callback(stream_chunk)  # type: ignore # streaming_callback is not None (verified in the run method)
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
