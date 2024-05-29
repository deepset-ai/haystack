# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import StreamingChunk
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable
from haystack.utils.hf import HFModelType, check_generation_params, check_valid_model, list_inference_deployed_models

with LazyImport(message="Run 'pip install \"huggingface_hub>=0.23.0\" transformers'") as transformers_import:
    from huggingface_hub import (
        InferenceClient,
        TextGenerationOutput,
        TextGenerationOutputToken,
        TextGenerationStreamOutput,
    )
    from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


@component
class HuggingFaceTGIGenerator:
    """
    Enables text generation using HuggingFace Hub hosted non-chat LLMs.

    This component is designed to seamlessly inference models deployed on the Text Generation Inference (TGI) backend.
    You can use this component for LLMs hosted on Hugging Face inference endpoints, the rate-limited
    Inference API tier.

    Key Features and Compatibility:
     - Primary Compatibility: designed to work seamlessly with any non-based model deployed using the TGI
       framework. For more information on TGI, visit [text-generation-inference](https://github.com/huggingface/text-generation-inference)

    - Hugging Face Inference Endpoints: Supports inference of TGI chat LLMs deployed on Hugging Face
       inference endpoints. For more details, refer to [inference-endpoints](https://huggingface.co/inference-endpoints)

    - Inference API Support: supports inference of TGI LLMs hosted on the rate-limited Inference
      API tier. Learn more about the Inference API at [inference-api](https://huggingface.co/inference-api).
      Discover available chat models using the following command: `wget -qO- https://api-inference.huggingface.co/framework/text-generation-inference | grep chat`
      and simply use the model ID as the model parameter for this component. You'll also need to provide a valid
      Hugging Face API token as the token parameter.

    - Custom TGI Endpoints: supports inference of TGI chat LLMs deployed on custom TGI endpoints. Anyone can
      deploy their own TGI endpoint using the TGI framework. For more details, refer to [inference-endpoints](https://huggingface.co/inference-endpoints)

     Input and Output Format:
      - String Format: This component uses the str format for structuring both input and output,
        ensuring coherent and contextually relevant responses in text generation scenarios.

    ```python
    from haystack.components.generators import HuggingFaceTGIGenerator
    from haystack.utils import Secret

    client = HuggingFaceTGIGenerator(model="mistralai/Mistral-7B-v0.1", token=Secret.from_token("<your-api-key>"))
    client.warm_up()
    response = client.run("What's Natural Language Processing?", generation_kwargs={"max_new_tokens": 120})
    print(response)
    ```

    Or for LLMs hosted on paid https://huggingface.co/inference-endpoints endpoint, and/or your own custom TGI endpoint.
    In these two cases, you'll need to provide the URL of the endpoint as well as a valid token:

    ```python
    from haystack.components.generators import HuggingFaceTGIGenerator
    client = HuggingFaceTGIGenerator(model="mistralai/Mistral-7B-v0.1",
                                     url="<your-tgi-endpoint-url>",
                                     token=Secret.from_token("<your-api-key>"))
    client.warm_up()
    response = client.run("What's Natural Language Processing?")
    print(response)
    ```
    """

    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-v0.1",
        url: Optional[str] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        generation_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Initialize the HuggingFaceTGIGenerator instance.

        :param model:
            A string representing the model id on HF Hub. Default is "mistralai/Mistral-7B-v0.1".
        :param url:
            An optional string representing the URL of the TGI endpoint. If the url is not provided, check if the model
            is deployed on the free tier of the HF inference API.
        :param token: The HuggingFace token to use as HTTP bearer authorization
            You can find your HF token in your [account settings](https://huggingface.co/settings/tokens)
        :param generation_kwargs:
            A dictionary containing keyword arguments to customize text generation.
                Some examples: `max_new_tokens`, `temperature`, `top_k`, `top_p`,...
                See Hugging Face's documentation for more information at: [TextGenerationParameters](https://huggingface.co/docs/huggingface_hub/v0.18.0.rc0/en/package_reference/inference_client#huggingface_hub.inference._text_generation.TextGenerationParameters
        :param stop_words: An optional list of strings representing the stop words.
        :param streaming_callback: An optional callable for handling streaming responses.
        """
        warnings.warn(
            "`HuggingFaceTGIGenerator` is deprecated and will be removed in Haystack 2.3.0."
            "Use `HuggingFaceAPIGenerator` instead.",
            DeprecationWarning,
        )

        transformers_import.check()

        if url:
            r = urlparse(url)
            is_valid_url = all([r.scheme in ["http", "https"], r.netloc])
            if not is_valid_url:
                raise ValueError(f"Invalid TGI endpoint URL provided: {url}")

        check_valid_model(model, HFModelType.GENERATION, token)

        # handle generation kwargs setup
        generation_kwargs = generation_kwargs.copy() if generation_kwargs else {}
        check_generation_params(generation_kwargs, ["n"])
        generation_kwargs["stop_sequences"] = generation_kwargs.get("stop_sequences", [])
        generation_kwargs["stop_sequences"].extend(stop_words or [])
        generation_kwargs.setdefault("max_new_tokens", 512)

        self.model = model
        self.url = url
        self.token = token
        self.generation_kwargs = generation_kwargs
        self.client = InferenceClient(url or model, token=token.resolve_value() if token else None)
        self.streaming_callback = streaming_callback
        self.tokenizer = None

    def warm_up(self) -> None:
        """
        Initializes the component.
        """
        if self.tokenizer is not None:
            return

        # is this user using HF free tier inference API?
        if self.model and not self.url:
            deployed_models = list_inference_deployed_models()
            # Determine if the specified model is deployed in the free tier.
            if self.model not in deployed_models:
                raise ValueError(
                    f"The model {self.model} is not deployed on the free tier of the HF inference API. "
                    "To use free tier models provide the model ID and the token. Valid models are: "
                    f"{deployed_models}"
                )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model, token=self.token.resolve_value() if self.token else None
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            A dictionary containing the serialized component.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.model,
            url=self.url,
            token=self.token.to_dict() if self.token else None,
            generation_kwargs=self.generation_kwargs,
            streaming_callback=callback_name,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceTGIGenerator":
        """
        Deserialize this component from a dictionary.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        # Don't send URL as it is sensitive information
        return {"model": self.model}

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Invoke the text generation inference for the given prompt and generation parameters.

        :param prompt:
            A string representing the prompt.
        :param generation_kwargs:
            Additional keyword arguments for text generation.
        :returns:
            A dictionary containing the generated replies and metadata. Both are lists of length n.
            - replies: A list of strings representing the generated replies.
        """
        if not self.tokenizer:
            raise RuntimeError(
                "The component HuggingFaceTGIGenerator was not warmed up. Please call warm_up() before running LLM inference."
            )

        # check generation kwargs given as parameters to override the default ones
        additional_params = ["n", "stop_words"]
        check_generation_params(generation_kwargs, additional_params)

        # update generation kwargs by merging with the default ones
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        num_responses = generation_kwargs.pop("n", 1)
        generation_kwargs.setdefault("stop_sequences", []).extend(generation_kwargs.pop("stop_words", []))

        prompt_token_count = len(self.tokenizer.encode(prompt, add_special_tokens=False))

        if self.streaming_callback:
            if num_responses > 1:
                raise ValueError("Cannot stream multiple responses, please set n=1.")

            return self._run_streaming(prompt, prompt_token_count, generation_kwargs)

        return self._run_non_streaming(prompt, prompt_token_count, num_responses, generation_kwargs)

    def _run_streaming(self, prompt: str, prompt_token_count: int, generation_kwargs: Dict[str, Any]):
        res_chunk: Iterable[TextGenerationStreamOutput] = self.client.text_generation(
            prompt, details=True, stream=True, **generation_kwargs
        )
        chunks: List[StreamingChunk] = []
        # pylint: disable=not-an-iterable
        for chunk in res_chunk:
            token: TextGenerationOutputToken = chunk.token
            if token.special:
                continue
            chunk_metadata = {**asdict(token), **(asdict(chunk.details) if chunk.details else {})}
            stream_chunk = StreamingChunk(token.text, chunk_metadata)
            chunks.append(stream_chunk)
            self.streaming_callback(stream_chunk)  # type: ignore # streaming_callback is not None (verified in the run method)
        metadata = {
            "finish_reason": chunks[-1].meta.get("finish_reason", None),
            "model": self.client.model,
            "usage": {
                "completion_tokens": chunks[-1].meta.get("generated_tokens", 0),
                "prompt_tokens": prompt_token_count,
                "total_tokens": prompt_token_count + chunks[-1].meta.get("generated_tokens", 0),
            },
        }
        return {"replies": ["".join([chunk.content for chunk in chunks])], "meta": [metadata]}

    def _run_non_streaming(
        self, prompt: str, prompt_token_count: int, num_responses: int, generation_kwargs: Dict[str, Any]
    ):
        responses: List[str] = []
        all_metadata: List[Dict[str, Any]] = []
        for _i in range(num_responses):
            tgr: TextGenerationOutput = self.client.text_generation(prompt, details=True, **generation_kwargs)
            if tgr.details:
                completion_tokens = len(tgr.details.tokens)
                prompt_token_count = prompt_token_count + completion_tokens
                finish_reason = tgr.details.finish_reason
            else:
                finish_reason = None
                completion_tokens = 0
            all_metadata.append(
                {
                    "model": self.client.model,
                    "index": _i,
                    "finish_reason": finish_reason,
                    "usage": {
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_token_count,
                        "total_tokens": prompt_token_count + completion_tokens,
                    },
                }
            )
            responses.append(tgr.generated_text)
        return {"replies": responses, "meta": all_metadata}
