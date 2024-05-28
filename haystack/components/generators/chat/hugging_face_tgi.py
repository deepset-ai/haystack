# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib.parse import urlparse

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage, StreamingChunk
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
class HuggingFaceTGIChatGenerator:
    """
    A Chat-based text generation component using Hugging Face's Text Generation Inference (TGI) framework.

    Enables text generation using HuggingFace Hub hosted chat-based LLMs. This component is designed to seamlessly
    inference chat-based models deployed on the Text Generation Inference (TGI) backend.

    You can use this component for chat LLMs hosted on Hugging Face inference endpoints, the rate-limited
    Inference API tier.

    Key Features and Compatibility:
     - Primary Compatibility: designed to work seamlessly with any chat-based model deployed using the TGI
       framework. For more information on TGI, visit [text-generation-inference](https://github.com/huggingface/text-generation-inference)
    - Hugging Face Inference Endpoints: Supports inference of TGI chat LLMs deployed on Hugging Face
       inference endpoints. For more details, refer to [inference-endpoints](https://huggingface.co/inference-endpoints)

    - Inference API Support: supports inference of TGI chat LLMs hosted on the rate-limited Inference
      API tier. Learn more about the Inference API at [inference-api](https://huggingface.co/inference-api).
      Discover available chat models using the following command: `wget -qO- https://api-inference.huggingface.co/framework/text-generation-inference | grep chat`
      and simply use the model ID as the model parameter for this component. You'll also need to provide a valid
      Hugging Face API token as the token parameter.

    - Custom TGI Endpoints: supports inference of TGI chat LLMs deployed on custom TGI endpoints. Anyone can
      deploy their own TGI endpoint using the TGI framework. For more details, refer to [inference-endpoints](https://huggingface.co/inference-endpoints)

    Input and Output Format:
      - ChatMessage Format: This component uses the ChatMessage format to structure both input and output,
        ensuring coherent and contextually relevant responses in chat-based text generation scenarios. Details on the
        ChatMessage format can be found [here](https://docs.haystack.deepset.ai/v2.0/docs/data-classes#chatmessage).


    ```python
    from haystack.components.generators.chat import HuggingFaceTGIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.utils import Secret

    messages = [ChatMessage.from_system("\\nYou are a helpful, respectful and honest assistant"),
                ChatMessage.from_user("What's Natural Language Processing?")]


    client = HuggingFaceTGIChatGenerator(model="HuggingFaceH4/zephyr-7b-beta", token=Secret.from_token("<your-api-key>"))
    client.warm_up()
    response = client.run(messages, generation_kwargs={"max_new_tokens": 120})
    print(response)
    ```

    For chat LLMs hosted on paid https://huggingface.co/inference-endpoints endpoint and/or your own custom TGI
    endpoint, you'll need to provide the URL of the endpoint as well as a valid token:

    ```python
    from haystack.components.generators.chat import HuggingFaceTGIChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_system("\\nYou are a helpful, respectful and honest assistant"),
                ChatMessage.from_user("What's Natural Language Processing?")]

    client = HuggingFaceTGIChatGenerator(model="HuggingFaceH4/zephyr-7b-beta",
                                         url="<your-tgi-endpoint-url>",
                                         token=Secret.from_token("<your-api-key>"))
    client.warm_up()
    response = client.run(messages, generation_kwargs={"max_new_tokens": 120})
    print(response)
    ```
    """

    def __init__(
        self,
        model: str = "HuggingFaceH4/zephyr-7b-beta",
        url: Optional[str] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        chat_template: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Initialize the HuggingFaceTGIChatGenerator instance.

        :param model: A string representing the model path or URL. Default is "HuggingFaceH4/zephyr-7b-beta".
        :param url: An optional string representing the URL of the TGI endpoint.
        :param chat_template: This optional parameter allows you to specify a Jinja template for formatting chat
            messages. While high-quality and well-supported chat models typically include their own chat templates
            accessible through their tokenizer, there are models that do not offer this feature. For such scenarios,
            or if you wish to use a custom template instead of the model's default, you can use this parameter to
            set your preferred chat template.
        :param token: The Hugging Face token for HTTP bearer authorization.
            You can find your HF token at https://huggingface.co/settings/tokens.
        :param generation_kwargs: A dictionary containing keyword arguments to customize text generation.
            Some examples: `max_new_tokens`, `temperature`, `top_k`, `top_p`,...
            See Hugging Face's [documentation](https://huggingface.co/docs/huggingface_hub/v0.18.0.rc0/en/package_reference/inference_client#huggingface_hub.inference._text_generation.TextGenerationParameters)
            for more information.
        :param stop_words: An optional list of strings representing the stop words.
        :param streaming_callback: An optional callable for handling streaming responses.
        """
        warnings.warn(
            "`HuggingFaceTGIChatGenerator` is deprecated and will be removed in Haystack 2.3.0."
            "Use `HuggingFaceAPIChatGenerator` instead.",
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
        self.chat_template = chat_template
        self.token = token
        self.generation_kwargs = generation_kwargs
        self.client = InferenceClient(url or model, token=token.resolve_value() if token else None)
        self.streaming_callback = streaming_callback
        self.tokenizer = None
        self._warmed_up: bool = False

    def warm_up(self) -> None:
        """
        Warm up the tokenizer by loading it from the model.

        If the url is not provided, check if the model is deployed on the free tier of the HF inference API.
        Load the tokenizer
        """
        if self._warmed_up:
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

        # mypy can't infer that chat_template attribute exists on the object returned by AutoTokenizer.from_pretrained
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if not chat_template and not self.chat_template:
            logger.warning(
                "The model '{model}' doesn't have a default chat_template, and no chat_template was supplied during "
                "this component's initialization. It’s possible that the model doesn't support ChatML inference "
                "format, potentially leading to unexpected behavior.",
                model=self.model,
            )

        self._warmed_up = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :return: A dictionary containing the serialized component.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            model=self.model,
            url=self.url,
            chat_template=self.chat_template,
            token=self.token.to_dict() if self.token else None,
            generation_kwargs=self.generation_kwargs,
            streaming_callback=callback_name,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceTGIChatGenerator":
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

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param generation_kwargs: Additional keyword arguments for text generation.
        :return: A list containing the generated responses as ChatMessage instances.
        """
        if not self._warmed_up:
            raise RuntimeError(
                "The component HuggingFaceTGIChatGenerator was not warmed up. Please call warm_up() before running."
            )

        # check generation kwargs given as parameters to override the default ones
        additional_params = ["n", "stop_words"]
        check_generation_params(generation_kwargs, additional_params)

        # update generation kwargs by merging with the default ones
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        num_responses = generation_kwargs.pop("n", 1)

        # merge stop_words and stop_sequences into a single list
        generation_kwargs["stop_sequences"] = generation_kwargs.get("stop_sequences", [])
        generation_kwargs["stop_sequences"].extend(generation_kwargs.pop("stop_words", []))

        if self.tokenizer is None:
            raise RuntimeError("Please call warm_up() before running LLM inference.")

        # apply either model's chat template or the user-provided one
        formatted_messages = [message.to_openai_format() for message in messages]
        prepared_prompt: str = self.tokenizer.apply_chat_template(
            conversation=formatted_messages, chat_template=self.chat_template, tokenize=False
        )
        prompt_token_count: int = len(self.tokenizer.encode(prepared_prompt, add_special_tokens=False))

        if self.streaming_callback:
            if num_responses > 1:
                raise ValueError("Cannot stream multiple responses, please set n=1.")

            return self._run_streaming(prepared_prompt, prompt_token_count, generation_kwargs)

        return self._run_non_streaming(prepared_prompt, prompt_token_count, num_responses, generation_kwargs)

    def _run_streaming(
        self, prepared_prompt: str, prompt_token_count: int, generation_kwargs: Dict[str, Any]
    ) -> Dict[str, List[ChatMessage]]:
        res: Iterable[TextGenerationStreamOutput] = self.client.text_generation(
            prepared_prompt, stream=True, details=True, **generation_kwargs
        )
        chunk = None
        # pylint: disable=not-an-iterable
        for chunk in res:
            token: TextGenerationOutputToken = chunk.token
            if token.special:
                continue
            chunk_metadata = {**asdict(token), **(asdict(chunk.details) if chunk.details else {})}
            stream_chunk = StreamingChunk(token.text, chunk_metadata)
            self.streaming_callback(stream_chunk)  # type: ignore # streaming_callback is not None (verified in the run method)

        message = ChatMessage.from_assistant(chunk.generated_text)
        message.meta.update(
            {
                "finish_reason": chunk.details.finish_reason if chunk.details else None,
                "index": 0,
                "model": self.client.model,
                "usage": {
                    "completion_tokens": chunk.details.generated_tokens if chunk.details else 0,
                    "prompt_tokens": prompt_token_count,
                    "total_tokens": prompt_token_count + chunk.details.generated_tokens if chunk.details else 0,
                },
            }
        )
        return {"replies": [message]}

    def _run_non_streaming(
        self, prepared_prompt: str, prompt_token_count: int, num_responses: int, generation_kwargs: Dict[str, Any]
    ) -> Dict[str, List[ChatMessage]]:
        chat_messages: List[ChatMessage] = []
        for _i in range(num_responses):
            tgr: TextGenerationOutput = self.client.text_generation(prepared_prompt, details=True, **generation_kwargs)
            message = ChatMessage.from_assistant(tgr.generated_text)
            if tgr.details:
                completion_tokens = len(tgr.details.tokens)
                prompt_token_count = prompt_token_count + completion_tokens
                finish_reason = tgr.details.finish_reason
            else:
                finish_reason = None
                completion_tokens = 0
            message.meta.update(
                {
                    "finish_reason": finish_reason,
                    "index": _i,
                    "model": self.client.model,
                    "usage": {
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": prompt_token_count,
                        "total_tokens": prompt_token_count + completion_tokens,
                    },
                }
            )
            chat_messages.append(message)
        return {"replies": chat_messages}
