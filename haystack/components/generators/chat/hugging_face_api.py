# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable
from haystack.utils.hf import HFGenerationAPIType, HFModelType, check_valid_model
from haystack.utils.url_validation import is_valid_http_url

with LazyImport(message="Run 'pip install \"huggingface_hub[inference]>=0.23.0\"'") as huggingface_hub_import:
    from huggingface_hub import ChatCompletionOutput, ChatCompletionStreamOutput, InferenceClient


logger = logging.getLogger(__name__)


def _convert_message_to_hfapi_format(message: ChatMessage) -> Dict[str, str]:
    """
    Convert a message to the format expected by Hugging Face APIs.

    :returns: A dictionary with the following keys:
        - `role`
        - `content`
        - `name` (optional)
    """
    formatted_msg = {"role": message.role.value, "content": message.content}
    if message.name:
        formatted_msg["name"] = message.name

    return formatted_msg


@component
class HuggingFaceAPIChatGenerator:
    """
    Completes chats using Hugging Face APIs.

    HuggingFaceAPIChatGenerator uses the [ChatMessage](https://docs.haystack.deepset.ai/docs/data-classes#chatmessage)
    format for input and output. Use it to generate text with Hugging Face APIs:
    - [Free Serverless Inference API](https://huggingface.co/inference-api)
    - [Paid Inference Endpoints](https://huggingface.co/inference-endpoints)
    - [Self-hosted Text Generation Inference](https://github.com/huggingface/text-generation-inference)

    ### Usage examples

    #### With the free serverless inference API

    ```python
    from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.utils import Secret
    from haystack.utils.hf import HFGenerationAPIType

    messages = [ChatMessage.from_system("\\nYou are a helpful, respectful and honest assistant"),
                ChatMessage.from_user("What's Natural Language Processing?")]

    # the api_type can be expressed using the HFGenerationAPIType enum or as a string
    api_type = HFGenerationAPIType.SERVERLESS_INFERENCE_API
    api_type = "serverless_inference_api" # this is equivalent to the above

    generator = HuggingFaceAPIChatGenerator(api_type=api_type,
                                            api_params={"model": "HuggingFaceH4/zephyr-7b-beta"},
                                            token=Secret.from_token("<your-api-key>"))

    result = generator.run(messages)
    print(result)
    ```

    #### With paid inference endpoints

    ```python
    from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
    from haystack.dataclasses import ChatMessage
    from haystack.utils import Secret

    messages = [ChatMessage.from_system("\\nYou are a helpful, respectful and honest assistant"),
                ChatMessage.from_user("What's Natural Language Processing?")]

    generator = HuggingFaceAPIChatGenerator(api_type="inference_endpoints",
                                            api_params={"url": "<your-inference-endpoint-url>"},
                                            token=Secret.from_token("<your-api-key>"))

    result = generator.run(messages)
    print(result)

    #### With self-hosted text generation inference

    ```python
    from haystack.components.generators.chat import HuggingFaceAPIChatGenerator
    from haystack.dataclasses import ChatMessage

    messages = [ChatMessage.from_system("\\nYou are a helpful, respectful and honest assistant"),
                ChatMessage.from_user("What's Natural Language Processing?")]

    generator = HuggingFaceAPIChatGenerator(api_type="text_generation_inference",
                                            api_params={"url": "http://localhost:8080"})

    result = generator.run(messages)
    print(result)
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        api_type: Union[HFGenerationAPIType, str],
        api_params: Dict[str, str],
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        generation_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Initialize the HuggingFaceAPIChatGenerator instance.

        :param api_type:
            The type of Hugging Face API to use. Available types:
            - `text_generation_inference`: See [TGI](https://github.com/huggingface/text-generation-inference).
            - `inference_endpoints`: See [Inference Endpoints](https://huggingface.co/inference-endpoints).
            - `serverless_inference_api`: See [Serverless Inference API](https://huggingface.co/inference-api).
        :param api_params:
            A dictionary with the following keys:
            - `model`: Hugging Face model ID. Required when `api_type` is `SERVERLESS_INFERENCE_API`.
            - `url`: URL of the inference endpoint. Required when `api_type` is `INFERENCE_ENDPOINTS` or
            `TEXT_GENERATION_INFERENCE`.
        :param token: The Hugging Face token to use as HTTP bearer authorization.
            Check your HF token in your [account settings](https://huggingface.co/settings/tokens).
        :param generation_kwargs:
            A dictionary with keyword arguments to customize text generation.
                Some examples: `max_tokens`, `temperature`, `top_p`.
                For details, see [Hugging Face chat_completion documentation](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client#huggingface_hub.InferenceClient.chat_completion).
        :param stop_words: An optional list of strings representing the stop words.
        :param streaming_callback: An optional callable for handling streaming responses.
        """

        huggingface_hub_import.check()

        if isinstance(api_type, str):
            api_type = HFGenerationAPIType.from_str(api_type)

        if api_type == HFGenerationAPIType.SERVERLESS_INFERENCE_API:
            model = api_params.get("model")
            if model is None:
                raise ValueError(
                    "To use the Serverless Inference API, you need to specify the `model` parameter in `api_params`."
                )
            check_valid_model(model, HFModelType.GENERATION, token)
            model_or_url = model
        elif api_type in [HFGenerationAPIType.INFERENCE_ENDPOINTS, HFGenerationAPIType.TEXT_GENERATION_INFERENCE]:
            url = api_params.get("url")
            if url is None:
                msg = (
                    "To use Text Generation Inference or Inference Endpoints, you need to specify the `url` parameter "
                    "in `api_params`."
                )
                raise ValueError(msg)
            if not is_valid_http_url(url):
                raise ValueError(f"Invalid URL: {url}")
            model_or_url = url
        else:
            msg = f"Unknown api_type {api_type}"
            raise ValueError(msg)

        # handle generation kwargs setup
        generation_kwargs = generation_kwargs.copy() if generation_kwargs else {}
        generation_kwargs["stop"] = generation_kwargs.get("stop", [])
        generation_kwargs["stop"].extend(stop_words or [])
        generation_kwargs.setdefault("max_tokens", 512)

        self.api_type = api_type
        self.api_params = api_params
        self.token = token
        self.generation_kwargs = generation_kwargs
        self.streaming_callback = streaming_callback
        self._client = InferenceClient(model_or_url, token=token.resolve_value() if token else None)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            A dictionary containing the serialized component.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        return default_to_dict(
            self,
            api_type=str(self.api_type),
            api_params=self.api_params,
            token=self.token.to_dict() if self.token else None,
            generation_kwargs=self.generation_kwargs,
            streaming_callback=callback_name,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceAPIChatGenerator":
        """
        Deserialize this component from a dictionary.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param messages: A list of ChatMessage objects representing the input messages.
        :param generation_kwargs: Additional keyword arguments for text generation.
        :returns: A dictionary with the following keys:
            - `replies`: A list containing the generated responses as ChatMessage objects.
        """

        # update generation kwargs by merging with the default ones
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        formatted_messages = [_convert_message_to_hfapi_format(message) for message in messages]

        if self.streaming_callback:
            return self._run_streaming(formatted_messages, generation_kwargs)

        return self._run_non_streaming(formatted_messages, generation_kwargs)

    def _run_streaming(self, messages: List[Dict[str, str]], generation_kwargs: Dict[str, Any]):
        api_output: Iterable[ChatCompletionStreamOutput] = self._client.chat_completion(
            messages, stream=True, **generation_kwargs
        )

        generated_text = ""

        for chunk in api_output:  # pylint: disable=not-an-iterable
            text = chunk.choices[0].delta.content
            if text:
                generated_text += text
            finish_reason = chunk.choices[0].finish_reason

            meta = {}
            if finish_reason:
                meta["finish_reason"] = finish_reason

            stream_chunk = StreamingChunk(text, meta)
            self.streaming_callback(stream_chunk)  # type: ignore # streaming_callback is not None (verified in the run method)

        message = ChatMessage.from_assistant(generated_text)
        message.meta.update(
            {
                "model": self._client.model,
                "finish_reason": finish_reason,
                "index": 0,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},  # not available in streaming
            }
        )
        return {"replies": [message]}

    def _run_non_streaming(
        self, messages: List[Dict[str, str]], generation_kwargs: Dict[str, Any]
    ) -> Dict[str, List[ChatMessage]]:
        chat_messages: List[ChatMessage] = []

        api_chat_output: ChatCompletionOutput = self._client.chat_completion(messages, **generation_kwargs)
        for choice in api_chat_output.choices:
            message = ChatMessage.from_assistant(choice.message.content)
            message.meta.update(
                {
                    "model": self._client.model,
                    "finish_reason": choice.finish_reason,
                    "index": choice.index,
                    "usage": api_chat_output.usage or {"prompt_tokens": 0, "completion_tokens": 0},
                }
            )
            chat_messages.append(message)

        return {"replies": chat_messages}
