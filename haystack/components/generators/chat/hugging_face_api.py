# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Iterable, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall
from haystack.lazy_imports import LazyImport
from haystack.tools.tool import Tool, _check_duplicate_tool_names, deserialize_tools_inplace
from haystack.utils import Secret, deserialize_callable, deserialize_secrets_inplace, serialize_callable
from haystack.utils.hf import HFGenerationAPIType, HFModelType, check_valid_model, convert_message_to_hf_format
from haystack.utils.url_validation import is_valid_http_url

with LazyImport(message="Run 'pip install \"huggingface_hub[inference]>=0.27.0\"'") as huggingface_hub_import:
    from huggingface_hub import (
        ChatCompletionInputTool,
        ChatCompletionOutput,
        ChatCompletionStreamOutput,
        InferenceClient,
    )


logger = logging.getLogger(__name__)


@component
class HuggingFaceAPIChatGenerator:
    """
    Completes chats using Hugging Face APIs.

    HuggingFaceAPIChatGenerator uses the [ChatMessage](https://docs.haystack.deepset.ai/docs/chatmessage)
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
        tools: Optional[List[Tool]] = None,
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
        :param token:
            The Hugging Face token to use as HTTP bearer authorization.
            Check your HF token in your [account settings](https://huggingface.co/settings/tokens).
        :param generation_kwargs:
            A dictionary with keyword arguments to customize text generation.
                Some examples: `max_tokens`, `temperature`, `top_p`.
                For details, see [Hugging Face chat_completion documentation](https://huggingface.co/docs/huggingface_hub/package_reference/inference_client#huggingface_hub.InferenceClient.chat_completion).
        :param stop_words:
            An optional list of strings representing the stop words.
        :param streaming_callback:
            An optional callable for handling streaming responses.
        :param tools:
            A list of tools for which the model can prepare calls.
            The chosen model should support tool/function calling, according to the model card.
            Support for tools in the Hugging Face API and TGI is not yet fully refined and you may experience
            unexpected behavior.
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

        if tools and streaming_callback is not None:
            raise ValueError("Using tools and streaming at the same time is not supported. Please choose one.")
        _check_duplicate_tool_names(tools)

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
        self.tools = tools

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.

        :returns:
            A dictionary containing the serialized component.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        serialized_tools = [tool.to_dict() for tool in self.tools] if self.tools else None
        return default_to_dict(
            self,
            api_type=str(self.api_type),
            api_params=self.api_params,
            token=self.token.to_dict() if self.token else None,
            generation_kwargs=self.generation_kwargs,
            streaming_callback=callback_name,
            tools=serialized_tools,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceAPIChatGenerator":
        """
        Deserialize this component from a dictionary.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        deserialize_tools_inplace(data["init_parameters"], key="tools")
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)
        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Tool]] = None,
    ):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param messages:
            A list of ChatMessage objects representing the input messages.
        :param generation_kwargs:
            Additional keyword arguments for text generation.
        :param tools:
            A list of tools for which the model can prepare calls. If set, it will override the `tools` parameter set
            during component initialization.
        :returns: A dictionary with the following keys:
            - `replies`: A list containing the generated responses as ChatMessage objects.
        """

        # update generation kwargs by merging with the default ones
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        formatted_messages = [convert_message_to_hf_format(message) for message in messages]

        tools = tools or self.tools
        if tools and self.streaming_callback:
            raise ValueError("Using tools and streaming at the same time is not supported. Please choose one.")
        _check_duplicate_tool_names(tools)

        if self.streaming_callback:
            return self._run_streaming(formatted_messages, generation_kwargs)

        hf_tools = None
        if tools:
            hf_tools = [{"type": "function", "function": {**t.tool_spec}} for t in tools]

        return self._run_non_streaming(formatted_messages, generation_kwargs, hf_tools)

    def _run_streaming(self, messages: List[Dict[str, str]], generation_kwargs: Dict[str, Any]):
        api_output: Iterable[ChatCompletionStreamOutput] = self._client.chat_completion(
            messages, stream=True, **generation_kwargs
        )

        generated_text = ""

        for chunk in api_output:
            # n is unused, so the API always returns only one choice
            # the argument is probably allowed for compatibility with OpenAI
            # see https://huggingface.co/docs/huggingface_hub/package_reference/inference_client#huggingface_hub.InferenceClient.chat_completion.n
            choice = chunk.choices[0]

            text = choice.delta.content
            if text:
                generated_text += text

            finish_reason = choice.finish_reason

            meta = {}
            if finish_reason:
                meta["finish_reason"] = finish_reason

            stream_chunk = StreamingChunk(text, meta)
            self.streaming_callback(stream_chunk)  # type: ignore # streaming_callback is not None (verified in the run method)

        meta.update(
            {
                "model": self._client.model,
                "finish_reason": finish_reason,
                "index": 0,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},  # not available in streaming
            }
        )

        message = ChatMessage.from_assistant(text=generated_text, meta=meta)

        return {"replies": [message]}

    def _run_non_streaming(
        self,
        messages: List[Dict[str, str]],
        generation_kwargs: Dict[str, Any],
        tools: Optional[List["ChatCompletionInputTool"]] = None,
    ) -> Dict[str, List[ChatMessage]]:
        api_chat_output: ChatCompletionOutput = self._client.chat_completion(
            messages=messages, tools=tools, **generation_kwargs
        )

        if len(api_chat_output.choices) == 0:
            return {"replies": []}

        # n is unused, so the API always returns only one choice
        # the argument is probably allowed for compatibility with OpenAI
        # see https://huggingface.co/docs/huggingface_hub/package_reference/inference_client#huggingface_hub.InferenceClient.chat_completion.n
        choice = api_chat_output.choices[0]

        text = choice.message.content
        tool_calls = []

        if hfapi_tool_calls := choice.message.tool_calls:
            for hfapi_tc in hfapi_tool_calls:
                tool_call = ToolCall(
                    tool_name=hfapi_tc.function.name, arguments=hfapi_tc.function.arguments, id=hfapi_tc.id
                )
                tool_calls.append(tool_call)

        meta = {"model": self._client.model, "finish_reason": choice.finish_reason, "index": choice.index}

        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        if api_chat_output.usage:
            usage = {
                "prompt_tokens": api_chat_output.usage.prompt_tokens,
                "completion_tokens": api_chat_output.usage.completion_tokens,
            }
        meta["usage"] = usage

        message = ChatMessage.from_assistant(text=text, tool_calls=tool_calls, meta=meta)
        return {"replies": [message]}
