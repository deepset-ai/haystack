# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Literal, Optional, Union, cast

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall, select_streaming_callback
from haystack.lazy_imports import LazyImport
from haystack.tools import (
    Tool,
    Toolset,
    _check_duplicate_tool_names,
    deserialize_tools_or_toolset_inplace,
    serialize_tools_or_toolset,
)
from haystack.utils import (
    ComponentDevice,
    Secret,
    deserialize_callable,
    deserialize_secrets_inplace,
    serialize_callable,
)

logger = logging.getLogger(__name__)

with LazyImport(message="Run 'pip install \"transformers[torch]\"'") as torch_and_transformers_import:
    from huggingface_hub import model_info
    from transformers import StoppingCriteriaList, pipeline
    from transformers.tokenization_utils import PreTrainedTokenizer
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

    from haystack.utils.hf import (  # pylint: disable=ungrouped-imports
        HFTokenStreamingHandler,
        StopWordsCriteria,
        convert_message_to_hf_format,
        deserialize_hf_model_kwargs,
        serialize_hf_model_kwargs,
    )


PIPELINE_SUPPORTED_TASKS = ["text-generation", "text2text-generation"]

DEFAULT_TOOL_PATTERN = (
    r"(?:<tool_call>)?"
    r'(?:\s*\{.*?"name"\s*:\s*"([^"]+)".*?"arguments"\s*:\s*(\{[^}]+\}).*?\}'
    r'|\{.*?"function"\s*:\s*\{.*?"name"\s*:\s*"([^"]+)".*?"arguments"\s*:\s*(\{[^}]+\}).*?\})'
)


def default_tool_parser(text: str) -> Optional[List[ToolCall]]:
    """
    Default implementation for parsing tool calls from model output text.

    Uses DEFAULT_TOOL_PATTERN to extract tool calls.

    :param text: The text to parse for tool calls.
    :returns: A list containing a single ToolCall if a valid tool call is found, None otherwise.
    """
    try:
        match = re.search(DEFAULT_TOOL_PATTERN, text, re.DOTALL)
    except re.error:
        logger.warning("Invalid regex pattern for tool parsing: {pattern}", pattern=DEFAULT_TOOL_PATTERN)
        return None

    if not match:
        return None

    name = match.group(1) or match.group(3)
    args_str = match.group(2) or match.group(4)

    try:
        arguments = json.loads(args_str)
        return [ToolCall(tool_name=name, arguments=arguments)]
    except json.JSONDecodeError:
        logger.warning("Failed to parse tool call arguments: {args_str}", args_str=args_str)
        return None


@component
class HuggingFaceLocalChatGenerator:
    """
    Generates chat responses using models from Hugging Face that run locally.

    Use this component with chat-based models,
    such as `HuggingFaceH4/zephyr-7b-beta` or `meta-llama/Llama-2-7b-chat-hf`.
    LLMs running locally may need powerful hardware.

    ### Usage example

    ```python
    from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
    from haystack.dataclasses import ChatMessage

    generator = HuggingFaceLocalChatGenerator(model="HuggingFaceH4/zephyr-7b-beta")
    generator.warm_up()
    messages = [ChatMessage.from_user("What's Natural Language Processing? Be brief.")]
    print(generator.run(messages))
    ```

    ```
    {'replies':
        [ChatMessage(_role=<ChatRole.ASSISTANT: 'assistant'>, _content=[TextContent(text=
        "Natural Language Processing (NLP) is a subfield of artificial intelligence that deals
        with the interaction between computers and human language. It enables computers to understand, interpret, and
        generate human language in a valuable way. NLP involves various techniques such as speech recognition, text
        analysis, sentiment analysis, and machine translation. The ultimate goal is to make it easier for computers to
        process and derive meaning from human language, improving communication between humans and machines.")],
        _name=None,
        _meta={'finish_reason': 'stop', 'index': 0, 'model':
              'mistralai/Mistral-7B-Instruct-v0.2',
              'usage': {'completion_tokens': 90, 'prompt_tokens': 19, 'total_tokens': 109}})
              ]
    }
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        model: str = "HuggingFaceH4/zephyr-7b-beta",
        task: Optional[Literal["text-generation", "text2text-generation"]] = None,
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        chat_template: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        huggingface_pipeline_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        tools: Optional[Union[List[Tool], Toolset]] = None,
        tool_parsing_function: Optional[Callable[[str], Optional[List[ToolCall]]]] = None,
        async_executor: Optional[ThreadPoolExecutor] = None,
    ):
        """
        Initializes the HuggingFaceLocalChatGenerator component.

        :param model: The Hugging Face text generation model name or path,
            for example, `mistralai/Mistral-7B-Instruct-v0.2` or `TheBloke/OpenHermes-2.5-Mistral-7B-16k-AWQ`.
            The model must be a chat model supporting the ChatML messaging
            format.
            If the model is specified in `huggingface_pipeline_kwargs`, this parameter is ignored.
        :param task: The task for the Hugging Face pipeline. Possible options:
            - `text-generation`: Supported by decoder models, like GPT.
            - `text2text-generation`: Supported by encoder-decoder models, like T5.
            If the task is specified in `huggingface_pipeline_kwargs`, this parameter is ignored.
            If not specified, the component calls the Hugging Face API to infer the task from the model name.
        :param device: The device for loading the model. If `None`, automatically selects the default device.
            If a device or device map is specified in `huggingface_pipeline_kwargs`, it overrides this parameter.
        :param token: The token to use as HTTP bearer authorization for remote files.
            If the token is specified in `huggingface_pipeline_kwargs`, this parameter is ignored.
        :param chat_template: Specifies an optional Jinja template for formatting chat
            messages. Most high-quality chat models have their own templates, but for models without this
            feature or if you prefer a custom template, use this parameter.
        :param generation_kwargs: A dictionary with keyword arguments to customize text generation.
            Some examples: `max_length`, `max_new_tokens`, `temperature`, `top_k`, `top_p`.
            See Hugging Face's documentation for more information:
            - - [customize-text-generation](https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation)
            - - [GenerationConfig](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig)
            The only `generation_kwargs` set by default is `max_new_tokens`, which is set to 512 tokens.
        :param huggingface_pipeline_kwargs: Dictionary with keyword arguments to initialize the
            Hugging Face pipeline for text generation.
            These keyword arguments provide fine-grained control over the Hugging Face pipeline.
            In case of duplication, these kwargs override `model`, `task`, `device`, and `token` init parameters.
            For kwargs, see [Hugging Face documentation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline.task).
            In this dictionary, you can also include `model_kwargs` to specify the kwargs for [model initialization](https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)
        :param stop_words: A list of stop words. If the model generates a stop word, the generation stops.
            If you provide this parameter, don't specify the `stopping_criteria` in `generation_kwargs`.
            For some chat models, the output includes both the new text and the original prompt.
            In these cases, make sure your prompt has no stop words.
        :param streaming_callback: An optional callable for handling streaming responses.
        :param tools: A list of tools or a Toolset for which the model can prepare calls.
            This parameter can accept either a list of `Tool` objects or a `Toolset` instance.
        :param tool_parsing_function:
            A callable that takes a string and returns a list of ToolCall objects or None.
            If None, the default_tool_parser will be used which extracts tool calls using a predefined pattern.
        :param async_executor:
            Optional ThreadPoolExecutor to use for async calls. If not provided, a single-threaded executor will be
            initialized and used
        """
        torch_and_transformers_import.check()

        if tools and streaming_callback is not None:
            raise ValueError("Using tools and streaming at the same time is not supported. Please choose one.")
        _check_duplicate_tool_names(list(tools or []))

        huggingface_pipeline_kwargs = huggingface_pipeline_kwargs or {}
        generation_kwargs = generation_kwargs or {}

        self.token = token
        token = token.resolve_value() if token else None

        # check if the huggingface_pipeline_kwargs contain the essential parameters
        # otherwise, populate them with values from other init parameters
        huggingface_pipeline_kwargs.setdefault("model", model)
        huggingface_pipeline_kwargs.setdefault("token", token)

        device = ComponentDevice.resolve_device(device)
        device.update_hf_kwargs(huggingface_pipeline_kwargs, overwrite=False)

        # task identification and validation
        if task is None:
            if "task" in huggingface_pipeline_kwargs:
                task = huggingface_pipeline_kwargs["task"]
            elif isinstance(huggingface_pipeline_kwargs["model"], str):
                task = model_info(
                    huggingface_pipeline_kwargs["model"], token=huggingface_pipeline_kwargs["token"]
                ).pipeline_tag  # type: ignore[assignment]  # we'll check below if task is in supported tasks

        if task not in PIPELINE_SUPPORTED_TASKS:
            raise ValueError(
                f"Task '{task}' is not supported. The supported tasks are: {', '.join(PIPELINE_SUPPORTED_TASKS)}."
            )
        huggingface_pipeline_kwargs["task"] = task

        # if not specified, set return_full_text to False for text-generation
        # only generated text is returned (excluding prompt)
        if task == "text-generation":
            generation_kwargs.setdefault("return_full_text", False)

        if stop_words and "stopping_criteria" in generation_kwargs:
            raise ValueError(
                "Found both the `stop_words` init parameter and the `stopping_criteria` key in `generation_kwargs`. "
                "Please specify only one of them."
            )
        generation_kwargs.setdefault("max_new_tokens", 512)
        generation_kwargs["stop_sequences"] = generation_kwargs.get("stop_sequences", [])
        generation_kwargs["stop_sequences"].extend(stop_words or [])

        self.tool_parsing_function = tool_parsing_function or default_tool_parser
        self.huggingface_pipeline_kwargs = huggingface_pipeline_kwargs
        self.generation_kwargs = generation_kwargs
        self.chat_template = chat_template
        self.streaming_callback = streaming_callback
        self.pipeline = None
        self.tools = tools

        self._owns_executor = async_executor is None
        self.executor = (
            ThreadPoolExecutor(thread_name_prefix=f"async-HFLocalChatGenerator-executor-{id(self)}", max_workers=1)
            if async_executor is None
            else async_executor
        )

    def __del__(self):
        """
        Cleanup when the instance is being destroyed.
        """
        if hasattr(self, "_owns_executor") and self._owns_executor and hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

    def shutdown(self):
        """
        Explicitly shutdown the executor if we own it.
        """
        if self._owns_executor:
            self.executor.shutdown(wait=True)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        if isinstance(self.huggingface_pipeline_kwargs["model"], str):
            return {"model": self.huggingface_pipeline_kwargs["model"]}
        return {"model": f"[object of type {type(self.huggingface_pipeline_kwargs['model'])}]"}

    def warm_up(self):
        """
        Initializes the component.
        """
        if self.pipeline is None:
            self.pipeline = pipeline(**self.huggingface_pipeline_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        callback_name = serialize_callable(self.streaming_callback) if self.streaming_callback else None
        serialization_dict = default_to_dict(
            self,
            huggingface_pipeline_kwargs=self.huggingface_pipeline_kwargs,
            generation_kwargs=self.generation_kwargs,
            streaming_callback=callback_name,
            token=self.token.to_dict() if self.token else None,
            chat_template=self.chat_template,
            tools=serialize_tools_or_toolset(self.tools),
            tool_parsing_function=serialize_callable(self.tool_parsing_function),
        )

        huggingface_pipeline_kwargs = serialization_dict["init_parameters"]["huggingface_pipeline_kwargs"]
        huggingface_pipeline_kwargs.pop("token", None)

        serialize_hf_model_kwargs(huggingface_pipeline_kwargs)
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceLocalChatGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        torch_and_transformers_import.check()  # leave this, cls method
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        deserialize_tools_or_toolset_inplace(data["init_parameters"], key="tools")
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)

        tool_parsing_function = init_params.get("tool_parsing_function")
        if tool_parsing_function:
            init_params["tool_parsing_function"] = deserialize_callable(tool_parsing_function)

        huggingface_pipeline_kwargs = init_params.get("huggingface_pipeline_kwargs", {})
        deserialize_hf_model_kwargs(huggingface_pipeline_kwargs)
        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        tools: Optional[Union[List[Tool], Toolset]] = None,
    ):
        """
        Invoke text generation inference based on the provided messages and generation parameters.

        :param messages: A list of ChatMessage objects representing the input messages.
        :param generation_kwargs: Additional keyword arguments for text generation.
        :param streaming_callback: An optional callable for handling streaming responses.
        :param tools:
            A list of tools or a Toolset for which the model can prepare calls. If set, it will override
            the `tools` parameter provided during initialization. This parameter can accept either a list
            of `Tool` objects or a `Toolset` instance.
        :returns:
            A list containing the generated responses as ChatMessage instances.
        """
        if self.pipeline is None:
            raise RuntimeError("The generation model has not been loaded. Please call warm_up() before running.")

        tools = tools or self.tools
        if tools and streaming_callback is not None:
            raise ValueError("Using tools and streaming at the same time is not supported. Please choose one.")
        _check_duplicate_tool_names(tools)

        tokenizer = self.pipeline.tokenizer

        # Check and update generation parameters
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        stop_words = generation_kwargs.pop("stop_words", []) + generation_kwargs.pop("stop_sequences", [])
        # pipeline call doesn't support stop_sequences, so we need to pop it
        stop_words = self._validate_stop_words(stop_words)

        # Set up stop words criteria if stop words exist
        stop_words_criteria = StopWordsCriteria(tokenizer, stop_words, self.pipeline.device) if stop_words else None
        if stop_words_criteria:
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([stop_words_criteria])

        streaming_callback = select_streaming_callback(
            self.streaming_callback, streaming_callback, requires_async=False
        )
        if streaming_callback:
            num_responses = generation_kwargs.get("num_return_sequences", 1)
            if num_responses > 1:
                msg = (
                    "Streaming is enabled, but the number of responses is set to {num_responses}. "
                    "Streaming is only supported for single response generation. "
                    "Setting the number of responses to 1."
                )
                logger.warning(msg, num_responses=num_responses)
                generation_kwargs["num_return_sequences"] = 1
            # streamer parameter hooks into HF streaming, HFTokenStreamingHandler is an adapter to our streaming
            generation_kwargs["streamer"] = HFTokenStreamingHandler(tokenizer, streaming_callback, stop_words)

        # convert messages to HF format
        hf_messages = [convert_message_to_hf_format(message) for message in messages]

        if isinstance(tools, Toolset):
            tools = list(tools)

        prepared_prompt = tokenizer.apply_chat_template(
            hf_messages,
            tokenize=False,
            chat_template=self.chat_template,
            add_generation_prompt=True,
            tools=[tc.tool_spec for tc in tools] if tools else None,
        )

        # Avoid some unnecessary warnings in the generation pipeline call
        generation_kwargs["pad_token_id"] = (
            generation_kwargs.get("pad_token_id", tokenizer.pad_token_id) or tokenizer.eos_token_id
        )

        # Generate responses
        output = self.pipeline(prepared_prompt, **generation_kwargs)
        replies = [o.get("generated_text", "") for o in output]

        # Remove stop words from replies if present
        for stop_word in stop_words or []:
            replies = [reply.replace(stop_word, "").rstrip() for reply in replies]

        chat_messages = [
            self.create_message(
                reply, r_index, tokenizer, prepared_prompt, generation_kwargs, parse_tool_calls=bool(tools)
            )
            for r_index, reply in enumerate(replies)
        ]

        return {"replies": chat_messages}

    def create_message(  # pylint: disable=too-many-positional-arguments
        self,
        text: str,
        index: int,
        tokenizer: Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"],
        prompt: str,
        generation_kwargs: Dict[str, Any],
        parse_tool_calls: bool = False,
    ) -> ChatMessage:
        """
        Create a ChatMessage instance from the provided text, populated with metadata.

        :param text: The generated text.
        :param index: The index of the generated text.
        :param tokenizer: The tokenizer used for generation.
        :param prompt: The prompt used for generation.
        :param generation_kwargs: The generation parameters.
        :param parse_tool_calls: Whether to attempt parsing tool calls from the text.
        :returns: A ChatMessage instance.
        """

        completion_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        prompt_token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        total_tokens = prompt_token_count + completion_tokens

        tool_calls = self.tool_parsing_function(text) if parse_tool_calls else None

        # Determine finish reason based on context
        if completion_tokens >= generation_kwargs.get("max_new_tokens", sys.maxsize):
            finish_reason = "length"
        elif tool_calls:
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"

        meta = {
            "finish_reason": finish_reason,
            "index": index,
            "model": self.huggingface_pipeline_kwargs["model"],
            "usage": {
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_token_count,
                "total_tokens": total_tokens,
            },
        }

        # If tool calls are detected, don't include the text content since it contains the raw tool call format
        return ChatMessage.from_assistant(tool_calls=tool_calls, text=None if tool_calls else text, meta=meta)

    @staticmethod
    def _validate_stop_words(stop_words: Optional[List[str]]) -> Optional[List[str]]:
        """
        Validates the provided stop words.

        :param stop_words: A list of stop words to validate.
        :return: A sanitized list of stop words or None if validation fails.
        """
        if stop_words and not all(isinstance(word, str) for word in stop_words):
            logger.warning(
                "Invalid stop words provided. Stop words must be specified as a list of strings. "
                "Ignoring stop words: {stop_words}",
                stop_words=stop_words,
            )
            return None

        return list(set(stop_words or []))

    @component.output_types(replies=List[ChatMessage])
    async def run_async(
        self,
        messages: List[ChatMessage],
        generation_kwargs: Optional[Dict[str, Any]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        tools: Optional[Union[List[Tool], Toolset]] = None,
    ):
        """
        Asynchronously invokes text generation inference based on the provided messages and generation parameters.

        This is the asynchronous version of the `run` method. It has the same parameters
        and return values but can be used with `await` in an async code.

        :param messages: A list of ChatMessage objects representing the input messages.
        :param generation_kwargs: Additional keyword arguments for text generation.
        :param streaming_callback: An optional callable for handling streaming responses.
        :param tools: A list of tools or a Toolset for which the model can prepare calls.
            This parameter can accept either a list of `Tool` objects or a `Toolset` instance.
        :returns: A dictionary with the following keys:
            - `replies`: A list containing the generated responses as ChatMessage instances.
        """
        if self.pipeline is None:
            raise RuntimeError("The generation model has not been loaded. Please call warm_up() before running.")

        tools = tools or self.tools
        if tools and streaming_callback is not None:
            raise ValueError("Using tools and streaming at the same time is not supported. Please choose one.")
        _check_duplicate_tool_names(tools)

        tokenizer = self.pipeline.tokenizer

        # Check and update generation parameters
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        stop_words = generation_kwargs.pop("stop_words", []) + generation_kwargs.pop("stop_sequences", [])
        stop_words = self._validate_stop_words(stop_words)

        # Set up stop words criteria if stop words exist
        stop_words_criteria = StopWordsCriteria(tokenizer, stop_words, self.pipeline.device) if stop_words else None
        if stop_words_criteria:
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([stop_words_criteria])

        # validate and select the streaming callback
        streaming_callback = select_streaming_callback(self.streaming_callback, streaming_callback, requires_async=True)

        if streaming_callback:
            return await self._run_streaming_async(
                messages, tokenizer, generation_kwargs, stop_words, streaming_callback
            )

        return await self._run_non_streaming_async(messages, tokenizer, generation_kwargs, stop_words, tools)

    async def _run_streaming_async(  # pylint: disable=too-many-positional-arguments
        self,
        messages: List[ChatMessage],
        tokenizer: Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"],
        generation_kwargs: Dict[str, Any],
        stop_words: Optional[List[str]],
        streaming_callback: Callable[[StreamingChunk], None],
    ):
        """
        Handles async streaming generation of responses.
        """
        # convert messages to HF format
        hf_messages = [convert_message_to_hf_format(message) for message in messages]
        prepared_prompt = tokenizer.apply_chat_template(
            hf_messages, tokenize=False, chat_template=self.chat_template, add_generation_prompt=True
        )

        # prepared_prompt is a string, but transformers has some type issues
        prepared_prompt = cast(str, prepared_prompt)

        # Avoid some unnecessary warnings in the generation pipeline call
        generation_kwargs["pad_token_id"] = (
            generation_kwargs.get("pad_token_id", tokenizer.pad_token_id) or tokenizer.eos_token_id
        )

        # Set up streaming handler
        generation_kwargs["streamer"] = HFTokenStreamingHandler(tokenizer, streaming_callback, stop_words)

        # Generate responses asynchronously
        output = await asyncio.get_running_loop().run_in_executor(
            self.executor,
            lambda: self.pipeline(prepared_prompt, **generation_kwargs),  # type: ignore # if self.executor was not passed it was initialized with max_workers=1 in init
        )

        replies = [o.get("generated_text", "") for o in output]

        # Remove stop words from replies if present
        for stop_word in stop_words or []:
            replies = [reply.replace(stop_word, "").rstrip() for reply in replies]

        chat_messages = [
            self.create_message(reply, r_index, tokenizer, prepared_prompt, generation_kwargs, parse_tool_calls=False)
            for r_index, reply in enumerate(replies)
        ]

        return {"replies": chat_messages}

    async def _run_non_streaming_async(  # pylint: disable=too-many-positional-arguments
        self,
        messages: List[ChatMessage],
        tokenizer: Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"],
        generation_kwargs: Dict[str, Any],
        stop_words: Optional[List[str]],
        tools: Optional[Union[List[Tool], Toolset]] = None,
    ):
        """
        Handles async non-streaming generation of responses.
        """
        # convert messages to HF format
        hf_messages = [convert_message_to_hf_format(message) for message in messages]

        if isinstance(tools, Toolset):
            tools = list(tools)

        prepared_prompt = tokenizer.apply_chat_template(
            hf_messages,
            tokenize=False,
            chat_template=self.chat_template,
            add_generation_prompt=True,
            tools=[tc.tool_spec for tc in tools] if tools else None,
        )

        # prepared_prompt is a string, but transformers has some type issues
        prepared_prompt = cast(str, prepared_prompt)

        # Avoid some unnecessary warnings in the generation pipeline call
        generation_kwargs["pad_token_id"] = (
            generation_kwargs.get("pad_token_id", tokenizer.pad_token_id) or tokenizer.eos_token_id
        )

        # Generate responses asynchronously
        output = await asyncio.get_running_loop().run_in_executor(
            self.executor,
            lambda: self.pipeline(prepared_prompt, **generation_kwargs),  # type: ignore # if self.executor was not passed it was initialized with max_workers=1 in init
        )

        replies = [o.get("generated_text", "") for o in output]

        # Remove stop words from replies if present
        for stop_word in stop_words or []:
            replies = [reply.replace(stop_word, "").rstrip() for reply in replies]

        chat_messages = [
            self.create_message(
                reply, r_index, tokenizer, prepared_prompt, generation_kwargs, parse_tool_calls=bool(tools)
            )
            for r_index, reply in enumerate(replies)
        ]

        return {"replies": chat_messages}
