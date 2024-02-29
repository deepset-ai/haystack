import sys
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.lazy_imports import LazyImport
from haystack.utils import (
    ComponentDevice,
    Secret,
    deserialize_callable,
    deserialize_secrets_inplace,
    serialize_callable,
)

logger = logging.getLogger(__name__)

with LazyImport(message="Run 'pip install transformers[torch]'") as torch_and_transformers_import:
    from huggingface_hub import model_info
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, StoppingCriteriaList, pipeline

    from haystack.utils.hf import (  # pylint: disable=ungrouped-imports
        HFTokenStreamingHandler,
        StopWordsCriteria,
        deserialize_hf_model_kwargs,
        serialize_hf_model_kwargs,
    )


PIPELINE_SUPPORTED_TASKS = ["text-generation", "text2text-generation"]


@component
class HuggingFaceLocalChatGenerator:
    """
    The `HuggingFaceLocalChatGenerator` class is a component designed for generating chat responses using models from
    Hugging Face's model hub. It is tailored for local runtime text generation tasks and provides a convenient interface
    for working with chat-based models, such as `HuggingFaceH4/zephyr-7b-beta` or `meta-llama/Llama-2-7b-chat-hf`
    etc.

    Usage example:
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
        [ChatMessage(content=' Natural Language Processing (NLP) is a subfield of artificial intelligence that deals
        with the interaction between computers and human language. It enables computers to understand, interpret, and
        generate human language in a valuable way. NLP involves various techniques such as speech recognition, text
        analysis, sentiment analysis, and machine translation. The ultimate goal is to make it easier for computers to
        process and derive meaning from human language, improving communication between humans and machines.',
        role=<ChatRole.ASSISTANT: 'assistant'>,
        name=None,
        meta={'finish_reason': 'stop', 'index': 0, 'model':
              'mistralai/Mistral-7B-Instruct-v0.2',
              'usage': {'completion_tokens': 90, 'prompt_tokens': 19, 'total_tokens': 109}})
              ]
    }
    ```
    """

    def __init__(
        self,
        model: str = "HuggingFaceH4/zephyr-7b-beta",
        task: Optional[Literal["text-generation", "text2text-generation"]] = None,
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        chat_template: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        huggingface_pipeline_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        :param model: The name or path of a Hugging Face model for text generation,
            for example, `mistralai/Mistral-7B-Instruct-v0.2`, `TheBloke/OpenHermes-2.5-Mistral-7B-16k-AWQ`, etc.
            The important aspect of the model is that it should be a chat model and that it supports ChatML messaging
            format.
            If the model is also specified in the `huggingface_pipeline_kwargs`, this parameter will be ignored.
        :param task: The task for the Hugging Face pipeline.
            Possible values are "text-generation" and "text2text-generation".
            Generally, decoder-only models like GPT support "text-generation",
            while encoder-decoder models like T5 support "text2text-generation".
            If the task is also specified in the `huggingface_pipeline_kwargs`, this parameter will be ignored.
            If not specified, the component will attempt to infer the task from the model name,
            calling the Hugging Face Hub API.
        :param device: The device on which the model is loaded. If `None`, the default device is automatically
            selected. If a device/device map is specified in `huggingface_pipeline_kwargs`, it overrides this parameter.
        :param token: The token to use as HTTP bearer authorization for remote files.
            If the token is also specified in the `huggingface_pipeline_kwargs`, this parameter will be ignored.
        :param chat_template: This optional parameter allows you to specify a Jinja template for formatting chat
            messages. While high-quality and well-supported chat models typically include their own chat templates
            accessible through their tokenizer, there are models that do not offer this feature. For such scenarios,
            or if you wish to use a custom template instead of the model's default, you can use this parameter to
            set your preferred chat template.
        :param generation_kwargs: A dictionary containing keyword arguments to customize text generation.
            Some examples: `max_length`, `max_new_tokens`, `temperature`, `top_k`, `top_p`, etc.
            See Hugging Face's documentation for more information:
            - - [customize-text-generation](https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation)
            - - [GenerationConfig](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig)
            - The only generation_kwargs we set by default is max_new_tokens, which is set to 512 tokens.
        :param huggingface_pipeline_kwargs: Dictionary containing keyword arguments used to initialize the
            Hugging Face pipeline for text generation.
            These keyword arguments provide fine-grained control over the Hugging Face pipeline.
            In case of duplication, these kwargs override `model`, `task`, `device`, and `token` init parameters.
            See Hugging Face's [documentation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline.task)
            for more information on the available kwargs.
            In this dictionary, you can also include `model_kwargs` to specify the kwargs for [model initialization](https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)
        :param stop_words: A list of stop words. If any one of the stop words is generated, the generation is stopped.
            If you provide this parameter, you should not specify the `stopping_criteria` in `generation_kwargs`.
            For some chat models, the output includes both the new text and the original prompt.
            In these cases, it's important to make sure your prompt has no stop words.
        :param streaming_callback: An optional callable for handling streaming responses.
        """
        torch_and_transformers_import.check()

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
                ).pipeline_tag

        if task not in PIPELINE_SUPPORTED_TASKS:
            raise ValueError(
                f"Task '{task}' is not supported. " f"The supported tasks are: {', '.join(PIPELINE_SUPPORTED_TASKS)}."
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

        self.huggingface_pipeline_kwargs = huggingface_pipeline_kwargs
        self.generation_kwargs = generation_kwargs
        self.chat_template = chat_template
        self.streaming_callback = streaming_callback
        self.pipeline = None

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
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)

        huggingface_pipeline_kwargs = init_params.get("huggingface_pipeline_kwargs", {})
        deserialize_hf_model_kwargs(huggingface_pipeline_kwargs)
        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Invoke text generation inference based on the provided messages and generation parameters.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param generation_kwargs: Additional keyword arguments for text generation.
        :returns:
            A list containing the generated responses as ChatMessage instances.
        """
        if self.pipeline is None:
            raise RuntimeError("The generation model has not been loaded. Please call warm_up() before running.")

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

        if self.streaming_callback:
            num_responses = generation_kwargs.get("num_return_sequences", 1)
            if num_responses > 1:
                logger.warning(
                    "Streaming is enabled, but the number of responses is set to %d. "
                    "Streaming is only supported for single response generation. "
                    "Setting the number of responses to 1.",
                    num_responses,
                )
                generation_kwargs["num_return_sequences"] = 1
            # streamer parameter hooks into HF streaming, HFTokenStreamingHandler is an adapter to our streaming
            generation_kwargs["streamer"] = HFTokenStreamingHandler(tokenizer, self.streaming_callback, stop_words)

        # Prepare the prompt for the model
        prepared_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, chat_template=self.chat_template, add_generation_prompt=True
        )

        # Avoid some unnecessary warnings in the generation pipeline call
        generation_kwargs["pad_token_id"] = (
            generation_kwargs.get("pad_token_id", tokenizer.pad_token_id) or tokenizer.eos_token_id
        )

        # Generate responses
        output = self.pipeline(prepared_prompt, **generation_kwargs)
        replies = [o.get("generated_text", "") for o in output]

        # Remove stop words from replies if present
        for stop_word in stop_words:
            replies = [reply.replace(stop_word, "").rstrip() for reply in replies]

        # Create ChatMessage instances for each reply
        chat_messages = [
            self.create_message(reply, r_index, tokenizer, prepared_prompt, generation_kwargs)
            for r_index, reply in enumerate(replies)
        ]
        return {"replies": chat_messages}

    def create_message(
        self,
        text: str,
        index: int,
        tokenizer: Union["PreTrainedTokenizer", "PreTrainedTokenizerFast"],
        prompt: str,
        generation_kwargs: Dict[str, Any],
    ) -> ChatMessage:
        """
        Create a ChatMessage instance from the provided text, populated with metadata.

        :param text: The generated text.
        :param index: The index of the generated text.
        :param tokenizer: The tokenizer used for generation.
        :param prompt: The prompt used for generation.
        :param generation_kwargs: The generation parameters.
        :returns: A ChatMessage instance.
        """
        completion_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        prompt_token_count = len(tokenizer.encode(prompt, add_special_tokens=False))
        total_tokens = prompt_token_count + completion_tokens

        # not the most sophisticated finish_reason detection, improve later to match
        # https://platform.openai.com/docs/guides/text-generation/chat-completions-response-format
        finish_reason = (
            "length" if completion_tokens >= generation_kwargs.get("max_new_tokens", sys.maxsize) else "stop"
        )

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

        return ChatMessage.from_assistant(text, meta=meta)

    def _validate_stop_words(self, stop_words: Optional[List[str]]) -> Optional[List[str]]:
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

        # deduplicate stop words
        stop_words = list(set(stop_words or []))
        return stop_words
