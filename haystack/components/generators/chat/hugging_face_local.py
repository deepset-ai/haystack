import logging
from typing import Any, Dict, List, Literal, Optional, Union, Callable

from haystack import component, default_to_dict, default_from_dict
from haystack.components.generators.hf_utils import StopWordsCriteria, check_generation_params, HFTokenStreamingHandler
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

SUPPORTED_TASKS = ["text-generation", "text2text-generation"]

with LazyImport(message="Run 'pip install transformers[torch]'") as torch_and_transformers_import:
    import torch
    from huggingface_hub import model_info
    from transformers import StoppingCriteriaList, pipeline


@component
class HuggingFaceLocalChatGenerator:
    """
    Generator based on a Hugging Face model.
    This component provides an interface to generate text using a Hugging Face model that runs locally.

    Usage example:
    ```python
    from haystack.components.generators import HuggingFaceLocalChatGenerator

    generator = HuggingFaceLocalChatGenerator(model="mistralai/Mistral-7B-Instruct-v0.2",
                                          generation_kwargs={
                                            "max_new_tokens": 500,
                                            })
    messages = [ChatMessage.from_system("\\nYou are a helpful, respectful and honest assistant"),
                ChatMessage.from_user("What's Natural Language Processing?")]
    print(generator.run(messages))
    # {'replies': ['John Cusack']}
    ```
    """

    def __init__(
        self,
        model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        task: Optional[Literal["text-generation", "text2text-generation"]] = None,
        device: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        chat_template: Optional[str] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        huggingface_pipeline_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        :param model: The name or path of a Hugging Face model for text generation,
            for example, "google/flan-t5-large".
            If the model is also specified in the `huggingface_pipeline_kwargs`, this parameter will be ignored.
        :param task: The task for the Hugging Face pipeline.
            Possible values are "text-generation" and "text2text-generation".
            Generally, decoder-only models like GPT support "text-generation",
            while encoder-decoder models like T5 support "text2text-generation".
            If the task is also specified in the `huggingface_pipeline_kwargs`, this parameter will be ignored.
            If not specified, the component will attempt to infer the task from the model name,
            calling the Hugging Face Hub API.
        :param device: The device on which the model is loaded. (e.g., "cpu", "cuda:0").
            If `device` or `device_map` is specified in the `huggingface_pipeline_kwargs`,
            this parameter will be ignored.
        :param token: The token to use as HTTP bearer authorization for remote files.
            If True, will use the token generated when running huggingface-cli login (stored in ~/.huggingface).
            If the token is also specified in the `huggingface_pipeline_kwargs`, this parameter will be ignored.
        :param chat_template: This optional parameter allows you to specify a Jinja template for formatting chat
            messages. While high-quality and well-supported chat models typically include their own chat templates
            accessible through their tokenizer, there are models that do not offer this feature. For such scenarios,
            or if you wish to use a custom template instead of the model's default, you can use this parameter to
            set your preferred chat template.
        :param generation_kwargs: A dictionary containing keyword arguments to customize text generation.
            Some examples: `max_length`, `max_new_tokens`, `temperature`, `top_k`, `top_p`,...
            See Hugging Face's documentation for more information:
            - https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation
            - https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig
        :param huggingface_pipeline_kwargs: Dictionary containing keyword arguments used to initialize the
            Hugging Face pipeline for text generation.
            These keyword arguments provide fine-grained control over the Hugging Face pipeline.
            In case of duplication, these kwargs override `model`, `task`, `device`, and `token` init parameters.
            See Hugging Face's [documentation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline.task)
            for more information on the available kwargs.
            In this dictionary, you can also include `model_kwargs` to specify the kwargs
            for model initialization:
            https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
        :param stop_words: A list of stop words. If any one of the stop words is generated, the generation is stopped.
            If you provide this parameter, you should not specify the `stopping_criteria` in `generation_kwargs`.
            For some chat models, the output includes both the new text and the original prompt.
            In these cases, it's important to make sure your prompt has no stop words.
        :param streaming_callback: An optional callable for handling streaming responses.
        """
        torch_and_transformers_import.check()

        huggingface_pipeline_kwargs = huggingface_pipeline_kwargs or {}
        generation_kwargs = generation_kwargs or {}

        # check if the huggingface_pipeline_kwargs contain the essential parameters
        # otherwise, populate them with values from other init parameters
        huggingface_pipeline_kwargs.setdefault("model", model)
        huggingface_pipeline_kwargs.setdefault("token", token)
        if (
            device is not None
            and "device" not in huggingface_pipeline_kwargs
            and "device_map" not in huggingface_pipeline_kwargs
        ):
            huggingface_pipeline_kwargs["device"] = device

        # task identification and validation
        if task is None:
            if "task" in huggingface_pipeline_kwargs:
                task = huggingface_pipeline_kwargs["task"]
            elif isinstance(huggingface_pipeline_kwargs["model"], str):
                task = model_info(
                    huggingface_pipeline_kwargs["model"], token=huggingface_pipeline_kwargs["token"]
                ).pipeline_tag

        if task not in SUPPORTED_TASKS:
            raise ValueError(
                f"Task '{task}' is not supported. " f"The supported tasks are: {', '.join(SUPPORTED_TASKS)}."
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

        self.huggingface_pipeline_kwargs = huggingface_pipeline_kwargs
        self.generation_kwargs = generation_kwargs
        self.stop_words = stop_words
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
        if self.pipeline is None:
            self.pipeline = pipeline(**self.huggingface_pipeline_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        serialization_dict = default_to_dict(
            self,
            huggingface_pipeline_kwargs=self.huggingface_pipeline_kwargs,
            generation_kwargs=self.generation_kwargs,
            stop_words=self.stop_words,
        )

        huggingface_pipeline_kwargs = serialization_dict["init_parameters"]["huggingface_pipeline_kwargs"]
        # we don't want to serialize valid tokens
        if isinstance(huggingface_pipeline_kwargs["token"], str):
            serialization_dict["init_parameters"]["huggingface_pipeline_kwargs"].pop("token")
        # convert torch.dtype to string for serialization
        # 1. torch_dtype can be specified in huggingface_pipeline_kwargs
        torch_dtype = huggingface_pipeline_kwargs.get("torch_dtype", None)
        if isinstance(torch_dtype, torch.dtype):
            serialization_dict["init_parameters"]["huggingface_pipeline_kwargs"]["torch_dtype"] = str(torch_dtype)
        # 2. torch_dtype and bnb_4bit_compute_dtype can be specified in model_kwargs
        model_kwargs = huggingface_pipeline_kwargs.get("model_kwargs", {})
        for key, value in model_kwargs.items():
            if key in ["torch_dtype", "bnb_4bit_compute_dtype"] and isinstance(value, torch.dtype):
                serialization_dict["init_parameters"]["huggingface_pipeline_kwargs"]["model_kwargs"][key] = str(value)
        # 3. bnb_4bit_compute_dtype can be specified in model_kwargs["quantization_config"]
        quantization_config = model_kwargs.get("quantization_config", {})
        bnb_4bit_compute_dtype = quantization_config.get("bnb_4bit_compute_dtype", None)
        if isinstance(bnb_4bit_compute_dtype, torch.dtype):
            serialization_dict["init_parameters"]["huggingface_pipeline_kwargs"]["model_kwargs"]["quantization_config"][
                "bnb_4bit_compute_dtype"
            ] = str(bnb_4bit_compute_dtype)

        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceLocalChatGenerator":
        """
        Deserialize this component from a dictionary.
        """
        torch_and_transformers_import.check()
        init_params = data.get("init_parameters", {})
        huggingface_pipeline_kwargs = init_params.get("huggingface_pipeline_kwargs", {})
        model_kwargs = huggingface_pipeline_kwargs.get("model_kwargs", {})

        # convert string to torch.dtype
        # 1. torch_dtype can be specified in huggingface_pipeline_kwargs
        torch_dtype = huggingface_pipeline_kwargs.get("torch_dtype", None)
        if torch_dtype and torch_dtype.startswith("torch."):
            data["init_parameters"]["huggingface_pipeline_kwargs"]["torch_dtype"] = getattr(
                torch, torch_dtype.strip("torch.")
            )
        # 2. torch_dtype and bnb_4bit_compute_dtype can be specified in model_kwargs
        for key, value in model_kwargs.items():
            if key in ["torch_dtype", "bnb_4bit_compute_dtype"] and value.startswith("torch."):
                data["init_parameters"]["huggingface_pipeline_kwargs"]["model_kwargs"][key] = getattr(
                    torch, value.strip("torch.")
                )
        # 3. bnb_4bit_compute_dtype can be specified in model_kwargs["quantization_config"]
        quantization_config = model_kwargs.get("quantization_config", {})
        bnb_4bit_compute_dtype = quantization_config.get("bnb_4bit_compute_dtype", None)
        if bnb_4bit_compute_dtype and bnb_4bit_compute_dtype.startswith("torch."):
            data["init_parameters"]["huggingface_pipeline_kwargs"]["model_kwargs"]["quantization_config"][
                "bnb_4bit_compute_dtype"
            ] = getattr(torch, bnb_4bit_compute_dtype.strip("torch."))

        return default_from_dict(cls, data)

    @component.output_types(replies=List[ChatMessage])
    def run(self, messages: List[ChatMessage], generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Invoke the text generation inference based on the provided messages and generation parameters.

        :param messages: A list of ChatMessage instances representing the input messages.
        :param generation_kwargs: Additional keyword arguments for text generation.
        :return: A list containing the generated responses as ChatMessage instances.
        """
        if self.pipeline is None:
            raise RuntimeError("The generation model has not been loaded. Please call warm_up() before running.")

        tokenizer = self.pipeline.tokenizer

        # check generation kwargs given as parameters to override the default ones
        additional_params = ["n", "stop_words"]
        check_generation_params(generation_kwargs, additional_params)

        # update generation kwargs by merging with the default ones
        generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}
        num_responses = generation_kwargs.pop("n", 1)
        if self.streaming_callback and num_responses > 1:
            logger.warning(
                f"Streaming is enabled, but the number of responses is set to {num_responses}. "
                f"Streaming is only supported for single response generation."
                f"Setting the number of responses to 1."
            )

        stop_words_criteria = (
            StopWordsCriteria(tokenizer, self.stop_words, self.pipeline.device) if self.stop_words else None
        )

        if stop_words_criteria:
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList([stop_words_criteria])
        if self.streaming_callback:
            generation_kwargs["streamer"] = HFTokenStreamingHandler(tokenizer, self.streaming_callback)

        # apply either model's chat template or the user-provided one
        prepared_prompt: str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        output = self.pipeline(prepared_prompt, **generation_kwargs)
        replies = [o["generated_text"] for o in output if "generated_text" in o]
        chat_messages = [ChatMessage.from_assistant(r) for r in replies]
        return {"replies": chat_messages}
