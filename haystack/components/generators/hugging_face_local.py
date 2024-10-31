# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, List, Literal, Optional

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import StreamingChunk
from haystack.lazy_imports import LazyImport
from haystack.utils import (
    ComponentDevice,
    Secret,
    deserialize_callable,
    deserialize_secrets_inplace,
    serialize_callable,
)
from haystack.utils.hf import deserialize_hf_model_kwargs, serialize_hf_model_kwargs

logger = logging.getLogger(__name__)

SUPPORTED_TASKS = ["text-generation", "text2text-generation"]

with LazyImport(message="Run 'pip install \"transformers[torch]\"'") as transformers_import:
    from transformers import StoppingCriteriaList, pipeline

    from haystack.utils.hf import (  # pylint: disable=ungrouped-imports
        HFTokenStreamingHandler,
        StopWordsCriteria,
        resolve_hf_pipeline_kwargs,
    )


@component
class HuggingFaceLocalGenerator:
    """
    Generates text using models from Hugging Face that run locally.

    LLMs running locally may need powerful hardware.

    ### Usage example

    ```python
    from haystack.components.generators import HuggingFaceLocalGenerator

    generator = HuggingFaceLocalGenerator(
        model="google/flan-t5-large",
        task="text2text-generation",
        generation_kwargs={"max_new_tokens": 100, "temperature": 0.9})

    generator.warm_up()

    print(generator.run("Who is the best American actor?"))
    # {'replies': ['John Cusack']}
    ```
    """

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        model: str = "google/flan-t5-base",
        task: Optional[Literal["text-generation", "text2text-generation"]] = None,
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        generation_kwargs: Optional[Dict[str, Any]] = None,
        huggingface_pipeline_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
    ):
        """
        Creates an instance of a HuggingFaceLocalGenerator.

        :param model: The Hugging Face text generation model name or path.
        :param task: The task for the Hugging Face pipeline. Possible options:
            - `text-generation`: Supported by decoder models, like GPT.
            - `text2text-generation`: Supported by encoder-decoder models, like T5.
            If the task is specified in `huggingface_pipeline_kwargs`, this parameter is ignored.
            If not specified, the component calls the Hugging Face API to infer the task from the model name.
        :param device: The device for loading the model. If `None`, automatically selects the default device.
            If a device or device map is specified in `huggingface_pipeline_kwargs`, it overrides this parameter.
        :param token: The token to use as HTTP bearer authorization for remote files.
            If the token is specified in `huggingface_pipeline_kwargs`, this parameter is ignored.
        :param generation_kwargs: A dictionary with keyword arguments to customize text generation.
            Some examples: `max_length`, `max_new_tokens`, `temperature`, `top_k`, `top_p`.
            See Hugging Face's documentation for more information:
            - [customize-text-generation](https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation)
            - [transformers.GenerationConfig](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig)
        :param huggingface_pipeline_kwargs: Dictionary with keyword arguments to initialize the
            Hugging Face pipeline for text generation.
            These keyword arguments provide fine-grained control over the Hugging Face pipeline.
            In case of duplication, these kwargs override `model`, `task`, `device`, and `token` init parameters.
            For available kwargs, see [Hugging Face documentation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline.task).
            In this dictionary, you can also include `model_kwargs` to specify the kwargs for model initialization:
            [transformers.PreTrainedModel.from_pretrained](https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)
        :param stop_words: If the model generates a stop word, the generation stops.
            If you provide this parameter, don't specify the `stopping_criteria` in `generation_kwargs`.
            For some chat models, the output includes both the new text and the original prompt.
            In these cases, make sure your prompt has no stop words.
        :param streaming_callback: An optional callable for handling streaming responses.
        """
        transformers_import.check()

        self.token = token
        generation_kwargs = generation_kwargs or {}

        huggingface_pipeline_kwargs = resolve_hf_pipeline_kwargs(
            huggingface_pipeline_kwargs=huggingface_pipeline_kwargs or {},
            model=model,
            task=task,
            supported_tasks=SUPPORTED_TASKS,
            device=device,
            token=token,
        )

        # if not specified, set return_full_text to False for text-generation
        # only generated text is returned (excluding prompt)
        task = huggingface_pipeline_kwargs["task"]
        if task == "text-generation":
            generation_kwargs.setdefault("return_full_text", False)

        if stop_words and "stopping_criteria" in generation_kwargs:
            raise ValueError(
                "Found both the `stop_words` init parameter and the `stopping_criteria` key in `generation_kwargs`. "
                "Please specify only one of them."
            )
        generation_kwargs.setdefault("max_new_tokens", 512)

        self.huggingface_pipeline_kwargs = huggingface_pipeline_kwargs
        self.generation_kwargs = generation_kwargs
        self.stop_words = stop_words
        self.pipeline = None
        self.stopping_criteria_list = None
        self.streaming_callback = streaming_callback

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        if isinstance(self.huggingface_pipeline_kwargs["model"], str):
            return {"model": self.huggingface_pipeline_kwargs["model"]}
        return {"model": f"[object of type {type(self.huggingface_pipeline_kwargs['model'])}]"}

    @property
    def _warmed_up(self) -> bool:
        if self.stop_words:
            return (self.pipeline is not None) and (self.stopping_criteria_list is not None)
        return self.pipeline is not None

    def warm_up(self):
        """
        Initializes the component.
        """
        if self._warmed_up:
            return

        if self.pipeline is None:
            self.pipeline = pipeline(**self.huggingface_pipeline_kwargs)

        if self.stop_words:
            stop_words_criteria = StopWordsCriteria(
                tokenizer=self.pipeline.tokenizer, stop_words=self.stop_words, device=self.pipeline.device
            )
            self.stopping_criteria_list = StoppingCriteriaList([stop_words_criteria])

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
            stop_words=self.stop_words,
            token=self.token.to_dict() if self.token else None,
        )

        huggingface_pipeline_kwargs = serialization_dict["init_parameters"]["huggingface_pipeline_kwargs"]
        huggingface_pipeline_kwargs.pop("token", None)

        serialize_hf_model_kwargs(huggingface_pipeline_kwargs)
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceLocalGenerator":
        """
        Deserializes the component from a dictionary.

        :param data:
            The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        init_params = data.get("init_parameters", {})
        serialized_callback_handler = init_params.get("streaming_callback")
        if serialized_callback_handler:
            data["init_parameters"]["streaming_callback"] = deserialize_callable(serialized_callback_handler)

        huggingface_pipeline_kwargs = init_params.get("huggingface_pipeline_kwargs", {})
        deserialize_hf_model_kwargs(huggingface_pipeline_kwargs)
        return default_from_dict(cls, data)

    @component.output_types(replies=List[str])
    def run(
        self,
        prompt: str,
        streaming_callback: Optional[Callable[[StreamingChunk], None]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Run the text generation model on the given prompt.

        :param prompt:
            A string representing the prompt.
        :param streaming_callback:
            A callback function that is called when a new token is received from the stream.
        :param generation_kwargs:
            Additional keyword arguments for text generation.

        :returns:
            A dictionary containing the generated replies.
            - replies: A list of strings representing the generated replies.
        """
        if not self._warmed_up:
            raise RuntimeError(
                "The component HuggingFaceLocalGenerator was not warmed up. Please call warm_up() before running."
            )

        if not prompt:
            return {"replies": []}

        # merge generation kwargs from init method with those from run method
        updated_generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        # check if streaming_callback is passed
        streaming_callback = streaming_callback or self.streaming_callback

        if streaming_callback:
            num_responses = updated_generation_kwargs.get("num_return_sequences", 1)
            if num_responses > 1:
                msg = (
                    "Streaming is enabled, but the number of responses is set to {num_responses}. "
                    "Streaming is only supported for single response generation. "
                    "Setting the number of responses to 1."
                )
                logger.warning(msg, num_responses=num_responses)
                updated_generation_kwargs["num_return_sequences"] = 1
            # streamer parameter hooks into HF streaming, HFTokenStreamingHandler is an adapter to our streaming
            updated_generation_kwargs["streamer"] = HFTokenStreamingHandler(
                self.pipeline.tokenizer,  # type: ignore
                streaming_callback,
                self.stop_words,  # type: ignore
            )

        output = self.pipeline(prompt, stopping_criteria=self.stopping_criteria_list, **updated_generation_kwargs)  # type: ignore
        replies = [o["generated_text"] for o in output if "generated_text" in o]

        if self.stop_words:
            # the output of the pipeline includes the stop word
            replies = [reply.replace(stop_word, "").rstrip() for reply in replies for stop_word in self.stop_words]

        return {"replies": replies}
