import logging
from typing import Any, Dict, List, Literal, Optional, Union
from copy import deepcopy

from haystack.preview import component, default_to_dict
from haystack.preview.lazy_imports import LazyImport

logger = logging.getLogger(__name__)

SUPPORTED_TASKS = ["text-generation", "text2text-generation"]

with LazyImport(message="Run 'pip install transformers[torch]'") as torch_and_transformers_import:
    import torch
    from huggingface_hub import model_info
    from transformers import (
        pipeline,
        StoppingCriteriaList,
        StoppingCriteria,
        PreTrainedTokenizer,
        PreTrainedTokenizerFast,
    )

    class StopWordsCriteria(StoppingCriteria):
        """
        Stops text generation if any one of the stop words is generated.

        Note: When a stop word is encountered, the generation of new text is stopped.
        However, if the stop word is in the prompt itself, it can stop generating new text
        prematurely after the first token. This is particularly important for LLMs designed
        for dialogue generation. For these models, like for example mosaicml/mpt-7b-chat,
        the output includes both the new text and the original prompt. Therefore, it's important
        to make sure your prompt has no stop words.
        """

        def __init__(
            self,
            tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
            stop_words: List[str],
            device: Union[str, torch.device] = "cpu",
        ):
            super().__init__()
            encoded_stop_words = tokenizer(stop_words, add_special_tokens=False, padding=True, return_tensors="pt")
            self.stop_ids = encoded_stop_words.input_ids.to(device)

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_id in self.stop_ids:
                found_stop_word = self.is_stop_word_found(input_ids, stop_id)
                if found_stop_word:
                    return True
            return False

        def is_stop_word_found(self, generated_text_ids: torch.Tensor, stop_id: torch.Tensor) -> bool:
            generated_text_ids = generated_text_ids[-1]
            len_generated_text_ids = generated_text_ids.size(0)
            len_stop_id = stop_id.size(0)
            result = all(generated_text_ids[len_generated_text_ids - len_stop_id :].eq(stop_id))
            return result


@component
class HuggingFaceLocalGenerator:
    """
    Generator based on a Hugging Face model.
    This component provides an interface to generate text using a Hugging Face model that runs locally.

    Usage example:
    ```python
    from haystack.preview.components.generators import HuggingFaceLocalGenerator

    generator = HuggingFaceLocalGenerator(model="google/flan-t5-large",
                                          task="text2text-generation",
                                          generation_kwargs={
                                            "max_new_tokens": 100,
                                            "temperature": 0.9,
                                            })

    print(generator.run("Who is the best American actor?"))
    # {'replies': ['John Cusack']}
    ```
    """

    def __init__(
        self,
        model_name_or_path: str = "google/flan-t5-base",
        task: Optional[Literal["text-generation", "text2text-generation"]] = None,
        device: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        huggingface_pipeline_kwargs: Optional[Dict[str, Any]] = None,
        stop_words: Optional[List[str]] = None,
    ):
        """
        :param model_name_or_path: The name or path of a Hugging Face model for text generation,
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
        :param generation_kwargs: A dictionary containing keyword arguments to customize text generation.
            Some examples: `max_length`, `max_new_tokens`, `temperature`, `top_k`, `top_p`,...
            See Hugging Face's documentation for more information:
            - https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation
            - https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig
        :param huggingface_pipeline_kwargs: Dictionary containing keyword arguments used to initialize the
            Hugging Face pipeline for text generation.
            These keyword arguments provide fine-grained control over the Hugging Face pipeline.
            In case of duplication, these kwargs override `model_name_or_path`, `task`, `device`, and `token` init parameters.
            See Hugging Face's [documentation](https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline.task)
            for more information on the available kwargs.
            In this dictionary, you can also include `model_kwargs` to specify the kwargs
            for model initialization:
            https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
        :param stop_words: A list of stop words. If any one of the stop words is generated, the generation is stopped.
            If you provide this parameter, you should not specify the `stopping_criteria` in `generation_kwargs`.
            For some chat models, the output includes both the new text and the original prompt.
            In these cases, it's important to make sure your prompt has no stop words.
        """
        torch_and_transformers_import.check()

        huggingface_pipeline_kwargs = huggingface_pipeline_kwargs or {}
        generation_kwargs = generation_kwargs or {}

        # check if the huggingface_pipeline_kwargs contain the essential parameters
        # otherwise, populate them with values from other init parameters
        huggingface_pipeline_kwargs.setdefault("model", model_name_or_path)
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
        self.pipeline = None
        self.stopping_criteria_list = None

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

        if self.stop_words and self.stopping_criteria_list is None:
            stop_words_criteria = StopWordsCriteria(
                tokenizer=self.pipeline.tokenizer, stop_words=self.stop_words, device=self.pipeline.device
            )
            self.stopping_criteria_list = StoppingCriteriaList([stop_words_criteria])

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        pipeline_kwargs_to_serialize = deepcopy(self.huggingface_pipeline_kwargs)

        # we don't want to serialize valid tokens
        if isinstance(pipeline_kwargs_to_serialize["token"], str):
            pipeline_kwargs_to_serialize["token"] = None

        return default_to_dict(
            self,
            huggingface_pipeline_kwargs=pipeline_kwargs_to_serialize,
            generation_kwargs=self.generation_kwargs,
            stop_words=self.stop_words,
        )

    @component.output_types(replies=List[str])
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        """
        Run the text generation model on the given prompt.

        :param prompt: A string representing the prompt.
        :param generation_kwargs: Additional keyword arguments for text generation.
        :return: A dictionary containing the generated replies.
        """
        if self.pipeline is None:
            raise RuntimeError("The generation model has not been loaded. Please call warm_up() before running.")

        if not prompt:
            return {"replies": []}

        # merge generation kwargs from init method with those from run method
        updated_generation_kwargs = {**self.generation_kwargs, **(generation_kwargs or {})}

        output = self.pipeline(prompt, stopping_criteria=self.stopping_criteria_list, **updated_generation_kwargs)
        replies = [o["generated_text"] for o in output if "generated_text" in o]

        if self.stop_words:
            # the output of the pipeline includes the stop word
            replies = [reply.replace(stop_word, "").rstrip() for reply in replies for stop_word in self.stop_words]

        return {"replies": replies}
