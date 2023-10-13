import logging
from typing import Any, Dict, List, Literal, Optional, Union

from haystack.preview import component, default_from_dict, default_to_dict
from haystack.preview.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install transformers'") as transformers_import:
    from huggingface_hub import model_info
    from transformers import pipeline

logger = logging.getLogger(__name__)


SUPPORTED_TASKS = ["text-generation", "text2text-generation"]


@component
class HuggingFaceLocalGenerator:
    """
    Generator based on a Hugging Face model.
    This component provides an interface to generate text using a Hugging Face model that runs locally.

    Usage example:
    ```python
    from haystack.preview.components.generators.hugging_face import HuggingFaceLocalGenerator

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
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param model_name_or_path: The name or path of a Hugging Face model for text generation.
            E.g. "google/flan-t5-large".
            If the model is also specified in the pipeline_kwargs, this parameter will be ignored.
        :param task: The task for the Hugging Face pipeline.
            Possible values are "text-generation" and "text2text-generation".
            Generally, decoder-only models like GPT support "text-generation",
            while encoder-decoder models like T5 support "text2text-generation".
            If the task is also specified in the pipeline_kwargs, this parameter will be ignored.
            If not specified, the component will attempt to infer the task from the model name,
            calling the Hugging Face Hub API.
        :param device: The device on which the model is loaded. (e.g., "cpu", "cuda:0").
            If device or device_map is specified in the pipeline_kwargs, this parameter will be ignored.
        :param token: The token to use as HTTP bearer authorization for remote files.
            If True, will use the token generated when running huggingface-cli login (stored in ~/.huggingface).
            If the token is also specified in the pipeline_kwargs, this parameter will be ignored.
        :param generation_kwargs: a dictionary containing keyword arguments to customize text generation.
            Some examples: `max_length`, `max_new_tokens`, `temperature`, `top_k`, `top_p`,...
            See Hugging Face's documentation for more information.
            https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation
            https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig
        :param pipeline_kwargs: Dictionary containing keyword arguments used to initialize the pipeline.
            These keyword arguments provide fine-grained control over the pipeline.
            In case of duplication, these kwargs override model_name_or_path, task, device, and token init parameters.
            See Hugging Face's documentation for more information on the available kwargs.
            https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline.task
            In this dictionary, you can also include "model_kwargs" to specify the kwargs
            for model initialization:
            https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
        """
        transformers_import.check()

        pipeline_kwargs = pipeline_kwargs or {}
        generation_kwargs = generation_kwargs or {}

        # check if the pipeline_kwargs contain the essential parameters
        # otherwise, populate them with values from other init parameters
        pipeline_kwargs.setdefault("model", model_name_or_path)
        pipeline_kwargs.setdefault("token", token)
        if device is not None and "device" not in pipeline_kwargs and "device_map" not in pipeline_kwargs:
            pipeline_kwargs["device"] = device

        # task identification and validation
        if task is None:
            if "task" in pipeline_kwargs:
                task = pipeline_kwargs["task"]
            elif isinstance(pipeline_kwargs["model"], str):
                task = model_info(pipeline_kwargs["model"], token=pipeline_kwargs["token"]).pipeline_tag

        if task not in SUPPORTED_TASKS:
            raise ValueError(
                f"Task '{task}' is not supported. " f"The supported tasks are: {', '.join(SUPPORTED_TASKS)}."
            )
        pipeline_kwargs["task"] = task

        # if not specified, set return_full_text to False for text-generation
        # only generated text is returned (excluding prompt)
        if task == "text-generation" and "return_full_text" not in generation_kwargs:
            generation_kwargs["return_full_text"] = False

        self.pipeline_kwargs = pipeline_kwargs
        self.generation_kwargs = generation_kwargs
        self.pipeline = None

    def warm_up(self):
        if self.pipeline is None:
            self.pipeline = pipeline(**self.pipeline_kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(self, pipeline_kwargs=self.pipeline_kwargs, generation_kwargs=self.generation_kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceLocalGenerator":
        """
        Deserialize this component from a dictionary.
        """
        return default_from_dict(cls, data)

    @component.output_types(replies=List[str], metadata=List[Dict[str, Any]])
    def run(self, prompt: str):
        if self.pipeline is None:
            raise RuntimeError("The generation model has not been loaded. Please call warm_up() before running.")

        replies = []
        if prompt:
            output = self.pipeline(prompt, **self.generation_kwargs)
            replies = [o["generated_text"] for o in output if "generated_text" in o]

        return {"replies": replies}
