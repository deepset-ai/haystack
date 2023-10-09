import logging
from typing import Any, Dict, List, Literal, Optional, Union

from haystack.preview import component, default_from_dict, default_to_dict
from haystack.preview.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install torch transformers'") as transformers_import:
    from huggingface_hub import model_info
    from transformers import pipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration


logger = logging.getLogger(__name__)


@component
class HuggingFaceLocalGenerator:
    """
    LLM Generator based on a local model downloaded from the Hugging Face Hub.

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
        :param model_name_or_path: The name or path of the underlying model,
            stored in the Hugging Face Hub. E.g. "google/flan-t5-large".
            If the model is also specified in the pipeline_kwargs, this parameter will be ignored.
        :param task: The task for Hugging Face's pipeline. You can find more information in the model card.
            Generally, decoder-only models like GPT support "text-generation",
            while encoder-decoder models like T5 support "text2text-generation".
            If not specified, the component will try to infer the task, contacting the Hugging Face Hub API.
            If the task is also specified in the pipeline_kwargs, this parameter will be ignored.
        :param device: The device on which the model is loaded. (e.g., "cpu", "cuda:0").
            If device or device_map is specified in the pipeline_kwargs, this parameter will be ignored.
        :param token: The token to use as HTTP bearer authorization for remote files.
            If True, will use the token generated when running huggingface-cli login (stored in ~/.huggingface).
            If the token is also specified in the pipeline_kwargs, this parameter will be ignored.
        :param generation_kwargs: Dictionary containing keyword arguments to customize text generation.
            See Hugging Face's documentation for more information.
            https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation
            https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.GenerationConfig
        :param pipeline_kwargs: Dictionary containing keyword arguments used to initialize the pipeline.
            You can use these keyword arguments for fine-grained control of the pipeline.
            In case of duplication, these kwargs override model_name_or_path, task, device, and token init parameters.
            See Hugging Face's documentation for more information on the available kwargs.
            https://huggingface.co/docs/transformers/en/main_classes/pipelines#transformers.pipeline.task
            In this dictionary, you can also specify enter a key called "model_kwargs" to specify the kwargs
            for the model initialization. See Hugging Face's documentation:
            https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained
        """
        transformers_import.check()

        self.generation_kwargs = generation_kwargs or {}

        pipeline_kwargs = pipeline_kwargs or {}

        # check if the essential kwargs are already in the pipeline_kwargs
        # otherwise, take them from the other init parameters (if not None)
        if "model" not in pipeline_kwargs:
            pipeline_kwargs["model"] = model_name_or_path
        if "token" not in pipeline_kwargs:
            pipeline_kwargs["token"] = token
        if device is not None and "device" not in pipeline_kwargs and "device_map" not in pipeline_kwargs:
            pipeline_kwargs["device"] = device

        # task identification and validation
        if task is None:
            if "task" in pipeline_kwargs:
                task = pipeline_kwargs["task"]
            elif isinstance(pipeline_kwargs["model"], str):
                task = model_info(pipeline_kwargs["model"], token=pipeline_kwargs["token"]).pipeline_tag

        if task not in ["text-generation", "text2text-generation"]:
            raise ValueError(
                f"Task name {task} is not supported. "
                f"We only support text2text-generation and text-generation tasks."
            )
        pipeline_kwargs["task"] = task

        self.pipeline_kwargs = pipeline_kwargs
        self.pipeline = None

    def warm_up(self):
        print(f"{self.pipeline_kwargs=}")

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

        output = self.pipeline(prompt, **self.generation_kwargs)
        generated_texts = [o["generated_text"] for o in output if "generated_text" in o]
        return {"replies": generated_texts}

    # serialization!


if __name__ == "__main__":
    import json

    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    hf = HuggingFaceLocalGenerator(
        pipeline_kwargs={"model": model, "tokenizer": tokenizer}, task="text2text-generation"
    )
    hf.warm_up()
    hf.pipeline_kwargs["model"] = model.config.to_dict()
    hf.pipeline_kwargs["tokenizer"] = tokenizer.name_or_path
    serialized = hf.to_dict()
    with open("serialized.txt", "w") as f:
        f.write(str(serialized))

    print(serialized)
    new_hf = HuggingFaceLocalGenerator.from_dict(serialized)
    print(new_hf.to_dict())
    new_hf.warm_up()
    print(new_hf.run("What is Google?"))
