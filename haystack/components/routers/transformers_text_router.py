# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install transformers[torch,sentencepiece]'") as torch_and_transformers_import:
    from transformers import AutoConfig, pipeline

    from haystack.utils.hf import (  # pylint: disable=ungrouped-imports
        deserialize_hf_model_kwargs,
        resolve_hf_pipeline_kwargs,
        serialize_hf_model_kwargs,
    )


@component
class TransformersTextRouter:
    """
    Routes a text input onto different output connections depending on which label it has been categorized into.

    This is useful for routing queries to different models in a pipeline depending on their categorization.
    The set of labels to be used for categorization are provided by the selected model.

    Example usage in a query pipeline that routes english queries to a text generator optimized for english text and
    german queries to a text generator optimized for german text.

    ```python
    from haystack.core.pipeline import Pipeline
    from haystack.components.routers import TransformersTextRouter
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators import HuggingFaceLocalGenerator

    p = Pipeline()
    p.add_component(
        instance=TransformersTextRouter(model="papluca/xlm-roberta-base-language-detection"),
        name="text_router"
    )
    p.add_component(
        instance=PromptBuilder(template="Answer the question: {{query}}\nAnswer:"),
        name="english_prompt_builder"
    )
    p.add_component(
        instance=PromptBuilder(template="Beantworte die Frage: {{query}}\nAntwort:"),
        name="german_prompt_builder"
    )

    p.add_component(
        instance=HuggingFaceLocalGenerator(model="DiscoResearch/Llama3-DiscoLeo-Instruct-8B-v0.1"),
        name="german_llm"
    )
    p.add_component(
        instance=HuggingFaceLocalGenerator(model="microsoft/Phi-3-mini-4k-instruct"),
        name="english_llm"
    )

    p.connect("text_router.en", "english_prompt_builder.query")
    p.connect("text_router.de", "german_prompt_builder.query")
    p.connect("english_prompt_builder.prompt", "english_llm.prompt")
    p.connect("german_prompt_builder.prompt", "german_llm.prompt")

    # English Example
    print(p.run({"text_router": {"text": "What is the capital of Germany?"}}))

    # German Example
    print(p.run({"text_router": {"text": "Was ist die Hauptstadt von Deutschland?"}}))
    ```
    """

    def __init__(
        self,
        model: str,
        labels: Optional[List[str]] = None,
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        huggingface_pipeline_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the TransformersTextRouter.

        :param model: The name or path of a Hugging Face model for text classification.
        :param labels: The list of labels that the model has been trained to predict. If not provided, the labels
            are fetched from the model configuration file hosted on the HuggingFace Hub using
            `transformers.AutoConfig.from_pretrained`.
        :param device: The device on which the model is loaded. If `None`, the default device is automatically
            selected. If a device/device map is specified in `huggingface_pipeline_kwargs`, it overrides this parameter.
        :param token: The API token used to download private models from Hugging Face.
            If `token` is set to `True`, the token generated when running
            `transformers-cli login` (stored in ~/.huggingface) is used.
        :param huggingface_pipeline_kwargs: Dictionary containing keyword arguments used to initialize the
            Hugging Face pipeline for text classification.
        """
        torch_and_transformers_import.check()

        self.token = token

        huggingface_pipeline_kwargs = resolve_hf_pipeline_kwargs(
            huggingface_pipeline_kwargs=huggingface_pipeline_kwargs or {},
            model=model,
            task="text-classification",
            supported_tasks=["text-classification"],
            device=device,
            token=token,
        )
        self.huggingface_pipeline_kwargs = huggingface_pipeline_kwargs

        if labels is None:
            config = AutoConfig.from_pretrained(
                huggingface_pipeline_kwargs["model"], token=huggingface_pipeline_kwargs["token"]
            )
            self.labels = list(config.label2id.keys())
        else:
            self.labels = labels
        component.set_output_types(self, **{label: str for label in self.labels})

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

        # Verify labels from the model configuration file match provided labels
        labels = set(self.pipeline.model.config.label2id.keys())
        if set(self.labels) != labels:
            raise ValueError(
                f"The provided labels do not match the labels in the model configuration file. "
                f"Provided labels: {self.labels}. Model labels: {labels}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialization_dict = default_to_dict(
            self,
            labels=self.labels,
            model=self.huggingface_pipeline_kwargs["model"],
            huggingface_pipeline_kwargs=self.huggingface_pipeline_kwargs,
            token=self.token.to_dict() if self.token else None,
        )

        huggingface_pipeline_kwargs = serialization_dict["init_parameters"]["huggingface_pipeline_kwargs"]
        huggingface_pipeline_kwargs.pop("token", None)

        serialize_hf_model_kwargs(huggingface_pipeline_kwargs)
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformersTextRouter":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        if data["init_parameters"].get("huggingface_pipeline_kwargs") is not None:
            deserialize_hf_model_kwargs(data["init_parameters"]["huggingface_pipeline_kwargs"])
        return default_from_dict(cls, data)

    @component.output_types(documents=Dict[str, str])
    def run(self, text: str):
        """
        Run the TransformersTextRouter.

        This method routes the text to one of the different edges based on which label it has been categorized into.

        :param text: A str to route to one of the different edges.
        :returns:
            A dictionary with the label as key and the text as value.

        :raises TypeError:
            If the input is not a str.
        :raises RuntimeError:
            If the pipeline has not been loaded because warm_up() was not called before.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "The component TextTransformersRouter wasn't warmed up. Run 'warm_up()' before calling 'run()'."
            )

        if not isinstance(text, str):
            raise TypeError("TransformersTextRouter expects a str as input.")

        prediction = self.pipeline([text], return_all_scores=False, function_to_apply="none")
        label = prediction[0]["label"]
        return {label: text}
