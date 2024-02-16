import logging
from pathlib import Path
from typing import Any, List, Dict, Optional, Union

from haystack import component
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)

SUPPORTED_TASKS = ["zero-shot-classification"]

with LazyImport(message="Run 'pip install transformers[torch,sentencepiece]'") as torch_and_transformers_import:
    from huggingface_hub import model_info
    from transformers import pipeline


@component
class TransformersTextRouter:
    """
    Routes a text input onto different output connections depending on which label it has been categorized into.
    This is useful for routing queries to different models in a pipeline depending on their categorization.
    The set of labels to be used for categorization can be specified.

    Example usage in a retrieval pipeline that passes question-like queries to an embedding retriever and keyword-like
    queries to a BM25 retriever:

    ```python
    document_store = InMemoryDocumentStore()
    p = Pipeline()
    p.add_component(instance=TransformersTextRouter(), name="text_router")
    p.add_component(instance=InMemoryBM25Retriever(document_store=document_store), name="retriever")
    p.connect("text_router.en", "retriever.query")
    p.run({"text_router": {"text": "What's your query?"}})
    ```
    """

    def __init__(
        self,
        labels: List[str],
        model: Union[str, Path] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        pipeline_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        :param labels:
        :param pipeline_kwargs: Dictionary containing keyword arguments used to initialize the
            Hugging Face pipeline for zero shot text classification.
        """
        torch_and_transformers_import.check()
        self.token = token
        self.labels = labels
        component.set_output_types(self, **{label: str for label in labels})

        token = token.resolve_value() if token else None

        # check if the pipeline_kwargs contain the essential parameters
        # otherwise, populate them with values from other init parameters
        pipeline_kwargs.setdefault("model", model)
        pipeline_kwargs.setdefault("token", token)

        device = ComponentDevice.resolve_device(device)
        device.update_hf_kwargs(pipeline_kwargs, overwrite=False)

        # task identification and validation
        task = "zero-shot-classification"
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

        self.pipeline_kwargs = pipeline_kwargs
        self.pipeline = None

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        if isinstance(self.pipeline_kwargs["model"], str):
            return {"model": self.pipeline_kwargs["model"]}
        return {"model": f"[object of type {type(self.pipeline_kwargs['model'])}]"}

    def warm_up(self):
        if self.pipeline is None:
            self.pipeline = pipeline(**self.pipeline_kwargs)

    def run(self, text: str) -> Dict[str, str]:
        """
        Run the TransformersTextRouter. This method routes the text to one of the different edges based on which label
        it has been categorized into.

        :param text: A str to route to one of the different edges.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "The zero-shot classification pipeline has not been loaded. Please call warm_up() before running."
            )

        if not isinstance(text, str):
            raise TypeError("TransformersTextRouter expects a str as input.")

        prediction = self.pipeline(sequences=[text], candidate_labels=self.labels, multi_label=self.multi_label)
        label = prediction[0]["labels"][0]
        return {label: text}
