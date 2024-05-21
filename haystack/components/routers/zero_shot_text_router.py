# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)


with LazyImport(message="Run 'pip install transformers[torch,sentencepiece]'") as torch_and_transformers_import:
    from transformers import pipeline

    from haystack.utils.hf import (  # pylint: disable=ungrouped-imports
        deserialize_hf_model_kwargs,
        resolve_hf_pipeline_kwargs,
        serialize_hf_model_kwargs,
    )


@component
class TransformersZeroShotTextRouter:
    """
    Routes a text input onto different output connections depending on which label it has been categorized into.

    This is useful for routing queries to different models in a pipeline depending on their categorization.
    The set of labels to be used for categorization can be specified.

    Example usage in a retrieval pipeline that passes question-like queries to a text embedder optimized for
    query-passage retrieval and passage-like queries to a text embedder optimized for passage-passage retrieval.

    ```python
    from haystack import Document
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.core.pipeline import Pipeline
    from haystack.components.routers import TransformersZeroShotTextRouter
    from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
    from haystack.components.retrievers import InMemoryEmbeddingRetriever

    document_store = InMemoryDocumentStore()
    doc_embedder = SentenceTransformersDocumentEmbedder(model="intfloat/e5-base-v2")
    doc_embedder.warm_up()
    docs = [
        Document(
            content="Germany, officially the Federal Republic of Germany, is a country in the western region of "
            "Central Europe. The nation's capital and most populous city is Berlin and its main financial centre "
            "is Frankfurt; the largest urban area is the Ruhr."
        ),
        Document(
            content="France, officially the French Republic, is a country located primarily in Western Europe. "
            "France is a unitary semi-presidential republic with its capital in Paris, the country's largest city "
            "and main cultural and commercial centre; other major urban areas include Marseille, Lyon, Toulouse, "
            "Lille, Bordeaux, Strasbourg, Nantes and Nice."
        )
    ]
    docs_with_embeddings = doc_embedder.run(docs)
    document_store.write_documents(docs_with_embeddings["documents"])

    p = Pipeline()
    p.add_component(instance=TransformersZeroShotTextRouter(labels=["passage", "query"]), name="text_router")
    p.add_component(
        instance=SentenceTransformersTextEmbedder(model="intfloat/e5-base-v2", prefix="passage: "),
        name="passage_embedder"
    )
    p.add_component(
        instance=SentenceTransformersTextEmbedder(model="intfloat/e5-base-v2", prefix="query: "),
        name="query_embedder"
    )
    p.add_component(
        instance=InMemoryEmbeddingRetriever(document_store=document_store),
        name="query_retriever"
    )
    p.add_component(
        instance=InMemoryEmbeddingRetriever(document_store=document_store),
        name="passage_retriever"
    )

    p.connect("text_router.passage", "passage_embedder.text")
    p.connect("passage_embedder.embedding", "passage_retriever.query_embedding")
    p.connect("text_router.query", "query_embedder.text")
    p.connect("query_embedder.embedding", "query_retriever.query_embedding")

    # Query Example
    p.run({"text_router": {"text": "What is the capital of Germany?"}})

    # Passage Example
    p.run({
        "text_router":{
            "text": "The United Kingdom of Great Britain and Northern Ireland, commonly known as the "\
            "United Kingdom (UK) or Britain, is a country in Northwestern Europe, off the north-western coast of "\
            "the continental mainland."
        }
    })
    ```
    """

    def __init__(
        self,
        labels: List[str],
        multi_label: bool = False,
        model: str = "MoritzLaurer/deberta-v3-base-zeroshot-v1.1-all-33",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        huggingface_pipeline_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the TransformersZeroShotTextRouter.

        :param labels: The set of possible class labels to classify each sequence into. Can be a single label,
            a string of comma-separated labels, or a list of labels.
        :param multi_label: Whether or not multiple candidate labels can be true.
            If False, the scores are normalized such that the sum of the label likelihoods for each sequence is 1.
            If True, the labels are considered independent and probabilities are normalized for each candidate by
            doing a softmax of the entailment score vs. the contradiction score.
        :param model: The name or path of a Hugging Face model for zero-shot text classification.
        :param device: The device on which the model is loaded. If `None`, the default device is automatically
            selected. If a device/device map is specified in `huggingface_pipeline_kwargs`, it overrides this parameter.
        :param token: The API token used to download private models from Hugging Face.
            If this parameter is set to `True`, the token generated when running
            `transformers-cli login` (stored in ~/.huggingface) is used.
        :param huggingface_pipeline_kwargs: Dictionary containing keyword arguments used to initialize the
            Hugging Face pipeline for zero shot text classification.
        """
        torch_and_transformers_import.check()

        self.token = token
        self.labels = labels
        self.multi_label = multi_label
        component.set_output_types(self, **{label: str for label in labels})

        huggingface_pipeline_kwargs = resolve_hf_pipeline_kwargs(
            huggingface_pipeline_kwargs=huggingface_pipeline_kwargs or {},
            model=model,
            task="zero-shot-classification",
            supported_tasks=["zero-shot-classification"],
            device=device,
            token=token,
        )
        self.huggingface_pipeline_kwargs = huggingface_pipeline_kwargs
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
        serialization_dict = default_to_dict(
            self,
            labels=self.labels,
            huggingface_pipeline_kwargs=self.huggingface_pipeline_kwargs,
            token=self.token.to_dict() if self.token else None,
        )

        huggingface_pipeline_kwargs = serialization_dict["init_parameters"]["huggingface_pipeline_kwargs"]
        huggingface_pipeline_kwargs.pop("token", None)

        serialize_hf_model_kwargs(huggingface_pipeline_kwargs)
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformersZeroShotTextRouter":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        deserialize_hf_model_kwargs(data["init_parameters"]["huggingface_pipeline_kwargs"])
        return default_from_dict(cls, data)

    @component.output_types(documents=Dict[str, str])
    def run(self, text: str):
        """
        Run the TransformersZeroShotTextRouter.

        This method routes the text to one of the different edges based on which label it has been categorized into.

        :param text: A str to route to one of the different edges.
        :returns:
            A dictionary with the label as key and the text as value.

        :raises TypeError:
            If the input is not a str.
        :raises RuntimeError:
            If the pipeline has not been loaded.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "The zero-shot classification pipeline has not been loaded. Please call warm_up() before running."
            )

        if not isinstance(text, str):
            raise TypeError("TransformersZeroShotTextRouter expects a str as input.")

        prediction = self.pipeline(sequences=[text], candidate_labels=self.labels, multi_label=self.multi_label)
        predicted_scores = prediction[0]["scores"]
        max_score_index = max(range(len(predicted_scores)), key=predicted_scores.__getitem__)
        label = prediction[0]["labels"][max_score_index]
        return {label: text}
