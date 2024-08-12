# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)

with LazyImport(message="Run 'pip install transformers[torch,sentencepiece]'") as torch_and_transformers_import:
    from transformers import pipeline

    from haystack.utils.hf import deserialize_hf_model_kwargs, resolve_hf_pipeline_kwargs, serialize_hf_model_kwargs


class TransformersZeroShotDocumentClassifier:
    def __init__(
        self,
        model: str,
        labels: List[str],
        multi_label: bool = False,
        classification_field: Optional[str] = None,
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        huggingface_pipeline_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the TransformersZeroShotDocumentClassifier.

        Available models for the task of zero-shot-classification include:
        - ``'valhalla/distilbart-mnli-12-3'``
        - ``'cross-encoder/nli-distilroberta-base'``

        See https://huggingface.co/models for full list of available models.
        Filter for zero-shot classification models (NLI): https://huggingface.co/models?pipeline_tag=zero-shot-classification&sort=downloads&search=nli

        :param model:
            The name or path of a Hugging Face model for zero shot document classification.
        :param labels:
            The set of possible class labels to classify each document into, e.g.,
            ["positive", "negative"] otherwise None. Given a LABEL, the sequence fed to the model is "<cls> sequence to
            classify <sep> This example is LABEL . <sep>" and the model predicts whether that sequence is a
            contradiction or an entailment.
        :param multi_label:
            Whether or not multiple candidate labels can be true.
            If `False`, the scores are normalized such that
            the sum of the label likelihoods for each sequence is 1. If `True`, the labels are considered
            independent and probabilities are normalized for each candidate by doing a softmax of the entailment
            score vs. the contradiction score.
        :param classification_field:
            Name of document's meta field to be used for classification.
            If left unset, `Document.content` is used by default.
        :param device:
            The device on which the model is loaded. If `None`, the default device is automatically
            selected. If a device/device map is specified in `huggingface_pipeline_kwargs`, it overrides this parameter.
        :param token:
            The API token used to download private models from Hugging Face. If `token` is set to `True`, the token
            generated when running `transformers-cli login` (stored in ~/.huggingface) is used.
        :param huggingface_pipeline_kwargs:
            Dictionary containing keyword arguments used to initialize the
            Hugging Face pipeline for text classification.
        """
        torch_and_transformers_import.check()

        self.classification_field = classification_field

        self.token = token
        self.labels = labels
        self.multi_label = multi_label

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
            model=self.huggingface_pipeline_kwargs["model"],
            huggingface_pipeline_kwargs=self.huggingface_pipeline_kwargs,
            token=self.token.to_dict() if self.token else None,
        )

        huggingface_pipeline_kwargs = serialization_dict["init_parameters"]["huggingface_pipeline_kwargs"]
        huggingface_pipeline_kwargs.pop("token", None)

        serialize_hf_model_kwargs(huggingface_pipeline_kwargs)
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformersZeroShotDocumentClassifier":
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

    @component.output_types(documents=Dict[str, List[Document]])
    def run(self, documents: List[Document], batch_size: int = 1):
        """
        This method classifies the documents based on the provided labels and adds it to their metadata.

        Documents are updated in place. The classification results are stored in the `classification` dict within
        each document's metadata. If `multi_label` is set to `True`, the scores for each label are available under
        the `details` key within the `classification` dictionary.

        :param documents:
            Documents to process.
        :param batch_size:
            Batch size used for processing the content in each document.
        """

        if self.pipeline is None:
            raise RuntimeError(
                "The component TransformerZeroShotDocumentClassifier wasn't warmed up. "
                "Run 'warm_up()' before calling 'run()'."
            )

        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "DocumentLanguageClassifier expects a list of Document as input. "
                "In case you want to classify a text, please use the TextLanguageClassifier."
            )

        texts = [
            doc.content if self.classification_field is None else doc.meta[self.classification_field]
            for doc in documents
        ]

        predictions = self.pipeline(texts, self.labels, multi_label=self.multi_label, batch_size=batch_size)

        for prediction, document in zip(predictions, documents):
            formatted_prediction = {
                "label": prediction["labels"][0],
                "score": prediction["scores"][0],
                "details": dict(zip(prediction["labels"], prediction["scores"])),
            }
            document.meta["classification"] = formatted_prediction

        return {"documents": documents}
