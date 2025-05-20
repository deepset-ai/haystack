# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, DeviceMap, Secret, deserialize_secrets_inplace
from haystack.utils.hf import deserialize_hf_model_kwargs, resolve_hf_device_map, serialize_hf_model_kwargs

with LazyImport(message="Run 'pip install transformers[torch,sentencepiece]'") as torch_and_transformers_import:
    import accelerate  # pylint: disable=unused-import # the library is used but not directly referenced
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


@component
class TransformersSimilarityRanker:
    """
    Ranks documents based on their semantic similarity to the query.

    It uses a pre-trained cross-encoder model from Hugging Face to embed the query and the documents.

    Note:
    This component is considered legacy and will no longer receive updates. It may be deprecated in a future release,
    with removal following after a deprecation period.
    Consider using SentenceTransformersSimilarityRanker instead, which provides the same functionality along with
    additional features.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.rankers import TransformersSimilarityRanker

    ranker = TransformersSimilarityRanker()
    docs = [Document(content="Paris"), Document(content="Berlin")]
    query = "City in Germany"
    ranker.warm_up()
    result = ranker.run(query=query, documents=docs)
    docs = result["documents"]
    print(docs[0].content)
    ```
    """

    def __init__(  # noqa: PLR0913, pylint: disable=too-many-positional-arguments
        self,
        model: Union[str, Path] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        top_k: int = 10,
        query_prefix: str = "",
        document_prefix: str = "",
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        scale_score: bool = True,
        calibration_factor: Optional[float] = 1.0,
        score_threshold: Optional[float] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        batch_size: int = 16,
    ):
        """
        Creates an instance of TransformersSimilarityRanker.

        :param model:
            The ranking model. Pass a local path or the Hugging Face model name of a cross-encoder model.
        :param device:
            The device on which the model is loaded. If `None`, overrides the default device.
        :param token:
            The API token to download private models from Hugging Face.
        :param top_k:
            The maximum number of documents to return per query.
        :param query_prefix:
            A string to add at the beginning of the query text before ranking.
            Use it to prepend the text with an instruction, as required by reranking models like `bge`.
        :param document_prefix:
            A string to add at the beginning of each document before ranking. You can use it to prepend the document
            with an instruction, as required by embedding models like `bge`.
        :param meta_fields_to_embed:
            List of metadata fields to embed with the document.
        :param embedding_separator:
            Separator to concatenate metadata fields to the document.
        :param scale_score:
            If `True`, scales the raw logit predictions using a Sigmoid activation function.
            If `False`, disables scaling of the raw logit predictions.
        :param calibration_factor:
            Use this factor to calibrate probabilities with `sigmoid(logits * calibration_factor)`.
            Used only if `scale_score` is `True`.
        :param score_threshold:
            Use it to return documents with a score above this threshold only.
        :param model_kwargs:
            Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
            when loading the model. Refer to specific model documentation for available kwargs.
        :param tokenizer_kwargs:
            Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
            Refer to specific model documentation for available kwargs.
        :param batch_size:
            The batch size to use for inference. The higher the batch size, the more memory is required.
            If you run into memory issues, reduce the batch size.

        :raises ValueError:
            If `top_k` is not > 0.
            If `scale_score` is True and `calibration_factor` is not provided.
        """
        torch_and_transformers_import.check()

        soft_deprecation_message = (
            "TransformersSimilarityRanker is considered legacy and will no longer receive updates. "
            "It may be deprecated in a future release, with removal following after a deprecation period. "
            "Consider using SentenceTransformersSimilarityRanker instead, which provides the same functionality "
            "along with additional features."
        )
        logger.warning(soft_deprecation_message)

        self.model_name_or_path = str(model)
        self.model = None
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.tokenizer = None
        self.device = None
        self.top_k = top_k
        self.token = token
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.scale_score = scale_score
        self.calibration_factor = calibration_factor
        self.score_threshold = score_threshold

        model_kwargs = resolve_hf_device_map(device=device, model_kwargs=model_kwargs)
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs or {}
        self.batch_size = batch_size

        # Parameter validation
        if self.scale_score and self.calibration_factor is None:
            raise ValueError(
                f"scale_score is True so calibration_factor must be provided, but got {calibration_factor}"
            )

        if self.top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model_name_or_path}

    def warm_up(self):
        """
        Initializes the component.
        """
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path, token=self.token.resolve_value() if self.token else None, **self.model_kwargs
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name_or_path,
                token=self.token.resolve_value() if self.token else None,
                **self.tokenizer_kwargs,
            )
            assert self.model is not None
            self.device = ComponentDevice.from_multiple(device_map=DeviceMap.from_hf(self.model.hf_device_map))

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialization_dict = default_to_dict(
            self,
            device=None,
            model=self.model_name_or_path,
            token=self.token.to_dict() if self.token else None,
            top_k=self.top_k,
            query_prefix=self.query_prefix,
            document_prefix=self.document_prefix,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            scale_score=self.scale_score,
            calibration_factor=self.calibration_factor,
            score_threshold=self.score_threshold,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
            batch_size=self.batch_size,
        )

        serialize_hf_model_kwargs(serialization_dict["init_parameters"]["model_kwargs"])
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TransformersSimilarityRanker":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        init_params = data["init_parameters"]
        if init_params.get("device") is not None:
            init_params["device"] = ComponentDevice.from_dict(init_params["device"])
        if init_params.get("model_kwargs") is not None:
            deserialize_hf_model_kwargs(init_params["model_kwargs"])
        deserialize_secrets_inplace(init_params, keys=["token"])

        return default_from_dict(cls, data)

    @component.output_types(documents=List[Document])
    def run(  # pylint: disable=too-many-positional-arguments
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        calibration_factor: Optional[float] = None,
        score_threshold: Optional[float] = None,
    ):
        """
        Returns a list of documents ranked by their similarity to the given query.

        :param query:
            The input query to compare the documents to.
        :param documents:
            A list of documents to be ranked.
        :param top_k:
            The maximum number of documents to return.
        :param scale_score:
            If `True`, scales the raw logit predictions using a Sigmoid activation function.
            If `False`, disables scaling of the raw logit predictions.
        :param calibration_factor:
            Use this factor to calibrate probabilities with `sigmoid(logits * calibration_factor)`.
            Used only if `scale_score` is `True`.
        :param score_threshold:
            Use it to return documents only with a score above this threshold.
        :returns:
            A dictionary with the following keys:
            - `documents`: A list of documents closest to the query, sorted from most similar to least similar.

        :raises ValueError:
            If `top_k` is not > 0.
            If `scale_score` is True and `calibration_factor` is not provided.
        :raises RuntimeError:
            If the model is not loaded because `warm_up()` was not called before.
        """
        # If a model path is provided but the model isn't loaded
        if self.model is None:
            raise RuntimeError(
                "The component TransformersSimilarityRanker wasn't warmed up. Run 'warm_up()' before calling 'run()'."
            )

        if not documents:
            return {"documents": []}

        top_k = top_k or self.top_k
        scale_score = scale_score or self.scale_score
        calibration_factor = calibration_factor or self.calibration_factor
        score_threshold = score_threshold or self.score_threshold

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        if scale_score and calibration_factor is None:
            raise ValueError(
                f"scale_score is True so calibration_factor must be provided, but got {calibration_factor}"
            )

        query_doc_pairs = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key]
            ]
            text_to_embed = self.embedding_separator.join(meta_values_to_embed + [doc.content or ""])
            query_doc_pairs.append([self.query_prefix + query, self.document_prefix + text_to_embed])

        class _Dataset(Dataset):
            def __init__(self, batch_encoding):
                self.batch_encoding = batch_encoding

            def __len__(self):
                return len(self.batch_encoding["input_ids"])

            def __getitem__(self, item):
                return {key: self.batch_encoding.data[key][item] for key in self.batch_encoding.data.keys()}

        batch_enc = self.tokenizer(query_doc_pairs, padding=True, truncation=True, return_tensors="pt").to(  # type: ignore
            self.device.first_device.to_torch()
        )
        dataset = _Dataset(batch_enc)
        inp_dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        similarity_scores = []
        with torch.inference_mode():
            for features in inp_dataloader:
                model_preds = self.model(**features).logits.squeeze(dim=1)  # type: ignore
                similarity_scores.extend(model_preds)
        similarity_scores = torch.stack(similarity_scores)

        if scale_score:
            similarity_scores = torch.sigmoid(similarity_scores * calibration_factor)

        _, sorted_indices = torch.sort(similarity_scores, descending=True)

        sorted_indices = sorted_indices.cpu().tolist()  # type: ignore
        similarity_scores = similarity_scores.cpu().tolist()
        ranked_docs = []
        for sorted_index in sorted_indices:
            i = sorted_index
            documents[i].score = similarity_scores[i]
            ranked_docs.append(documents[i])

        if score_threshold is not None:
            ranked_docs = [doc for doc in ranked_docs if doc.score >= score_threshold]

        return {"documents": ranked_docs[:top_k]}
