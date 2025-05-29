# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from copy import copy
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.lazy_imports import LazyImport
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace
from haystack.utils.hf import deserialize_hf_model_kwargs, serialize_hf_model_kwargs

with LazyImport(message="Run 'pip install \"sentence-transformers>=4.1.0\"'") as torch_and_sentence_transformers_import:
    from sentence_transformers import CrossEncoder
    from torch.nn import Identity, Sigmoid


@component
class SentenceTransformersSimilarityRanker:
    """
    Ranks documents based on their semantic similarity to the query.

    It uses a pre-trained cross-encoder model from Hugging Face to embed the query and the documents.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.rankers import SentenceTransformersSimilarityRanker

    ranker = SentenceTransformersSimilarityRanker()
    docs = [Document(content="Paris"), Document(content="Berlin")]
    query = "City in Germany"
    ranker.warm_up()
    result = ranker.run(query=query, documents=docs)
    docs = result["documents"]
    print(docs[0].content)
    ```
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model: Union[str, Path] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        top_k: int = 10,
        query_prefix: str = "",
        document_prefix: str = "",
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        scale_score: bool = True,
        score_threshold: Optional[float] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        batch_size: int = 16,
    ):
        """
        Creates an instance of SentenceTransformersSimilarityRanker.

        :param model:
            The ranking model. Pass a local path or the Hugging Face model name of a cross-encoder model.
        :param device:
            The device on which the model is loaded. If `None`, the default device is automatically selected.
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
        :param score_threshold:
            Use it to return documents with a score above this threshold only.
        :param model_kwargs:
            Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
            when loading the model. Refer to specific model documentation for available kwargs.
        :param tokenizer_kwargs:
            Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
            Refer to specific model documentation for available kwargs.
        :param config_kwargs:
            Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
        :param backend:
            The backend to use for the Sentence Transformers model. Choose from "torch", "onnx", or "openvino".
            Refer to the [Sentence Transformers documentation](https://sbert.net/docs/sentence_transformer/usage/efficiency.html)
            for more information on acceleration and quantization options.
        :param batch_size:
            The batch size to use for inference. The higher the batch size, the more memory is required.
            If you run into memory issues, reduce the batch size.

        :raises ValueError:
            If `top_k` is not > 0.
        """
        torch_and_sentence_transformers_import.check()

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        self.model = str(model)
        self._cross_encoder = None
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.device = ComponentDevice.resolve_device(device)
        self.top_k = top_k
        self.token = token
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.scale_score = scale_score
        self.score_threshold = score_threshold
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.config_kwargs = config_kwargs
        self.backend = backend
        self.batch_size = batch_size

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def warm_up(self) -> None:
        """
        Initializes the component.
        """
        if self._cross_encoder is None:
            self._cross_encoder = CrossEncoder(
                model_name_or_path=self.model,
                device=self.device.to_torch_str(),
                token=self.token.resolve_value() if self.token else None,
                model_kwargs=self.model_kwargs,
                tokenizer_kwargs=self.tokenizer_kwargs,
                config_kwargs=self.config_kwargs,
                backend=self.backend,
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialization_dict = default_to_dict(
            self,
            device=self.device.to_dict(),
            model=self.model,
            token=self.token.to_dict() if self.token else None,
            top_k=self.top_k,
            query_prefix=self.query_prefix,
            document_prefix=self.document_prefix,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            scale_score=self.scale_score,
            score_threshold=self.score_threshold,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
            config_kwargs=self.config_kwargs,
            backend=self.backend,
            batch_size=self.batch_size,
        )
        if serialization_dict["init_parameters"].get("model_kwargs") is not None:
            serialize_hf_model_kwargs(serialization_dict["init_parameters"]["model_kwargs"])
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceTransformersSimilarityRanker":
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
    def run(
        self,
        *,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        scale_score: Optional[bool] = None,
        score_threshold: Optional[float] = None,
    ) -> Dict[str, List[Document]]:
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
            If set, overrides the value set at initialization.
        :param score_threshold:
            Use it to return documents only with a score above this threshold.
            If set, overrides the value set at initialization.
        :returns:
            A dictionary with the following keys:
            - `documents`: A list of documents closest to the query, sorted from most similar to least similar.

        :raises ValueError:
            If `top_k` is not > 0.
        :raises RuntimeError:
            If the model is not loaded because `warm_up()` was not called before.
        """
        if self._cross_encoder is None:
            raise RuntimeError(
                "The component SentenceTransformersSimilarityRanker wasn't warmed up. "
                "Run 'warm_up()' before calling 'run()'."
            )

        if not documents:
            return {"documents": []}

        top_k = top_k or self.top_k
        scale_score = scale_score or self.scale_score
        score_threshold = score_threshold or self.score_threshold

        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, but got {top_k}")

        prepared_query = self.query_prefix + query
        prepared_documents = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key]
            ]
            prepared_documents.append(
                self.document_prefix + self.embedding_separator.join(meta_values_to_embed + [doc.content or ""])
            )

        activation_fn = Sigmoid() if scale_score else Identity()

        ranking_result = self._cross_encoder.rank(
            query=prepared_query,
            documents=prepared_documents,
            batch_size=self.batch_size,
            activation_fn=activation_fn,
            convert_to_numpy=True,
            return_documents=False,
        )

        ranked_docs = []
        for el in ranking_result:
            index = el["corpus_id"]
            score = el["score"]
            document = copy(documents[index])
            document.score = score
            ranked_docs.append(document)

        if score_threshold is not None:
            ranked_docs = [doc for doc in ranked_docs if doc.score >= score_threshold]

        return {"documents": ranked_docs[:top_k]}
