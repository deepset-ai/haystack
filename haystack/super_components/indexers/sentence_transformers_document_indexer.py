# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Literal, Optional

from haystack import Pipeline, component, default_from_dict, default_to_dict
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack.core.super_component import SuperComponent
from haystack.document_stores.types import DocumentStore, DuplicatePolicy
from haystack.utils import (
    ComponentDevice,
    Secret,
    deserialize_document_store_in_init_params_inplace,
    deserialize_secrets_inplace,
)
from haystack.utils.hf import deserialize_hf_model_kwargs, serialize_hf_model_kwargs


@component
class SentenceTransformersDocumentIndexer(SuperComponent):
    """
    A document indexer that takes a list of documents, embeds them using SentenceTransformers, and stores them.

    Usage:

    ```python
    >>> from haystack import Document
    >>> from haystack.document_stores.in_memory import InMemoryDocumentStore
    >>> document_store = InMemoryDocumentStore()
    >>> doc = Document(content="I love pizza!")
    >>> indexer = SentenceTransformersDocumentIndexer(document_store=document_store)
    >>> indexer.warm_up()
    >>> result = indexer.run(documents=[doc])
    >>> print(result)
    {'documents_written': 1}
    >>> document_store.count_documents()
    1
    ```
    """

    def __init__(  # noqa: PLR0913 (too-many-arguments) # pylint: disable=too-many-positional-arguments
        self,
        document_store: DocumentStore,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        normalize_embeddings: bool = False,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
        trust_remote_code: bool = False,
        truncate_dim: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
        duplicate_policy: DuplicatePolicy = DuplicatePolicy.OVERWRITE,
    ) -> None:
        """
        Initialize the SentenceTransformersDocumentIndexer component.

        :param document_store: The document store where the documents should be stored.
        :param model: The embedding model to use (local path or Hugging Face model ID).
        :param device: The device to use for loading the model.
        :param token: The API token to download private models from Hugging Face.
        :param prefix: String to add at the beginning of each document text.
        :param suffix: String to add at the end of each document text.
        :param batch_size: Number of documents to embed at once.
        :param progress_bar: If True, shows a progress bar when embedding documents.
        :param normalize_embeddings: If True, embeddings are L2 normalized.
        :param meta_fields_to_embed: List of metadata fields to embed along with the document text.
        :param embedding_separator: Separator used to concatenate metadata fields to document text.
        :param trust_remote_code: If True, allows custom models and scripts.
        :param truncate_dim: Dimension to truncate sentence embeddings to.
        :param model_kwargs: Additional keyword arguments for model initialization.
        :param tokenizer_kwargs: Additional keyword arguments for tokenizer initialization.
        :param config_kwargs: Additional keyword arguments for model configuration.
        :param precision: The precision to use for the embeddings.
        :param duplicate_policy: The duplicate policy to use when writing documents.
        """
        self.document_store = document_store
        self.model = model
        self.device = device
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize_embeddings = normalize_embeddings
        self.meta_fields_to_embed = meta_fields_to_embed
        self.embedding_separator = embedding_separator
        self.trust_remote_code = trust_remote_code
        self.truncate_dim = truncate_dim
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.config_kwargs = config_kwargs
        self.precision = precision
        self.duplicate_policy = duplicate_policy

        pipeline = Pipeline()

        pipeline.add_component(
            "embedder",
            SentenceTransformersDocumentEmbedder(
                model=self.model,
                device=self.device,
                token=self.token,
                prefix=self.prefix,
                suffix=self.suffix,
                batch_size=self.batch_size,
                progress_bar=self.progress_bar,
                normalize_embeddings=self.normalize_embeddings,
                meta_fields_to_embed=self.meta_fields_to_embed,
                embedding_separator=self.embedding_separator,
                trust_remote_code=self.trust_remote_code,
                truncate_dim=self.truncate_dim,
                model_kwargs=self.model_kwargs,
                tokenizer_kwargs=self.tokenizer_kwargs,
                config_kwargs=self.config_kwargs,
                precision=self.precision,
            ),
        )
        pipeline.add_component(
            "writer", DocumentWriter(document_store=self.document_store, policy=self.duplicate_policy)
        )

        pipeline.connect("embedder.documents", "writer.documents")

        super(SentenceTransformersDocumentIndexer, self).__init__(
            pipeline=pipeline,
            input_mapping={"documents": ["embedder.documents"]},
            output_mapping={"writer.documents_written": "documents_written"},
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this instance to a dictionary.
        """
        serialization_dict = default_to_dict(
            self,
            document_store=self.document_store.to_dict(),
            model=self.model,
            device=self.device.to_dict() if self.device else None,
            token=self.token.to_dict() if self.token else None,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            trust_remote_code=self.trust_remote_code,
            truncate_dim=self.truncate_dim,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
            config_kwargs=self.config_kwargs,
            precision=self.precision,
            duplicate_policy=self.duplicate_policy.value,
        )

        if serialization_dict["init_parameters"].get("model_kwargs") is not None:
            serialize_hf_model_kwargs(serialization_dict["init_parameters"]["model_kwargs"])

        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceTransformersDocumentIndexer":
        """
        Load an instance of this component from a dictionary.
        """
        deserialize_document_store_in_init_params_inplace(data)
        init_params = data.get("init_parameters", {})

        # Handle device deserialization
        if init_params.get("device") is not None:
            init_params["device"] = ComponentDevice.from_dict(init_params["device"])

        # Handle secrets deserialization
        deserialize_secrets_inplace(init_params, keys=["token"])

        # Handle model kwargs deserialization
        if init_params.get("model_kwargs") is not None:
            deserialize_hf_model_kwargs(init_params["model_kwargs"])

        # Handle duplicate policy deserialization
        if policy_value := init_params.get("duplicate_policy"):
            init_params["duplicate_policy"] = DuplicatePolicy(policy_value)

        data["init_parameters"] = init_params
        return default_from_dict(cls, data)
