# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from typing import Any, Literal, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.components.embedders.backends.sentence_transformers_sparse_backend import (
    _SentenceTransformersSparseEmbeddingBackendFactory,
    _SentenceTransformersSparseEncoderEmbeddingBackend,
)
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace
from haystack.utils.hf import deserialize_hf_model_kwargs, serialize_hf_model_kwargs


@component
class SentenceTransformersSparseDocumentEmbedder:
    """
    Calculates document sparse embeddings using sparse embedding models from Sentence Transformers.

    It stores the sparse embeddings in the `sparse_embedding` metadata field of each document.
    You can also embed documents' metadata.
    Use this component in indexing pipelines to embed input documents
    and send them to DocumentWriter to write a into a Document Store.

    ### Usage example:

    ```python
    from haystack import Document
    from haystack.components.embedders import SentenceTransformersSparseDocumentEmbedder

    doc = Document(content="I love pizza!")
    doc_embedder = SentenceTransformersSparseDocumentEmbedder()
    doc_embedder.warm_up()

    result = doc_embedder.run([doc])
    print(result['documents'][0].sparse_embedding)

    # SparseEmbedding(indices=[999, 1045, ...], values=[0.918, 0.867, ...])
    ```
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model: str = "prithivida/Splade_PP_en_v2",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[list[str]] = None,
        embedding_separator: str = "\n",
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        model_kwargs: Optional[dict[str, Any]] = None,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
        config_kwargs: Optional[dict[str, Any]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
    ):
        """
        Creates a SentenceTransformersSparseDocumentEmbedder component.

        :param model:
            The model to use for calculating sparse embeddings.
            Pass a local path or ID of the model on Hugging Face.
        :param device:
            The device to use for loading the model.
            Overrides the default device.
        :param token:
            The API token to download private models from Hugging Face.
        :param prefix:
            A string to add at the beginning of each document text.
        :param suffix:
            A string to add at the end of each document text.
        :param batch_size:
            Number of documents to embed at once.
        :param progress_bar:
            If `True`, shows a progress bar when embedding documents.
        :param meta_fields_to_embed:
            List of metadata fields to embed along with the document text.
        :param embedding_separator:
            Separator used to concatenate the metadata fields to the document text.
        :param trust_remote_code:
            If `False`, allows only Hugging Face verified model architectures.
            If `True`, allows custom models and scripts.
        :param revision:
            The specific model version to use. It can be a branch name, a tag name, or a commit id,
            for a stored model on Hugging Face.
        :param local_files_only:
            If `True`, does not attempt to download the model from Hugging Face Hub and only looks at local files.
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
        """

        self.model = model
        self.device = ComponentDevice.resolve_device(device)
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self.local_files_only = local_files_only
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.config_kwargs = config_kwargs
        self.embedding_backend: Optional[_SentenceTransformersSparseEncoderEmbeddingBackend] = None
        self.backend = backend

    def _get_telemetry_data(self) -> dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        serialization_dict = default_to_dict(
            self,
            model=self.model,
            device=self.device.to_dict(),
            token=self.token.to_dict() if self.token else None,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision,
            local_files_only=self.local_files_only,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
            config_kwargs=self.config_kwargs,
            backend=self.backend,
        )
        if serialization_dict["init_parameters"].get("model_kwargs") is not None:
            serialize_hf_model_kwargs(serialization_dict["init_parameters"]["model_kwargs"])
        return serialization_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SentenceTransformersSparseDocumentEmbedder":
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
        deserialize_secrets_inplace(init_params, keys=["token"])
        if init_params.get("model_kwargs") is not None:
            deserialize_hf_model_kwargs(init_params["model_kwargs"])
        return default_from_dict(cls, data)

    def warm_up(self):
        """
        Initializes the component.
        """
        if self.embedding_backend is None:
            self.embedding_backend = _SentenceTransformersSparseEmbeddingBackendFactory.get_embedding_backend(
                model=self.model,
                device=self.device.to_torch_str(),
                auth_token=self.token,
                trust_remote_code=self.trust_remote_code,
                revision=self.revision,
                local_files_only=self.local_files_only,
                model_kwargs=self.model_kwargs,
                tokenizer_kwargs=self.tokenizer_kwargs,
                config_kwargs=self.config_kwargs,
                backend=self.backend,
            )
            if self.tokenizer_kwargs and self.tokenizer_kwargs.get("model_max_length"):
                self.embedding_backend.model.max_seq_length = self.tokenizer_kwargs["model_max_length"]

    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        """
        Embed a list of documents.

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: Documents with sparse embeddings under the `sparse_embedding` field.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "SentenceTransformersSparseDocumentEmbedder expects a list of Documents as input."
                "In case you want to embed a list of strings, please use the SentenceTransformersSparseTextEmbedder."
            )
        if self.embedding_backend is None:
            raise RuntimeError("The embedding model has not been loaded. Please call warm_up() before running.")

        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key]
            ]
            text_to_embed = (
                self.prefix + self.embedding_separator.join(meta_values_to_embed + [doc.content or ""]) + self.suffix
            )
            texts_to_embed.append(text_to_embed)

        embeddings = self.embedding_backend.embed(
            data=texts_to_embed, batch_size=self.batch_size, show_progress_bar=self.progress_bar
        )

        documents_with_embeddings = []
        for doc, emb in zip(documents, embeddings):
            documents_with_embeddings.append(replace(doc, sparse_embedding=emb))

        return {"documents": documents_with_embeddings}
