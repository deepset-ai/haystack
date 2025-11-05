# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Literal, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.components.embedders.backends.sentence_transformers_sparse_backend import (
    _SentenceTransformersSparseEmbeddingBackendFactory,
    _SentenceTransformersSparseEncoderEmbeddingBackend,
)
from haystack.dataclasses.sparse_embedding import SparseEmbedding
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace
from haystack.utils.hf import deserialize_hf_model_kwargs, serialize_hf_model_kwargs


@component
class SentenceTransformersSparseTextEmbedder:
    """
    Embeds strings using sparse embedding models from Sentence Transformers.

    You can use it to embed user query and send it to a sparse embedding retriever.

    Usage example:
    ```python
    from haystack.components.embedders import SentenceTransformersSparseTextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = SentenceTransformersSparseTextEmbedder()
    text_embedder.warm_up()

    print(text_embedder.run(text_to_embed))

    # {'sparse_embedding': SparseEmbedding(indices=[999, 1045, ...], values=[0.918, 0.867, ...])}
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
        trust_remote_code: bool = False,
        local_files_only: bool = False,
        model_kwargs: Optional[dict[str, Any]] = None,
        tokenizer_kwargs: Optional[dict[str, Any]] = None,
        config_kwargs: Optional[dict[str, Any]] = None,
        encode_kwargs: Optional[dict[str, Any]] = None,
        backend: Literal["torch", "onnx", "openvino"] = "torch",
        revision: Optional[str] = None,
    ):
        """
        Create a SentenceTransformersSparseTextEmbedder component.

        :param model:
            The model to use for calculating sparse embeddings.
            Specify the path to a local model or the ID of the model on Hugging Face.
        :param device:
            Overrides the default device used to load the model.
        :param token:
            An API token to use private models from Hugging Face.
        :param prefix:
            A string to add at the beginning of each text to be embedded.
        :param suffix:
            A string to add at the end of each text to embed.
        :param trust_remote_code:
            If `False`, permits only Hugging Face verified model architectures.
            If `True`, permits custom models and scripts.
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
        :param revision:
            The specific model version to use. It can be a branch name, a tag name, or a commit id,
            for a stored model on Hugging Face.
        """

        self.model = model
        self.device = ComponentDevice.resolve_device(device)
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
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
    def from_dict(cls, data: dict[str, Any]) -> "SentenceTransformersSparseTextEmbedder":
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

    @component.output_types(sparse_embedding=SparseEmbedding)
    def run(self, text: str):
        """
        Embed a single string.

        :param text:
            Text to embed.

        :returns:
            A dictionary with the following keys:
            - `sparse_embedding`: The sparse embedding of the input text.
        """
        if not isinstance(text, str):
            raise TypeError(
                "SentenceTransformersSparseTextEmbedder expects a string as input."
                "In case you want to embed a list of Documents, please use the"
                "SentenceTransformersSparseDocumentEmbedder."
            )
        if self.embedding_backend is None:
            raise RuntimeError("The embedding model has not been loaded. Please call warm_up() before running.")

        text_to_embed = self.prefix + text + self.suffix

        sparse_embedding = self.embedding_backend.embed(data=[text_to_embed])[0]

        return {"sparse_embedding": sparse_embedding}
