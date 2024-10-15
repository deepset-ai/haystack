# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Literal, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.components.embedders.backends.sentence_transformers_backend import (
    _SentenceTransformersEmbeddingBackendFactory,
)
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace
from haystack.utils.hf import deserialize_hf_model_kwargs, serialize_hf_model_kwargs


@component
class SentenceTransformersTextEmbedder:
    """
    Embeds strings using Sentence Transformers models.

    You can use it to embed user query and send it to an embedding retriever.

    Usage example:
    ```python
    from haystack.components.embedders import SentenceTransformersTextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = SentenceTransformersTextEmbedder()
    text_embedder.warm_up()

    print(text_embedder.run(text_to_embed))

    # {'embedding': [-0.07804739475250244, 0.1498992145061493,, ...]}
    ```
    """

    def __init__(  # noqa: PLR0913
        self,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        normalize_embeddings: bool = False,
        trust_remote_code: bool = False,
        truncate_dim: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] = "float32",
    ):
        """
        Create a SentenceTransformersTextEmbedder component.

        :param model:
            The model to use for calculating embeddings.
            Specify the path to a local model or the ID of the model on Hugging Face.
        :param device:
            Overrides the default device used to load the model.
        :param token:
            An API token to use private models from Hugging Face.
        :param prefix:
            A string to add at the beginning of each text to be embedded.
            You can use it to prepend the text with an instruction, as required by some embedding models,
            such as E5 and bge.
        :param suffix:
            A string to add at the end of each text to embed.
        :param batch_size:
            Number of texts to embed at once.
        :param progress_bar:
            If `True`, shows a progress bar for calculating embeddings.
            If `False`, disables the progress bar.
        :param normalize_embeddings:
            If `True`, returned vectors have a length of 1.
        :param trust_remote_code:
            If `False`, permits only Hugging Face verified model architectures.
            If `True`, permits custom models and scripts.
        :param truncate_dim:
            The dimension to truncate sentence embeddings to. `None` does no truncation.
            If the model has not been trained with Matryoshka Representation Learning,
            truncation of embeddings can significantly affect performance.
        :param model_kwargs:
            Additional keyword arguments for `AutoModelForSequenceClassification.from_pretrained`
            when loading the model. Refer to specific model documentation for available kwargs.
        :param tokenizer_kwargs:
            Additional keyword arguments for `AutoTokenizer.from_pretrained` when loading the tokenizer.
            Refer to specific model documentation for available kwargs.
        :param config_kwargs:
            Additional keyword arguments for `AutoConfig.from_pretrained` when loading the model configuration.
        :param precision:
            The precision to use for the embeddings.
            All non-float32 precisions are quantized embeddings.
            Quantized embeddings are smaller in size and faster to compute, but may have a lower accuracy.
            They are useful for reducing the size of the embeddings of a corpus for semantic search, among other tasks.
        """

        self.model = model
        self.device = ComponentDevice.resolve_device(device)
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize_embeddings = normalize_embeddings
        self.trust_remote_code = trust_remote_code
        self.truncate_dim = truncate_dim
        self.model_kwargs = model_kwargs
        self.tokenizer_kwargs = tokenizer_kwargs
        self.config_kwargs = config_kwargs
        self.embedding_backend = None
        self.precision = precision

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
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
            normalize_embeddings=self.normalize_embeddings,
            trust_remote_code=self.trust_remote_code,
            truncate_dim=self.truncate_dim,
            model_kwargs=self.model_kwargs,
            tokenizer_kwargs=self.tokenizer_kwargs,
            config_kwargs=self.config_kwargs,
            precision=self.precision,
        )
        if serialization_dict["init_parameters"].get("model_kwargs") is not None:
            serialize_hf_model_kwargs(serialization_dict["init_parameters"]["model_kwargs"])
        return serialization_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceTransformersTextEmbedder":
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
            self.embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
                model=self.model,
                device=self.device.to_torch_str(),
                auth_token=self.token,
                trust_remote_code=self.trust_remote_code,
                truncate_dim=self.truncate_dim,
                model_kwargs=self.model_kwargs,
                tokenizer_kwargs=self.tokenizer_kwargs,
                config_kwargs=self.config_kwargs,
            )
            if self.tokenizer_kwargs and self.tokenizer_kwargs.get("model_max_length"):
                self.embedding_backend.model.max_seq_length = self.tokenizer_kwargs["model_max_length"]

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """
        Embed a single string.

        :param text:
            Text to embed.

        :returns:
            A dictionary with the following keys:
            - `embedding`: The embedding of the input text.
        """
        if not isinstance(text, str):
            raise TypeError(
                "SentenceTransformersTextEmbedder expects a string as input."
                "In case you want to embed a list of Documents, please use the SentenceTransformersDocumentEmbedder."
            )
        if self.embedding_backend is None:
            raise RuntimeError("The embedding model has not been loaded. Please call warm_up() before running.")

        text_to_embed = self.prefix + text + self.suffix
        embedding = self.embedding_backend.embed(
            [text_to_embed],
            batch_size=self.batch_size,
            show_progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            precision=self.precision,
        )[0]
        return {"embedding": embedding}
