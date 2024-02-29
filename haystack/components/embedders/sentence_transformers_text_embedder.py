from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from haystack.components.embedders.backends.sentence_transformers_backend import (
    _SentenceTransformersEmbeddingBackendFactory,
)
from haystack.utils import ComponentDevice, Secret, deserialize_secrets_inplace


@component
class SentenceTransformersTextEmbedder:
    """
    A component for embedding strings using Sentence Transformers models.

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

    def __init__(
        self,
        model: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[ComponentDevice] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        prefix: str = "",
        suffix: str = "",
        batch_size: int = 32,
        progress_bar: bool = True,
        normalize_embeddings: bool = False,
    ):
        """
        Create a SentenceTransformersTextEmbedder component.

        :param model: Local path or name of the model in Hugging Face's model hub,
            such as ``'sentence-transformers/all-mpnet-base-v2'``.
        :param device: The device on which the model is loaded. If `None`, the default device is automatically
            selected.
        :param token: The API token used to download private models from Hugging Face.
        :param prefix: A string to add to the beginning of each Document text before embedding.
            Can be used to prepend the text with an instruction, as required by some embedding models,
            such as E5 and bge.
        :param suffix: A string to add to the end of each text.
        :param batch_size: Number of strings to encode at once.
        :param progress_bar: If true, displays progress bar during embedding.
        :param normalize_embeddings: If set to true, returned vectors will have length 1.
        """

        self.model = model
        self.device = ComponentDevice.resolve_device(device)
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.normalize_embeddings = normalize_embeddings

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        return {"model": self.model}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize this component to a dictionary.
        """
        return default_to_dict(
            self,
            model=self.model,
            device=self.device.to_dict(),
            token=self.token.to_dict() if self.token else None,
            prefix=self.prefix,
            suffix=self.suffix,
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceTransformersTextEmbedder":
        serialized_device = data["init_parameters"]["device"]
        data["init_parameters"]["device"] = ComponentDevice.from_dict(serialized_device)

        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        return default_from_dict(cls, data)

    def warm_up(self):
        """
        Load the embedding backend.
        """
        if not hasattr(self, "embedding_backend"):
            self.embedding_backend = _SentenceTransformersEmbeddingBackendFactory.get_embedding_backend(
                model=self.model, device=self.device.to_torch_str(), auth_token=self.token
            )

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """Embed a string."""
        if not isinstance(text, str):
            raise TypeError(
                "SentenceTransformersTextEmbedder expects a string as input."
                "In case you want to embed a list of Documents, please use the SentenceTransformersDocumentEmbedder."
            )
        if not hasattr(self, "embedding_backend"):
            raise RuntimeError("The embedding model has not been loaded. Please call warm_up() before running.")

        text_to_embed = self.prefix + text + self.suffix
        embedding = self.embedding_backend.embed(
            [text_to_embed],
            batch_size=self.batch_size,
            show_progress_bar=self.progress_bar,
            normalize_embeddings=self.normalize_embeddings,
        )[0]
        return {"embedding": embedding}
