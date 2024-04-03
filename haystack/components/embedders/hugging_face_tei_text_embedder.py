import warnings
from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.hf import HFModelType, check_valid_model
from haystack.utils.url_validation import is_valid_url

with LazyImport(message="Run 'pip install \"huggingface_hub>=0.22.0\"'") as huggingface_hub_import:
    from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


# TODO: remove the default model in Haystack 2.3.0, as explained in the deprecation warning
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


@component
class HuggingFaceTEITextEmbedder:
    """
    A component for embedding strings using HuggingFace Text-Embeddings-Inference endpoints.

    This component can be used with embedding models hosted on Hugging Face Inference endpoints,
    on the rate-limited Hugging Face Inference API or on your own custom TEI endpoint.

    Usage example:
    ```python
    from haystack.components.embedders import HuggingFaceTEITextEmbedder
    from haystack.utils import Secret

    text_to_embed = "I love pizza!"

    text_embedder = HuggingFaceTEITextEmbedder(
        model="BAAI/bge-small-en-v1.5", token=Secret.from_token("<your-api-key>")
    )

    print(text_embedder.run(text_to_embed))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    ```
    """

    def __init__(
        self,
        model: Optional[str] = None,
        url: Optional[str] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        prefix: str = "",
        suffix: str = "",
    ):
        """
        Create an HuggingFaceTEITextEmbedder component.

        :param model:
            An optional string representing the ID of the model on HuggingFace Hub.
            If not provided, the `url` parameter must be set to a valid TEI endpoint.
            In case both `model` and `url` are provided, the `url` parameter will be used.
        :param url:
            An optional string representing the URL of your self-deployed Text-Embeddings-Inference service
            or the URL of your paid HF Inference Endpoint.
            If not provided, the `model` parameter must be set to a valid model ID and the Hugging Face Inference API
            will be used.
            In case both `model` and `url` are provided, the `url` parameter will be used.
        :param token:
            The HuggingFace Hub token. This is needed if you are using a paid HF Inference Endpoint or serving
            a private or gated model.
        :param prefix:
            A string to add at the beginning of each text.
        :param suffix:
            A string to add at the end of each text.
        """
        huggingface_hub_import.check()

        if not model and not url:
            warnings.warn(
                f"Neither `model` nor `url` is provided. The component will use the default model: {DEFAULT_MODEL}. "
                "This behavior is deprecated and will be removed in Haystack 2.3.0.",
                DeprecationWarning,
            )
            model = DEFAULT_MODEL
        elif model and url:
            logger.warning("Both `model` and `url` are provided. The `model` parameter will be ignored. ")

        if url and not is_valid_url(url):
            raise ValueError(f"Invalid TEI endpoint URL provided: {url}")
        if not url and model:
            check_valid_model(model, HFModelType.EMBEDDING, token)

        self.model = model
        self.url = url
        self.token = token
        self.client = InferenceClient(url or model, token=token.resolve_value() if token else None)
        self.prefix = prefix
        self.suffix = suffix

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            model=self.model,
            url=self.url,
            prefix=self.prefix,
            suffix=self.suffix,
            token=self.token.to_dict() if self.token else None,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceTEITextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        return default_from_dict(cls, data)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        # Don't send URL as it is sensitive information
        return {"model": self.model}

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
                "HuggingFaceTEITextEmbedder expects a string as an input."
                "In case you want to embed a list of Documents, please use the HuggingFaceTEIDocumentEmbedder."
            )

        text_to_embed = self.prefix + text + self.suffix

        embeddings = self.client.feature_extraction(text=[text_to_embed])
        # The client returns a numpy array
        embedding = embeddings.tolist()[0]

        return {"embedding": embedding}
