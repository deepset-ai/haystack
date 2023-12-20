import logging
import os
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from haystack import component, default_to_dict
from haystack.components.embedders.hf_utils import check_valid_model
from haystack.lazy_imports import LazyImport

with LazyImport(message="Run 'pip install transformers'") as transformers_import:
    from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


@component
class HuggingFaceTEITextEmbedder:
    """
    A component for embedding strings using HuggingFace Text-Embeddings-Inference endpoints. This component
    is designed to seamlessly inference models deployed on the Text Embeddings Inference (TEI) backend.

    You can use this component for embedding models hosted on Hugging Face Inference endpoints, the rate-limited
    Inference API tier:
    ```python
    from haystack.components.embedders import HuggingFaceTEITextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = HuggingFaceTEITextEmbedder(
        model="BAAI/bge-small-en-v1.5", token="<your-token>"
    )

    print(text_embedder.run(text_to_embed))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    ```

    Or for embedding models hosted on paid https://huggingface.co/inference-endpoints endpoint, and/or your own custom
    TEI endpoint. In these two cases, you'll need to provide the URL of the endpoint as well as a valid token:

    ```python
    from haystack.components.embedders import HuggingFaceTEITextEmbedder

    text_to_embed = "I love pizza!"

    text_embedder = HuggingFaceTEITextEmbedder(
        model="BAAI/bge-small-en-v1.5", url="<your-tei-endpoint-url>", token="<your-token>"
    )

    print(text_embedder.run(text_to_embed))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    ```

    Key Features and Compatibility:
        - **Primary Compatibility**: Designed to work seamlessly with any embedding model deployed using the TEI
        framework. For more information on TEI, visit https://github.com/huggingface/text-embeddings-inference.
        - **Hugging Face Inference Endpoints**: Supports inference of TEI embedding models deployed on Hugging Face
        Inference endpoints. For more details refer to https://huggingface.co/inference-endpoints.
        - **Inference API Support**: Supports inference of TEI embedding models hosted on the rate-limited Inference
        API tier. Learn more about the Inference API at: https://huggingface.co/inference-api
        Discover available embedding models using the following command:
        ```
        wget -qO- https://api-inference.huggingface.co/framework/sentence-transformers
        ```
        And simply use the model ID as the model parameter for this component. You'll also need to provide a valid
        Hugging Face API token as the token parameter.
        - **Custom TEI Endpoints**: Supports inference of embedding models deployed on custom TEI endpoints. Anyone can
        deploy their own TEI endpoint using the TEI framework. For more details refer
        to https://huggingface.co/inference-endpoints.
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        url: Optional[str] = None,
        token: Optional[str] = None,
        prefix: str = "",
        suffix: str = "",
    ):
        """
        Create an HuggingFaceTEITextEmbedder component.

        :param model: A string representing the model id on HF Hub. Default is "BAAI/bge-small-en-v1.5".
        :param url: The URL of your self-deployed Text-Embeddings-Inference service or the URL of your paid HF Inference
                    Endpoint.
        :param token: The HuggingFace Hub token. This is needed if you are using a paid HF Inference Endpoint or serving
                      a private or gated model. It can be explicitly provided or automatically read from the environment
                      variable HF_API_TOKEN (recommended).
        :param prefix: A string to add to the beginning of each text.
        :param suffix: A string to add to the end of each text.
        """
        transformers_import.check()

        if url:
            r = urlparse(url)
            is_valid_url = all([r.scheme in ["http", "https"], r.netloc])
            if not is_valid_url:
                raise ValueError(f"Invalid TEI endpoint URL provided: {url}")

        # The user does not need to provide a token if it is a local server or free public HF Inference Endpoint.
        token = token or os.environ.get("HF_API_TOKEN")

        check_valid_model(model, token)

        self.model = model
        self.url = url
        self.token = token
        self.client = InferenceClient(url or model, token=token)
        self.prefix = prefix
        self.suffix = suffix

    def to_dict(self) -> Dict[str, Any]:
        """
        This method overrides the default serializer in order to avoid leaking the `token` value passed
        to the constructor.
        """
        return default_to_dict(self, model=self.model, url=self.url, prefix=self.prefix, suffix=self.suffix)

    def _get_telemetry_data(self) -> Dict[str, Any]:
        """
        Data that is sent to Posthog for usage analytics.
        """
        # Don't send URL as it is sensitive information
        return {"model": self.model}

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """Embed a string."""
        if not isinstance(text, str):
            raise TypeError(
                "HuggingFaceTEITextEmbedder expects a string as an input."
                "In case you want to embed a list of Documents, please use the HuggingFaceTEIDocumentEmbedder."
            )

        text_to_embed = self.prefix + text + self.suffix

        embedding = self.client.feature_extraction(text=text_to_embed)
        # The client returns a numpy array
        embedding = embedding.tolist()

        return {"embedding": embedding}
