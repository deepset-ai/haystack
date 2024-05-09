# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
import warnings
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.hf import HFModelType, check_valid_model

with LazyImport(message="Run 'pip install \"huggingface_hub>=0.23.0\"'") as huggingface_hub_import:
    from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


@component
class HuggingFaceTEITextEmbedder:
    """
    A component for embedding strings using HuggingFace Text-Embeddings-Inference endpoints.

    This component can be used with embedding models hosted on Hugging Face Inference endpoints, the rate-limited
    Inference API tier, for embedding models hosted on [the paid inference endpoint](https://huggingface.co/inference-endpoints)
    and/or your own custom TEI endpoint.

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
        model: str = "BAAI/bge-small-en-v1.5",
        url: Optional[str] = None,
        token: Optional[Secret] = Secret.from_env_var("HF_API_TOKEN", strict=False),
        prefix: str = "",
        suffix: str = "",
        truncate: bool = True,
        normalize: bool = False,
    ):
        """
        Create an HuggingFaceTEITextEmbedder component.

        :param model:
            ID of the model on HuggingFace Hub.
        :param url:
            The URL of your self-deployed Text-Embeddings-Inference service or the URL of your paid HF Inference
            Endpoint.
        :param token:
            The HuggingFace Hub token. This is needed if you are using a paid HF Inference Endpoint or serving
            a private or gated model.
        :param prefix:
            A string to add at the beginning of each text.
        :param suffix:
            A string to add at the end of each text.
        :param truncate:
            Truncate input text from the end to the maximum length supported by the model. This option is only available
            for self-deployed Text Embedding Inference (TEI) endpoints and paid HF Inference Endpoints deployed with
            TEI. It will be ignored when used with free HF Inference endpoints or paid HF Inference endpoints deployed
            without TEI.
        :param normalize:
            Normalize the embeddings to unit length. This option is only available for self-deployed Text Embedding
            Inference (TEI) endpoints and paid HF Inference Endpoints deployed with TEI. It will be ignored when used
            with free HF Inference endpoints or paid HF Inference endpoints deployed without TEI.
        """
        warnings.warn(
            "`HuggingFaceTEITextEmbedder` is deprecated and will be removed in Haystack 2.3.0."
            "Use `HuggingFaceAPITextEmbedder` instead.",
            DeprecationWarning,
        )

        huggingface_hub_import.check()

        if url:
            r = urlparse(url)
            is_valid_url = all([r.scheme in ["http", "https"], r.netloc])
            if not is_valid_url:
                raise ValueError(f"Invalid TEI endpoint URL provided: {url}")

        check_valid_model(model, HFModelType.EMBEDDING, token)

        self.model = model
        self.url = url
        self.token = token
        self.client = InferenceClient(url or model, token=token.resolve_value() if token else None)
        self.prefix = prefix
        self.suffix = suffix
        self.truncate = truncate
        self.normalize = normalize

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
            truncate=self.truncate,
            normalize=self.normalize,
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

        response = self.client.post(
            json={"inputs": [text_to_embed], "truncate": self.truncate, "normalize": self.normalize},
            task="feature-extraction",
        )
        embedding = json.loads(response.decode())[0]

        return {"embedding": embedding}
