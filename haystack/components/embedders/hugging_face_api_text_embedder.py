# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any, Dict, List, Optional, Union

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.hf import HFEmbeddingAPIType, HFModelType, check_valid_model
from haystack.utils.url_validation import is_valid_http_url

with LazyImport(message="Run 'pip install \"huggingface_hub>=0.27.0\"'") as huggingface_hub_import:
    from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)


@component
class HuggingFaceAPITextEmbedder:
    """
    Embeds strings using Hugging Face APIs.

    Use it with the following Hugging Face APIs:
    - [Free Serverless Inference API](https://huggingface.co/inference-api)
    - [Paid Inference Endpoints](https://huggingface.co/inference-endpoints)
    - [Self-hosted Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)

    ### Usage examples

    #### With free serverless inference API

    ```python
    from haystack.components.embedders import HuggingFaceAPITextEmbedder
    from haystack.utils import Secret

    text_embedder = HuggingFaceAPITextEmbedder(api_type="serverless_inference_api",
                                               api_params={"model": "BAAI/bge-small-en-v1.5"},
                                               token=Secret.from_token("<your-api-key>"))

    print(text_embedder.run("I love pizza!"))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    ```

    #### With paid inference endpoints

    ```python
    from haystack.components.embedders import HuggingFaceAPITextEmbedder
    from haystack.utils import Secret
    text_embedder = HuggingFaceAPITextEmbedder(api_type="inference_endpoints",
                                               api_params={"model": "BAAI/bge-small-en-v1.5"},
                                               token=Secret.from_token("<your-api-key>"))

    print(text_embedder.run("I love pizza!"))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    ```

    #### With self-hosted text embeddings inference

    ```python
    from haystack.components.embedders import HuggingFaceAPITextEmbedder
    from haystack.utils import Secret

    text_embedder = HuggingFaceAPITextEmbedder(api_type="text_embeddings_inference",
                                               api_params={"url": "http://localhost:8080"})

    print(text_embedder.run("I love pizza!"))

    # {'embedding': [0.017020374536514282, -0.023255806416273117, ...],
    ```
    """

    def __init__(
        self,
        api_type: Union[HFEmbeddingAPIType, str],
        api_params: Dict[str, str],
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        prefix: str = "",
        suffix: str = "",
        truncate: bool = True,
        normalize: bool = False,
    ):  # pylint: disable=too-many-positional-arguments
        """
        Creates a HuggingFaceAPITextEmbedder component.

        :param api_type:
            The type of Hugging Face API to use.
        :param api_params:
            A dictionary with the following keys:
            - `model`: Hugging Face model ID. Required when `api_type` is `SERVERLESS_INFERENCE_API`.
            - `url`: URL of the inference endpoint. Required when `api_type` is `INFERENCE_ENDPOINTS` or
            `TEXT_EMBEDDINGS_INFERENCE`.
        :param token: The Hugging Face token to use as HTTP bearer authorization.
            Check your HF token in your [account settings](https://huggingface.co/settings/tokens).
        :param prefix:
            A string to add at the beginning of each text.
        :param suffix:
            A string to add at the end of each text.
        :param truncate:
            Truncates the input text to the maximum length supported by the model.
            Applicable when `api_type` is `TEXT_EMBEDDINGS_INFERENCE`, or `INFERENCE_ENDPOINTS`
            if the backend uses Text Embeddings Inference.
            If `api_type` is `SERVERLESS_INFERENCE_API`, this parameter is ignored.
            It is always set to `True` and cannot be changed.
        :param normalize:
            Normalizes the embeddings to unit length.
            Applicable when `api_type` is `TEXT_EMBEDDINGS_INFERENCE`, or `INFERENCE_ENDPOINTS`
            if the backend uses Text Embeddings Inference.
            If `api_type` is `SERVERLESS_INFERENCE_API`, this parameter is ignored.
            It is always set to `False` and cannot be changed.
        """
        huggingface_hub_import.check()

        if isinstance(api_type, str):
            api_type = HFEmbeddingAPIType.from_str(api_type)

        if api_type == HFEmbeddingAPIType.SERVERLESS_INFERENCE_API:
            model = api_params.get("model")
            if model is None:
                raise ValueError(
                    "To use the Serverless Inference API, you need to specify the `model` parameter in `api_params`."
                )
            check_valid_model(model, HFModelType.EMBEDDING, token)
            model_or_url = model
        elif api_type in [HFEmbeddingAPIType.INFERENCE_ENDPOINTS, HFEmbeddingAPIType.TEXT_EMBEDDINGS_INFERENCE]:
            url = api_params.get("url")
            if url is None:
                msg = (
                    "To use Text Embeddings Inference or Inference Endpoints, you need to specify the `url` "
                    "parameter in `api_params`."
                )
                raise ValueError(msg)
            if not is_valid_http_url(url):
                raise ValueError(f"Invalid URL: {url}")
            model_or_url = url
        else:
            msg = f"Unknown api_type {api_type}"
            raise ValueError(msg)

        self.api_type = api_type
        self.api_params = api_params
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
        self.truncate = truncate
        self.normalize = normalize
        self._client = InferenceClient(model_or_url, token=token.resolve_value() if token else None)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            api_type=str(self.api_type),
            api_params=self.api_params,
            prefix=self.prefix,
            suffix=self.suffix,
            token=self.token.to_dict() if self.token else None,
            truncate=self.truncate,
            normalize=self.normalize,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceAPITextEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        return default_from_dict(cls, data)

    @component.output_types(embedding=List[float])
    def run(self, text: str):
        """
        Embeds a single string.

        :param text:
            Text to embed.

        :returns:
            A dictionary with the following keys:
            - `embedding`: The embedding of the input text.
        """
        if not isinstance(text, str):
            raise TypeError(
                "HuggingFaceAPITextEmbedder expects a string as an input."
                "In case you want to embed a list of Documents, please use the HuggingFaceAPIDocumentEmbedder."
            )

        text_to_embed = self.prefix + text + self.suffix

        response = self._client.post(
            json={"inputs": [text_to_embed], "truncate": self.truncate, "normalize": self.normalize},
            task="feature-extraction",
        )
        embedding = json.loads(response.decode())[0]

        return {"embedding": embedding}
