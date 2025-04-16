# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple, Union

from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from haystack import component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.lazy_imports import LazyImport
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.hf import HFEmbeddingAPIType, HFModelType, check_valid_model
from haystack.utils.url_validation import is_valid_http_url

with LazyImport(message="Run 'pip install \"huggingface_hub>=0.27.0\"'") as huggingface_hub_import:
    from huggingface_hub import AsyncInferenceClient, InferenceClient

logger = logging.getLogger(__name__)


@component
class HuggingFaceAPIDocumentEmbedder:
    """
    Embeds documents using Hugging Face APIs.

    Use it with the following Hugging Face APIs:
    - [Free Serverless Inference API](https://huggingface.co/inference-api)
    - [Paid Inference Endpoints](https://huggingface.co/inference-endpoints)
    - [Self-hosted Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)


    ### Usage examples

    #### With free serverless inference API

    ```python
    from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
    from haystack.utils import Secret
    from haystack.dataclasses import Document

    doc = Document(content="I love pizza!")

    doc_embedder = HuggingFaceAPIDocumentEmbedder(api_type="serverless_inference_api",
                                                  api_params={"model": "BAAI/bge-small-en-v1.5"},
                                                  token=Secret.from_token("<your-api-key>"))

    result = document_embedder.run([doc])
    print(result["documents"][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```

    #### With paid inference endpoints

    ```python
    from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
    from haystack.utils import Secret
    from haystack.dataclasses import Document

    doc = Document(content="I love pizza!")

    doc_embedder = HuggingFaceAPIDocumentEmbedder(api_type="inference_endpoints",
                                                  api_params={"url": "<your-inference-endpoint-url>"},
                                                  token=Secret.from_token("<your-api-key>"))

    result = document_embedder.run([doc])
    print(result["documents"][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```

    #### With self-hosted text embeddings inference

    ```python
    from haystack.components.embedders import HuggingFaceAPIDocumentEmbedder
    from haystack.dataclasses import Document

    doc = Document(content="I love pizza!")

    doc_embedder = HuggingFaceAPIDocumentEmbedder(api_type="text_embeddings_inference",
                                                  api_params={"url": "http://localhost:8080"})

    result = document_embedder.run([doc])
    print(result["documents"][0].embedding)

    # [0.017020374536514282, -0.023255806416273117, ...]
    ```
    """

    def __init__(
        self,
        api_type: Union[HFEmbeddingAPIType, str],
        api_params: Dict[str, str],
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        prefix: str = "",
        suffix: str = "",
        truncate: Optional[bool] = True,
        normalize: Optional[bool] = False,
        batch_size: int = 32,
        progress_bar: bool = True,
        meta_fields_to_embed: Optional[List[str]] = None,
        embedding_separator: str = "\n",
    ):  # pylint: disable=too-many-positional-arguments
        """
        Creates a HuggingFaceAPIDocumentEmbedder component.

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
        :param normalize:
            Normalizes the embeddings to unit length.
            Applicable when `api_type` is `TEXT_EMBEDDINGS_INFERENCE`, or `INFERENCE_ENDPOINTS`
            if the backend uses Text Embeddings Inference.
            If `api_type` is `SERVERLESS_INFERENCE_API`, this parameter is ignored.
        :param batch_size:
            Number of documents to process at once.
        :param progress_bar:
            If `True`, shows a progress bar when running.
        :param meta_fields_to_embed:
            List of metadata fields to embed along with the document text.
        :param embedding_separator:
            Separator used to concatenate the metadata fields to the document text.
        """
        huggingface_hub_import.check()

        if isinstance(api_type, str):
            api_type = HFEmbeddingAPIType.from_str(api_type)

        api_params = api_params or {}

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

        client_args: Dict[str, Any] = {"model": model_or_url, "token": token.resolve_value() if token else None}

        self.api_type = api_type
        self.api_params = api_params
        self.token = token
        self.prefix = prefix
        self.suffix = suffix
        self.truncate = truncate
        self.normalize = normalize
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.meta_fields_to_embed = meta_fields_to_embed or []
        self.embedding_separator = embedding_separator
        self._client = InferenceClient(**client_args)
        self._async_client = AsyncInferenceClient(**client_args)

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
            batch_size=self.batch_size,
            progress_bar=self.progress_bar,
            meta_fields_to_embed=self.meta_fields_to_embed,
            embedding_separator=self.embedding_separator,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceAPIDocumentEmbedder":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        return default_from_dict(cls, data)

    def _prepare_texts_to_embed(self, documents: List[Document]) -> List[str]:
        """
        Prepare the texts to embed by concatenating the Document text with the metadata fields to embed.
        """
        texts_to_embed = []
        for doc in documents:
            meta_values_to_embed = [
                str(doc.meta[key]) for key in self.meta_fields_to_embed if key in doc.meta and doc.meta[key] is not None
            ]

            text_to_embed = (
                self.prefix + self.embedding_separator.join(meta_values_to_embed + [doc.content or ""]) + self.suffix
            )

            texts_to_embed.append(text_to_embed)
        return texts_to_embed

    @staticmethod
    def _adjust_api_parameters(
        truncate: Optional[bool], normalize: Optional[bool], api_type: HFEmbeddingAPIType
    ) -> Tuple[Optional[bool], Optional[bool]]:
        """
        Adjust the truncate and normalize parameters based on the API type.
        """
        if api_type == HFEmbeddingAPIType.SERVERLESS_INFERENCE_API:
            if truncate is not None:
                msg = "`truncate` parameter is not supported for Serverless Inference API. It will be ignored."
                logger.warning(msg)
                truncate = None
            if normalize is not None:
                msg = "`normalize` parameter is not supported for Serverless Inference API. It will be ignored."
                logger.warning(msg)
                normalize = None
        return truncate, normalize

    def _embed_batch(self, texts_to_embed: List[str], batch_size: int) -> List[List[float]]:
        """
        Embed a list of texts in batches.
        """
        truncate, normalize = self._adjust_api_parameters(self.truncate, self.normalize, self.api_type)

        all_embeddings: List = []
        for i in tqdm(
            range(0, len(texts_to_embed), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i : i + batch_size]

            np_embeddings = self._client.feature_extraction(
                # this method does not officially support list of strings, but works as expected
                text=batch,  # type: ignore[arg-type]
                truncate=truncate,
                normalize=normalize,
            )

            if np_embeddings.ndim != 2 or np_embeddings.shape[0] != len(batch):
                raise ValueError(f"Expected embedding shape ({batch_size}, embedding_dim), got {np_embeddings.shape}")

            all_embeddings.extend(np_embeddings.tolist())

        return all_embeddings

    async def _embed_batch_async(self, texts_to_embed: List[str], batch_size: int) -> List[List[float]]:
        """
        Embed a list of texts in batches asynchronously.
        """
        truncate, normalize = self._adjust_api_parameters(self.truncate, self.normalize, self.api_type)

        all_embeddings: List = []
        for i in async_tqdm(
            range(0, len(texts_to_embed), batch_size), disable=not self.progress_bar, desc="Calculating embeddings"
        ):
            batch = texts_to_embed[i : i + batch_size]

            np_embeddings = await self._async_client.feature_extraction(
                # this method does not officially support list of strings, but works as expected
                text=batch,  # type: ignore[arg-type]
                truncate=truncate,
                normalize=normalize,
            )

            if np_embeddings.ndim != 2 or np_embeddings.shape[0] != len(batch):
                raise ValueError(f"Expected embedding shape ({batch_size}, embedding_dim), got {np_embeddings.shape}")

            all_embeddings.extend(np_embeddings.tolist())

        return all_embeddings

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Embeds a list of documents.

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of documents with embeddings.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "HuggingFaceAPIDocumentEmbedder expects a list of Documents as input."
                " In case you want to embed a string, please use the HuggingFaceAPITextEmbedder."
            )

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        embeddings = self._embed_batch(texts_to_embed=texts_to_embed, batch_size=self.batch_size)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents}

    @component.output_types(documents=List[Document])
    async def run_async(self, documents: List[Document]):
        """
        Embeds a list of documents asynchronously.

        :param documents:
            Documents to embed.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of documents with embeddings.
        """
        if not isinstance(documents, list) or documents and not isinstance(documents[0], Document):
            raise TypeError(
                "HuggingFaceAPIDocumentEmbedder expects a list of Documents as input."
                " In case you want to embed a string, please use the HuggingFaceAPITextEmbedder."
            )

        texts_to_embed = self._prepare_texts_to_embed(documents=documents)

        embeddings = await self._embed_batch_async(texts_to_embed=texts_to_embed, batch_size=self.batch_size)

        for doc, emb in zip(documents, embeddings):
            doc.embedding = emb

        return {"documents": documents}
