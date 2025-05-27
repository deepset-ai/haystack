# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import copy
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.utils import Secret, deserialize_secrets_inplace
from haystack.utils.requests_utils import async_request_with_retry, request_with_retry


class TruncationDirection(str, Enum):
    """
    Defines the direction to truncate text when input length exceeds the model's limit.

    Attributes:
        LEFT: Truncate text from the left side (start of text).
        RIGHT: Truncate text from the right side (end of text).
    """

    LEFT = "Left"
    RIGHT = "Right"


@component
class HuggingFaceTEIRanker:
    """
    Ranks documents based on their semantic similarity to the query.

    It can be used with a Text Embeddings Inference (TEI) API endpoint:
    - [Self-hosted Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference)
    - [Hugging Face Inference Endpoints](https://huggingface.co/inference-endpoints)

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.rankers import HuggingFaceTEIRanker
    from haystack.utils import Secret

    reranker = HuggingFaceTEIRanker(
        url="http://localhost:8080",
        top_k=5,
        timeout=30,
        token=Secret.from_token("my_api_token")
    )

    docs = [Document(content="The capital of France is Paris"), Document(content="The capital of Germany is Berlin")]

    result = reranker.run(query="What is the capital of France?", documents=docs)

    ranked_docs = result["documents"]
    print(ranked_docs)
    >> {'documents': [Document(id=..., content: 'the capital of France is Paris', score: 0.9979767),
    >>                Document(id=..., content: 'the capital of Germany is Berlin', score: 0.13982213)]}
    ```
    """

    def __init__(
        self,
        *,
        url: str,
        top_k: int = 10,
        raw_scores: bool = False,
        timeout: Optional[int] = 30,
        max_retries: int = 3,
        retry_status_codes: Optional[List[int]] = None,
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
    ) -> None:
        """
        Initializes the TEI reranker component.

        :param url: Base URL of the TEI reranking service (for example, "https://api.example.com").
        :param top_k: Maximum number of top documents to return.
        :param raw_scores: If True, include raw relevance scores in the API payload.
        :param timeout: Request timeout in seconds.
        :param max_retries: Maximum number of retry attempts for failed requests.
        :param retry_status_codes: List of HTTP status codes that will trigger a retry.
            When None, HTTP 408, 418, 429 and 503 will be retried (default: None).
        :param token: The Hugging Face token to use as HTTP bearer authorization. Not always required
            depending on your TEI server configuration.
            Check your HF token in your [account settings](https://huggingface.co/settings/tokens).
        """
        self.url = url
        self.top_k = top_k
        self.timeout = timeout
        self.token = token
        self.max_retries = max_retries
        self.retry_status_codes = retry_status_codes
        self.raw_scores = raw_scores

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(
            self,
            url=self.url,
            top_k=self.top_k,
            timeout=self.timeout,
            token=self.token.to_dict() if self.token else None,
            max_retries=self.max_retries,
            retry_status_codes=self.retry_status_codes,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceTEIRanker":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary to deserialize from.
        :returns:
            Deserialized component.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        return default_from_dict(cls, data)

    def _compose_response(
        self, result: Union[Dict[str, str], List[Dict[str, Any]]], top_k: Optional[int], documents: List[Document]
    ) -> Dict[str, List[Document]]:
        """
        Processes the API response into a structured format.

        :param result: The raw response from the API.

        :returns: A dictionary with the following keys:
            - `documents`: A list of reranked documents.

        :raises requests.exceptions.RequestException:
            - If the API request fails.

        :raises RuntimeError:
            - If the API returns an error response.
        """
        if isinstance(result, dict) and "error" in result:
            error_type = result.get("error_type", "UnknownError")
            error_msg = result.get("error", "No additional information.")
            raise RuntimeError(f"HuggingFaceTEIRanker API call failed ({error_type}): {error_msg}")

        # Ensure we have a list of score dicts
        if not isinstance(result, list):
            # Expected list or dict, but encountered an unknown response format.
            error_msg = f"Expected a list of score dictionaries, but got `{type(result).__name__}`. "
            error_msg += f"Response content: {result}"
            raise RuntimeError(f"Unexpected response format from text-embeddings-inference rerank API: {error_msg}")

        # Determine number of docs to return
        final_k = min(top_k or self.top_k, len(result))

        # Select and return the top_k documents
        ranked_docs = []
        for item in result[:final_k]:
            index: int = item["index"]
            doc_copy = copy.copy(documents[index])
            doc_copy.score = item["score"]
            ranked_docs.append(doc_copy)
        return {"documents": ranked_docs}

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        truncation_direction: Optional[TruncationDirection] = None,
    ) -> Dict[str, List[Document]]:
        """
        Reranks the provided documents by relevance to the query using the TEI API.

        :param query: The user query string to guide reranking.
        :param documents: List of `Document` objects to rerank.
        :param top_k: Optional override for the maximum number of documents to return.
        :param truncation_direction: If set, enables text truncation in the specified direction.

        :returns: A dictionary with the following keys:
            - `documents`: A list of reranked documents.

        :raises requests.exceptions.RequestException:
            - If the API request fails.

        :raises RuntimeError:
            - If the API returns an error response.
        """
        # Return empty if no documents provided
        if not documents:
            return {"documents": []}

        # Prepare the payload
        texts = [doc.content for doc in documents]
        payload: Dict[str, Any] = {"query": query, "texts": texts, "raw_scores": self.raw_scores}
        if truncation_direction:
            payload.update({"truncate": True, "truncation_direction": truncation_direction.value})

        headers = {}
        if self.token and self.token.resolve_value():
            headers["Authorization"] = f"Bearer {self.token.resolve_value()}"

        # Call the external service with retry
        response = request_with_retry(
            method="POST",
            url=urljoin(self.url, "/rerank"),
            json=payload,
            timeout=self.timeout,
            headers=headers,
            attempts=self.max_retries,
            status_codes_to_retry=self.retry_status_codes,
        )

        result: Union[Dict[str, str], List[Dict[str, Any]]] = response.json()

        return self._compose_response(result, top_k, documents)

    @component.output_types(documents=List[Document])
    async def run_async(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        truncation_direction: Optional[TruncationDirection] = None,
    ) -> Dict[str, List[Document]]:
        """
        Asynchronously reranks the provided documents by relevance to the query using the TEI API.

        :param query: The user query string to guide reranking.
        :param documents: List of `Document` objects to rerank.
        :param top_k: Optional override for the maximum number of documents to return.
        :param truncation_direction: If set, enables text truncation in the specified direction.

        :returns: A dictionary with the following keys:
            - `documents`: A list of reranked documents.

        :raises httpx.RequestError:
            - If the API request fails.
        :raises RuntimeError:
            - If the API returns an error response.
        """
        # Return empty if no documents provided
        if not documents:
            return {"documents": []}

        # Prepare the payload
        texts = [doc.content for doc in documents]
        payload: Dict[str, Any] = {"query": query, "texts": texts, "raw_scores": self.raw_scores}
        if truncation_direction:
            payload.update({"truncate": True, "truncation_direction": truncation_direction.value})

        headers = {}
        if self.token and self.token.resolve_value():
            headers["Authorization"] = f"Bearer {self.token.resolve_value()}"

        # Call the external service with retry
        response = await async_request_with_retry(
            method="POST",
            url=urljoin(self.url, "/rerank"),
            json=payload,
            timeout=self.timeout,
            headers=headers,
            attempts=self.max_retries,
            status_codes_to_retry=self.retry_status_codes,
        )

        result: Union[Dict[str, str], List[Dict[str, Any]]] = response.json()

        return self._compose_response(result, top_k, documents)
