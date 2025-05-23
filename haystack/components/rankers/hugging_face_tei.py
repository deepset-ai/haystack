# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

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
    Ranks a list of documents by their relevance to a given query using an external TEI reranking API.

    This component sends the query and document texts to a configured TEI API endpoint
    and retrieves a relevance-based ordering of the documents.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.rankers import HuggingFaceTEIRanker
    from haystack.utils import Secret

    reranker = HuggingFaceTEIRanker(
        url="https://api.my-tei-service.com",
        top_k=5,
        timeout=30,
        token=Secret.from_token("my_api_token")
    )
    docs = [Document(content="First doc"), Document(content="Second doc"), ...]
    result = reranker.run(query="example query", documents=docs)
    ranked_docs = result["documents"]
    ```
    """

    def __init__(
        self,
        url: str,
        top_k: int = 10,
        timeout: Optional[int] = 30,
        token: Optional[Secret] = Secret.from_env_var(["HF_API_TOKEN", "HF_TOKEN"], strict=False),
        max_retries: int = 3,
        retry_status_codes: Optional[List[int]] = None,
    ) -> None:
        """
        Initializes the TEI reranker component.

        :param url: Base URL of the TEI reranking service (e.g., "https://api.example.com").
        :param top_k: Maximum number of top documents to return (default: 10).
        :param timeout: Request timeout in seconds (default: 30).
        :param token: The Hugging Face token to use as HTTP bearer authorization.
            Check your HF token in your [account settings](https://huggingface.co/settings/tokens).
        :param max_retries: Maximum number of retry attempts for failed requests (default: 3).
        :param retry_status_codes: List of HTTP status codes that will trigger a retry.
            When None, HTTP 408, 418, 429 and 503 will be retried (default: None).
        """
        # Construct the full rerank endpoint
        self.url = url
        self.top_k = top_k
        self.timeout = timeout
        self.token = token
        self.max_retries = max_retries
        self.retry_status_codes = retry_status_codes

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component configuration to a dictionary for saving or logging.

        :returns: A dict containing the component's initialization parameters.
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
        Deserializes the component from a configuration dictionary.

        :param data: Configuration dict produced by `to_dict()`.

        :returns: An initialized `HuggingFaceTEIRanker` instance.
        """
        deserialize_secrets_inplace(data["init_parameters"], keys=["token"])
        return default_from_dict(cls, data)

    def _compose_response(
        self,
        result: Union[Dict[str, str], List[Dict[str, Any]]],
        top_k: Optional[int],
        documents: List[Document],
    ) -> Dict[str, List[Document]]:
        """
        Processes the API response into a structured format.

        :param result: The raw response from the API.

        :returns: A dict with key `documents` mapping to the reranked top documents.
        """
        if isinstance(result, dict) and "error" in result:
            error_type = result.get("error_type", "UnknownError")
            error_msg = result.get("error", "No additional information.")
            raise RuntimeError(f"HuggingFaceTEIRanker API call failed ({error_type}): {error_msg}")

        # Ensure we have a list of score dicts
        if not isinstance(result, list):
            raise RuntimeError("Unexpected response format from HuggingFaceTEIRanker API.")

        # Determine number of docs to return
        final_k = min(top_k or self.top_k, len(result))

        # Select and return the top_k documents
        ranked_docs = []
        for item in result[:final_k]:
            index: int = item["index"]
            documents[index].score = item["score"]
            ranked_docs.append(documents[index])
        return {"documents": ranked_docs}

    @component.output_types(documents=List[Document])
    def run(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
        raw_scores: bool = False,
        truncation_direction: Optional[TruncationDirection] = None,
    ) -> Dict[str, List[Document]]:
        """
        Reranks the provided documents by relevance to the query via the TEI API.

        :param query: The user query string to guide reranking.
        :param documents: List of `Document` objects to rerank.
        :param top_k: Optional override for the maximum number of documents to return.
        :param raw_scores: If True, include raw relevance scores in the API payload.
        :param truncation_direction: If set, enable text truncation in the specified direction.

        :returns: A dict with key `documents` mapping to the reranked top documents.

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
        payload: Dict[str, Any] = {"query": query, "texts": texts, "raw_scores": raw_scores}
        if truncation_direction:
            payload.update({"truncate": True, "truncation_direction": truncation_direction.value})

        # Call the external service with retry
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token.resolve_value()}"

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
        raw_scores: bool = False,
        truncation_direction: Optional[TruncationDirection] = None,
    ) -> Dict[str, List[Document]]:
        """
        Asynchronously reranks the provided documents by relevance to the query via the TEI API.

        :param query: The user query string to guide reranking.
        :param documents: List of `Document` objects to rerank.
        :param top_k: Optional override for the maximum number of documents to return.
        :param raw_scores: If True, include raw relevance scores in the API payload.
        :param truncation_direction: If set, enable text truncation in the specified direction.

        :returns: A dict with key `documents` mapping to the reranked top documents.

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
        payload: Dict[str, Any] = {"query": query, "texts": texts, "raw_scores": raw_scores}
        if truncation_direction:
            payload.update({"truncate": True, "truncation_direction": truncation_direction.value})

        # Call the external service with retry
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token.resolve_value()}"

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
