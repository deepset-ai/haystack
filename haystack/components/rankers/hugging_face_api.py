# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from haystack import Document, component, default_from_dict, default_to_dict


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
class HuggingFaceAPIRanker:
    """
    Ranks a list of documents by their relevance to a given query using an external TEI reranking API.

    This component sends the query and document texts to a configured TEI API endpoint
    and retrieves a relevance-based ordering of the documents.

    Usage example:
    ```python
    from haystack import Document
    from haystack.components.rankers import HuggingFaceAPIRanker

    reranker = HuggingFaceAPIRanker(
        url="https://api.my-tei-service.com",
        top_k=5,
        timeout=30,
        token="my_api_token"
    )
    reranker.warm_up()
    docs = [Document(content="First doc"), Document(content="Second doc"), ...]
    result = reranker.run(query="example query", documents=docs)
    ranked_docs = result["documents"]
    ```
    """

    def __init__(
        self, url: str, top_k: int = 10, timeout: Optional[int] = 30, token: Optional[str] = None
    ) -> None:
        """
        Initializes the TEI reranker component.

        :param url: Base URL of the TEI reranking service (e.g., "https://api.example.com").
        :param top_k: Maximum number of top documents to return (default: 10).
        :param timeout: Request timeout in seconds (default: 30).
        :param token: Optional bearer token for API authentication (default: None).
        """
        # Construct the full rerank endpoint
        self.url = url
        self.top_k = top_k
        self.timeout = timeout
        self.token = token

        # Initialize a persistent HTTP session for performance
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {token}"})

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component configuration to a dictionary for saving or logging.

        :returns: A dict containing the component's initialization parameters.
        """
        return default_to_dict(self, url=self.url, top_k=self.top_k, timeout=self.timeout, token=self.token)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HuggingFaceAPIRanker":
        """
        Deserializes the component from a configuration dictionary.

        :param data: Configuration dict produced by `to_dict()`.
        :returns: An initialized `HuggingFaceAPIRanker` instance.
        """
        return default_from_dict(cls, data)

    def warm_up(self) -> None:
        """
        Sends a dummy request to the reranking endpoint to establish connections and verify availability.

        Raises an exception if the endpoint is unreachable or returns an error.
        """
        payload = {"query": "warmup", "texts": ["warmup"]}
        response = self.session.post(urljoin(self.url, "/rerank"), json=payload, timeout=self.timeout)
        response.raise_for_status()

    @component.output_types(documents=List[Document])
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
    )
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

        # Call the external service
        response = self.session.post(urljoin(self.url, "/rerank"), json=payload, timeout=self.timeout)
        response.raise_for_status()
        result = response.json()

        # Handle error structure from the TEI API
        if isinstance(result, dict) and "error" in result:
            error_type = result.get("error_type", "UnknownError")
            error_msg = result.get("error", "No additional information.")
            raise RuntimeError(f"HuggingFaceAPIRanker API call failed ({error_type}): {error_msg}")

        # Ensure we have a list of score dicts
        if not isinstance(result, list):
            raise RuntimeError("Unexpected response format from HuggingFaceAPIRanker API.")

        # Determine number of docs to return
        final_k = min(top_k or self.top_k, len(result))

        # Select and return the top_k documents
        ranked_docs = []
        for item in result[:final_k]:
            index: int = item["index"]
            documents[index].score = item["score"]
            ranked_docs.append(documents[index])
        return {"documents": ranked_docs}
