# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import httpx

from haystack import ComponentError, Document, component, default_from_dict, default_to_dict, logging
from haystack.utils import Secret

logger = logging.getLogger(__name__)

SIFTQ_BASE_URL = "https://api.siftq.com/v1/search"
SIFTQ_DEFAULT_API_KEY = "mk-1D3D81EFC32A25683B0C2C3B315F8579"  # intentionally public — free tier key

SIFTQ_SCOPE_MAP: dict[str, tuple[str, str]] = {
    "webpage": ("webpages", "link"),
    "document": ("documents", "link"),
    "scholar": ("scholars", "link"),
    "image": ("images", "imageUrl"),
    "video": ("videos", "link"),
    "podcast": ("podcasts", "link"),
}


class SiftqError(ComponentError): ...


_SIFTQ_ERROR_CODES: dict[int, str] = {
    2005: "API key rejected. Check your SiftQ API key.",
    3003: "daily search limit reached. See: https://siftq.com/playground",
}


@component
class SiftqWebSearch:
    """
    Uses [SiftQ](https://siftq.com) to search the web for relevant documents.

    Supports multi-scope search: webpages, documents, scholarly papers, images, videos, and podcasts.
    Use the `scope` parameter to select which scope to query.

    Use `search_params={"includeSummary": True}` to enhance recall with page summaries,
    and `search_params={"includeRawContent": True}` to fetch raw text content from source pages.

    Usage example:
    <!-- test-ignore -->
    ```python
    from haystack.components.websearch import SiftqWebSearch
    from haystack.utils import Secret

    websearch = SiftqWebSearch(top_k=10, api_key=Secret.from_env_var("SIFTQ_API_KEY"))
    results = websearch.run(query="What is the capital of France?")

    assert results["documents"]
    assert results["links"]
    ```
    """

    def __init__(
        self,
        api_key: Secret = Secret.from_env_var("SIFTQ_API_KEY", strict=False),
        top_k: int | None = 10,
        scope: str = "webpage",
        allowed_domains: list[str] | None = None,
        search_params: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize the SiftqWebSearch component.

        :param api_key: API key for the SiftQ API. Falls back to the built-in free tier key
            if not provided.
        :param top_k: Number of documents to return.
        :param scope: Search scope. One of "webpage", "document", "scholar", "image", "video", "podcast".
            Defaults to "webpage".
        :param allowed_domains: List of domains to limit the search to. Only applies to "webpage" scope.
        :param search_params: Additional parameters passed to the SiftQ API.
            For example: `{"includeSummary": true, "includeRawContent": true, "conciseSnippet": true}.
        """
        if scope not in SIFTQ_SCOPE_MAP:
            raise ValueError(
                f"Unknown scope '{scope}'. Must be one of: {', '.join(SIFTQ_SCOPE_MAP)}"
            )
        self.api_key = api_key
        self.top_k = top_k
        self.scope = scope
        self.allowed_domains = allowed_domains
        self.search_params = search_params or {}

    def _resolve_api_key(self) -> str:
        key = self.api_key.resolve_value()
        return key or SIFTQ_DEFAULT_API_KEY

    def to_dict(self) -> dict[str, Any]:
        return default_to_dict(
            self,
            top_k=self.top_k,
            scope=self.scope,
            allowed_domains=self.allowed_domains,
            search_params=self.search_params,
            api_key=self.api_key,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SiftqWebSearch":
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document], links=list[str])
    def run(self, query: str) -> dict[str, list[Document] | list[str]]:
        """
        Uses [SiftQ](https://siftq.com) to search the web.

        :param query: Search query.
        :returns: A dictionary with the following keys:
            - "documents": List of documents returned by the search engine.
            - "links": List of links returned by the search engine.
        :raises TimeoutError: If the request to the SiftQ API times out.
        :raises SiftqError: If an error occurs while querying the SiftQ API.
        """
        result_key, link_field = SIFTQ_SCOPE_MAP[self.scope]
        payload, headers = self._build_payload(query), self._build_headers()
        try:
            response = httpx.post(SIFTQ_BASE_URL, headers=headers, json=payload, timeout=90)
            response.raise_for_status()
        except httpx.ConnectTimeout as error:
            raise TimeoutError(f"Request to {self.__class__.__name__} timed out.") from error
        except httpx.HTTPStatusError as e:
            raise SiftqError(
                f"An error occurred while querying {self.__class__.__name__}. "
                f"Error: {e}, Response: {e.response.text}"
            ) from e
        except httpx.HTTPError as e:
            raise SiftqError(f"An error occurred while querying {self.__class__.__name__}. Error: {e}") from e
        self._check_api_error(response)

        documents, links = self._parse_response(response, result_key, link_field)

        logger.debug(
            "SiftQ returned {number_documents} documents for the query '{query}'",
            number_documents=len(documents),
            query=query,
        )
        return {"documents": documents[: self.top_k], "links": links[: self.top_k]}

    @component.output_types(documents=list[Document], links=list[str])
    async def run_async(self, query: str) -> dict[str, list[Document] | list[str]]:
        """
        Asynchronously uses [SiftQ](https://siftq.com) to search the web.

        This is the asynchronous version of the `run` method with the same parameters and return values.

        :param query: Search query.
        :returns: A dictionary with the following keys:
            - "documents": List of documents returned by the search engine.
            - "links": List of links returned by the search engine.
        :raises TimeoutError: If the request to the SiftQ API times out.
        :raises SiftqError: If an error occurs while querying the SiftQ API.
        """
        result_key, link_field = SIFTQ_SCOPE_MAP[self.scope]
        payload, headers = self._build_payload(query), self._build_headers()
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(SIFTQ_BASE_URL, headers=headers, json=payload, timeout=90)
                response.raise_for_status()
        except httpx.ConnectTimeout as error:
            raise TimeoutError(f"Request to {self.__class__.__name__} timed out.") from error
        except httpx.HTTPStatusError as e:
            raise SiftqError(
                f"An error occurred while querying {self.__class__.__name__}. "
                f"Error: {e}, Response: {e.response.text}"
            ) from e
        except httpx.HTTPError as e:
            raise SiftqError(f"An error occurred while querying {self.__class__.__name__}. Error: {e}") from e
        self._check_api_error(response)

        documents, links = self._parse_response(response, result_key, link_field)

        logger.debug(
            "SiftQ returned {number_documents} documents for the query '{query}'",
            number_documents=len(documents),
            query=query,
        )
        return {"documents": documents[: self.top_k], "links": links[: self.top_k]}

    @staticmethod
    def _check_api_error(response: httpx.Response) -> None:
        try:
            body = response.json()
        except Exception:
            return
        code = body.get("code")
        if not code:
            return
        msg = _SIFTQ_ERROR_CODES.get(code) or body.get("message", f"unknown error code {code}")
        raise SiftqError(f"SiftQ API error: {msg}")

    def _build_headers(self) -> dict[str, str]:
        api_key = self._resolve_api_key()
        return {"Authorization": f"Bearer {api_key}"}

    def _build_payload(self, query: str) -> dict[str, Any]:
        query_prepend = "OR ".join(f"site:{domain} " for domain in self.allowed_domains) if self.allowed_domains else ""
        payload: dict[str, Any] = {
            "q": query_prepend + query,
            "scope": self.scope,
            "size": self.top_k or 10,
            **self.search_params,
        }
        return payload

    def _parse_response(
        self, response: httpx.Response, result_key: str, link_field: str
    ) -> tuple[list[Document], list[str]]:
        json_result = response.json()
        documents: list[Document] = []
        links: list[str] = []

        for item in json_result.get(result_key, []):
            link = item.get(link_field, "")
            snippet_fields = ["snippet", "summary", "description", "title"]
            content = ""
            for f in snippet_fields:
                if item.get(f):
                    content = item[f]
                    break
            doc = Document(
                content=content,
                meta={
                    "title": item.get("title", ""),
                    "link": link,
                    "score": item.get("score", ""),
                    "position": item.get("position", 0),
                    "authors": item.get("authors", []),
                    "date": item.get("date", ""),
                },
            )
            if result_key == "scholars":
                doc.meta["citationCount"] = item.get("citationCount", 0)
                doc.meta["venue"] = item.get("venue", "")
                doc.meta["doi"] = item.get("doi", "")
            elif result_key == "videos":
                doc.meta["duration"] = item.get("duration", "")
                doc.meta["viewCount"] = item.get("viewCount", "")
            elif result_key == "podcasts":
                doc.meta["podcastName"] = item.get("podcastName", "")
                doc.meta["audioUrl"] = item.get("audioUrl", "")
                doc.meta["duration"] = item.get("duration", "")
            documents.append(doc)
            links.append(link)

        return documents, links
