# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from urllib.parse import urlencode, urlparse

import requests

from haystack import ComponentError, component, default_from_dict, default_to_dict, logging
from haystack.dataclasses import Document
from haystack.utils import Secret, deserialize_secrets_inplace

logger = logging.getLogger(__name__)

BRIGHT_DATA_REQUEST_URL = "https://api.brightdata.com/request"


class BrightDataWebSearchError(ComponentError):
    """Error raised when something goes wrong while talking to Bright Data."""


@component
class BrightDataWebSearch:
    """
    Web search using Bright Data SERP API.

    Uses `https://api.brightdata.com/request` with `data_format="parsed_light"`.

    - Search engine: Google (hard-coded)
    - Required input: query
    - Optional input: page_number (pagination via Google `start`)
    - Env vars:
        - BRIGHT_DATA_API_TOKEN
        - BRIGHT_DATA_ZONE
    """

    def __init__(
        self,
        api_token: Secret = Secret.from_env_var("BRIGHT_DATA_API_TOKEN"),
        zone: str | None = None,
        top_k: int | None = 5,
        country: str | None = "us",
        language: str | None = "en",
        allowed_domains: list[str] | None = None,
        extra_params: dict[str, Any] | None = None,
        timeout: float = 90.0,
    ):
        """
        Initializes the BrightDataWebSearch component.

        :param api_token: Bright Data API token (Secret).
        :param zone: Bright Data zone name. If None, read BRIGHT_DATA_ZONE.
        :param top_k: Number of documents/links to return.
        :param country: Google `gl` parameter.
        :param language: Google `hl` parameter.
        :param allowed_domains: If set, keep only results whose hostname matches allowed domains.
        :param extra_params: Extra Bright Data request params merged into the payload.
        :param timeout: Request timeout in seconds.
        ...
        """

        if zone is None:
            zone = Secret.from_env_var("BRIGHT_DATA_ZONE").resolve_value()

        if not zone:
            raise ValueError("BrightDataWebSearch requires a zone (pass zone=... or set BRIGHT_DATA_ZONE).")

        self.api_token = api_token
        self.zone = zone
        self.top_k = top_k
        self.country = country
        self.language = language
        self.allowed_domains = allowed_domains
        self.extra_params = extra_params or {}
        self.timeout = timeout

        # Validate token is present (will raise if env var missing)
        _ = self.api_token.resolve_value()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the component to a dictionary."""
        ...
        return default_to_dict(
            self,
            api_token=self.api_token.to_dict(),
            zone=self.zone,
            top_k=self.top_k,
            country=self.country,
            language=self.language,
            allowed_domains=self.allowed_domains,
            extra_params=self.extra_params,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BrightDataWebSearch":
        """Deserialize the component from a dictionary."""
        ...
        deserialize_secrets_inplace(data["init_parameters"], keys=["api_token"])
        return default_from_dict(cls, data)

    @component.output_types(documents=list[Document], links=list[str])
    def run(self, query: str, page_number: int = 1) -> dict[str, list[Document] | list[str]]:
        """
        Run a Google web search via Bright Data SERP API.

        :param query: Search query string.
        :param page_number: Page number for pagination (>= 1). Uses Google `start`.
        :returns: A dict with `documents` and `links`.
        :raises ValueError: If page_number < 1.
        :raises TimeoutError: If Bright Data's API times out.
        :raises BrightDataWebSearchError: If the request fails or response is invalid.
        """
        ...
        if page_number < 1:
            raise ValueError("page_number must be >= 1")

        if not query:
            return {"documents": [], "links": []}

        search_url = self._build_search_url(query=query, page_number=page_number)

        payload: dict[str, Any] = {"zone": self.zone, "url": search_url, "format": "raw", "data_format": "parsed_light"}
        payload.update(self.extra_params)

        headers = {"Authorization": f"Bearer {self.api_token.resolve_value()}", "Content-Type": "application/json"}

        try:
            response = requests.post(BRIGHT_DATA_REQUEST_URL, json=payload, headers=headers, timeout=self.timeout)
            response.raise_for_status()
        except requests.Timeout as error:
            raise TimeoutError(f"Request to {self.__class__.__name__} timed out.") from error
        except requests.RequestException as error:
            raise BrightDataWebSearchError(
                f"An error occurred while querying {self.__class__.__name__}. Error: {error}"
            ) from error

        try:
            raw_data = response.json()
        except ValueError as error:
            raise BrightDataWebSearchError(f"{self.__class__.__name__} returned a non-JSON response.") from error

        cleaned = self._clean_google_search_payload(raw_data)
        documents, links = self._build_documents(cleaned)

        logger.debug(
            "BrightData returned {number_documents} documents for query '{query}'",
            number_documents=len(documents),
            query=query,
        )

        if self.top_k is not None:
            documents = documents[: self.top_k]
            links = links[: self.top_k]

        return {"documents": documents, "links": links}

    def _build_search_url(self, query: str, page_number: int) -> str:
        params: dict[str, Any] = {"q": query}
        if self.language:
            params["hl"] = self.language
        if self.country:
            params["gl"] = self.country

        # Google pagination: start = (page-1) * 10
        if page_number > 1:
            params["start"] = (page_number - 1) * 10

        return f"https://www.google.com/search?{urlencode(params)}"

    @staticmethod
    def _clean_google_search_payload(raw_data: Any) -> dict[str, list[dict[str, Any]]]:
        data = raw_data if isinstance(raw_data, dict) else {}
        organic = data.get("organic", [])
        if not isinstance(organic, list):
            organic = []

        cleaned: list[dict[str, Any]] = []
        for entry in organic:
            if not isinstance(entry, dict):
                continue

            link = (entry.get("link") or "").strip()
            title = (entry.get("title") or "").strip()
            description = (entry.get("description") or "").strip()

            if not link or not title:
                continue

            cleaned.append(
                {
                    "link": link,
                    "title": title,
                    "description": description,
                    "global_rank": entry.get("global_rank"),
                    "extensions": entry.get("extensions"),
                }
            )

        return {"organic": cleaned}

    def _build_documents(self, data: dict[str, list[dict[str, Any]]]) -> tuple[list[Document], list[str]]:
        organic = data.get("organic") or []

        documents: list[Document] = []
        links: list[str] = []

        for entry in organic:
            link = entry.get("link")
            title = entry.get("title")
            description = entry.get("description") or ""

            if not link:
                continue

            if self.allowed_domains and not self._is_domain_allowed(link):
                continue

            meta: dict[str, Any] = {"title": title, "link": link}
            if entry.get("global_rank") is not None:
                meta["global_rank"] = entry["global_rank"]
            if entry.get("extensions") is not None:
                meta["extensions"] = entry["extensions"]

            documents.append(Document(content=description or title, meta=meta))
            links.append(link)

        return documents, links

    def _is_domain_allowed(self, url: str) -> bool:
        if not self.allowed_domains:
            return True

        hostname = urlparse(url).hostname or ""
        return any(hostname == domain or hostname.endswith("." + domain) for domain in self.allowed_domains)
