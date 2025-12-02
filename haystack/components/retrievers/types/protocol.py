# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional, Protocol

# Ellipsis are needed to define the Protocol but pylint complains. See https://github.com/pylint-dev/pylint/issues/9319.
# pylint: disable=unnecessary-ellipsis


class TextRetriever(Protocol):
    """
    This protocol defines the minimal interface that all keyword-based BM25 Retrievers must implement.

    Retrievers are components that process a query and, based on that query, return relevant documents from a document
    store or other data source. They return a dictionary with a list of Document objects.
    """

    def run(self, query: str, filters: Optional[dict[str, Any]] = None, top_k: Optional[int] = None) -> dict[str, Any]:
        """
        Retrieve documents that are relevant to the query.

        Implementing classes may accept additional optional parameters in their run method.

        :param query: The input query string.
        :param filters: A dictionary of filters to apply when retrieving documents.
        :param top_k: The maximum number of documents to return.

        :returns:
            A dictionary containing:
                `documents`: List of retrieved documents sorted by relevance score.
        """
        ...


class EmbeddingRetriever(Protocol):
    """
    This protocol defines the minimal interface that all embedding-based Retrievers must implement.

    Retrievers are components that process a query and, based on that query, return relevant documents from a document
    store or other data source. They return a dictionary with a list of Document objects.
    """

    def run(
        self, query_embedding: list[float], filters: Optional[dict[str, Any]] = None, top_k: Optional[int] = None
    ) -> dict[str, Any]:
        """
        Retrieve documents that are relevant to the query.

        Implementing classes may accept additional optional parameters in their run method.

        :param query_embedding: The input query embedding.
        :param filters: A dictionary of filters to apply when retrieving documents.
        :param top_k: The maximum number of documents to return.
        :returns:
            A dictionary containing:
                `documents`: List of retrieved documents sorted by relevance score.
        """
        ...
