# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from haystack import Document, component
from haystack.components.retrievers.filter_retriever import FilterRetriever
from haystack.document_stores.types import DocumentStore
from haystack.utils.misc import _deduplicate_documents


@component
class MultiFilterRetriever:
    """
    A component that retrieves documents using multiple filters in parallel.

    This component takes a list of filter dictionaries and retrieves matching documents for each filter set
    in parallel.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.retrievers import MultiFilterRetriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore
    from haystack.components.writers import DocumentWriter
    from haystack.document_stores.types import DuplicatePolicy

    documents = [
        Document(content="Python is a popular programming language", meta={"lang": "en"}),
        Document(content="python ist eine beliebte Programmiersprache", meta={"lang": "de"}),
    ]

    document_store = InMemoryDocumentStore()
    writer = DocumentWriter(document_store=document_store, policy=DuplicatePolicy.SKIP)
    writer.run(documents=documents)

    multi_filter_retriever = MultiFilterRetriever(document_store=document_store)

    filters = [
        {"field": "meta.lang", "operator": "==", "value": "en"},
        {"field": "meta.lang", "operator": "==", "value": "de"},
    ]

    result = multi_filter_retriever.run(filters=filters)
    for doc in result["documents"]:
        print(doc.content)
    ```
    """

    def __init__(self, document_store: DocumentStore, max_workers: int = 3) -> None:
        """
        Initialize MultiFilterRetriever.

        :param document_store: The document store to retrieve documents from.
        :param max_workers: Maximum number of worker threads for parallel processing.
        """
        self.document_store = document_store
        self.max_workers = max_workers
        self._retriever = FilterRetriever(document_store=document_store)

    @component.output_types(documents=list[Document])
    def run(self, filters: list[dict[str, Any]]) -> dict[str, list[Document]]:
        """
        Retrieve documents using multiple filters in parallel.

        :param filters: List of filter dictionaries to process.
        :returns:
            A dictionary containing:
                - `documents`: List of retrieved documents.
        """
        docs: list[Document] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            filters_results = executor.map(self._run_on_thread, filters)
            for result in filters_results:
                if not result:
                    continue
                docs.extend(result)

        docs = _deduplicate_documents(docs)

        return {"documents": docs}

    def _run_on_thread(self, filters: dict[str, Any]) -> list[Document] | None:
        result = self._retriever.run(filters=filters)
        if result and "documents" in result:
            return result["documents"]
        return None
