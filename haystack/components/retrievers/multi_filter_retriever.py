# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.components.retrievers.types import FilterRetriever
from haystack.core.serialization import component_to_dict
from haystack.utils.misc import _deduplicate_documents


@component
class MultiFilterRetriever:
    """
    A component that retrieves documents using multiple filters in parallel.

    This component takes a list of filter dictionaries and uses a filter-capable retriever to retrieve matching
    documents for each filter set in parallel. The results are combined, de-duplicated, and sorted by score.

    ### Usage example

    ```python
    from haystack import Document
    from haystack.components.retrievers import FilterRetriever, MultiFilterRetriever
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

    filter_retriever = FilterRetriever(document_store=document_store)
    multi_filter_retriever = MultiFilterRetriever(retriever=filter_retriever)

    filters = [
        {"field": "meta.lang", "operator": "==", "value": "en"},
        {"field": "meta.lang", "operator": "==", "value": "de"},
    ]

    result = multi_filter_retriever.run(filters=filters)
    for doc in result["documents"]:
        print(doc.content)
    ```
    """

    def __init__(self, *, retriever: FilterRetriever, max_workers: int = 3) -> None:
        """
        Initialize MultiFilterRetriever.

        :param retriever: The filter-capable retriever to use for document retrieval.
        :param max_workers: Maximum number of worker threads for parallel processing.
        """
        self.retriever = retriever
        self.max_workers = max_workers
        self._is_warmed_up = False

    def warm_up(self) -> None:
        """
        Warm up the retriever if it has a warm_up method.
        """
        if not self._is_warmed_up:
            if hasattr(self.retriever, "warm_up") and callable(self.retriever.warm_up):
                self.retriever.warm_up()
            self._is_warmed_up = True

    @component.output_types(documents=list[Document])
    def run(
        self, filters: list[dict[str, Any]], retriever_kwargs: dict[str, Any] | None = None
    ) -> dict[str, list[Document]]:
        """
        Retrieve documents using multiple filters in parallel.

        :param filters: List of filter dictionaries to process.
        :param retriever_kwargs: Optional dictionary of arguments to pass to the retriever's run method.
        :returns:
            A dictionary containing:
                - `documents`: List of retrieved documents sorted by relevance score.
        """
        docs: list[Document] = []
        retriever_kwargs = retriever_kwargs or {}

        if not self._is_warmed_up:
            self.warm_up()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            filters_results = executor.map(lambda filt: self._run_on_thread(filt, retriever_kwargs), filters)
            for result in filters_results:
                if not result:
                    continue
                docs.extend(result)

        docs = _deduplicate_documents(docs)
        docs.sort(key=lambda x: x.score or 0.0, reverse=True)
        return {"documents": docs}

    def _run_on_thread(
        self, filters: dict[str, Any], retriever_kwargs: dict[str, Any] | None = None
    ) -> list[Document] | None:
        """
        Process a single filter set on a separate thread.

        :param filters: The filter dictionary to process.
        :param retriever_kwargs: Optional dictionary of arguments to pass to the retriever's run method.
        :returns:
            List of retrieved documents or None if no results.
        """
        result = self.retriever.run(filters=filters, **(retriever_kwargs or {}))
        if result and "documents" in result:
            return result["documents"]
        return None

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            A dictionary representing the serialized component.
        """
        return default_to_dict(
            self, retriever=component_to_dict(obj=self.retriever, name="retriever"), max_workers=self.max_workers
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MultiFilterRetriever":
        """
        Deserializes the component from a dictionary.

        :param data: The dictionary to deserialize from.
        :returns:
            The deserialized component.
        """
        return default_from_dict(cls, data)
