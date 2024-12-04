# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.document_stores.types import DocumentStore
from haystack.utils import deserialize_document_store_in_init_params_inplace


@component
class SentenceWindowRetriever:
    """
    Retrieves documents adjacent to a given document in the Document Store.

    During indexing, documents are broken into smaller chunks, or sentences. When you submit a query,
    the Retriever fetches the most relevant sentence. To provide full context,
    SentenceWindowRetriever fetches a number of neighboring sentences before and after each
    relevant one. You can set this number with the `window_size` parameter.
    It uses `source_id` and `doc.meta['split_id']` to locate the surrounding documents.

    This component works with existing Retrievers, like BM25Retriever or
    EmbeddingRetriever. First, use a Retriever to find documents based on a query and then use
    SentenceWindowRetriever to get the surrounding documents for context.

    The SentenceWindowRetriever is compatible with the following DocumentStores:
    - [Astra](https://docs.haystack.deepset.ai/docs/astradocumentstore)
    - [Elasticsearch](https://docs.haystack.deepset.ai/docs/elasticsearch-document-store)
    - [OpenSearch](https://docs.haystack.deepset.ai/docs/opensearch-document-store)
    - [Pgvector](https://docs.haystack.deepset.ai/docs/pgvectordocumentstore)
    - [Pinecone](https://docs.haystack.deepset.ai/docs/pinecone-document-store)
    - [Qdrant](https://docs.haystack.deepset.ai/docs/qdrant-document-store)

    ### Usage example

    ```python
    from haystack import Document, Pipeline
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
    from haystack.components.retrievers import SentenceWindowRetriever
    from haystack.components.preprocessors import DocumentSplitter
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    splitter = DocumentSplitter(split_length=10, split_overlap=5, split_by="word")
    text = (
            "This is a text with some words. There is a second sentence. And there is also a third sentence. "
            "It also contains a fourth sentence. And a fifth sentence. And a sixth sentence. And a seventh sentence"
    )
    doc = Document(content=text)
    docs = splitter.run([doc])
    doc_store = InMemoryDocumentStore()
    doc_store.write_documents(docs["documents"])


    rag = Pipeline()
    rag.add_component("bm25_retriever", InMemoryBM25Retriever(doc_store, top_k=1))
    rag.add_component("sentence_window_retriever", SentenceWindowRetriever(document_store=doc_store, window_size=2))
    rag.connect("bm25_retriever", "sentence_window_retriever")

    rag.run({'bm25_retriever': {"query":"third"}})

    >> {'sentence_window_retriever': {'context_windows': ['some words. There is a second sentence.
    >> And there is also a third sentence. It also contains a fourth sentence. And a fifth sentence. And a sixth
    >> sentence. And a'], 'context_documents': [[Document(id=..., content: 'some words. There is a second sentence.
    >> And there is ', meta: {'source_id': '...', 'page_number': 1, 'split_id': 1, 'split_idx_start': 20,
    >> '_split_overlap': [{'doc_id': '...', 'range': (20, 43)}, {'doc_id': '...', 'range': (0, 30)}]}),
    >> Document(id=..., content: 'second sentence. And there is also a third sentence. It ',
    >> meta: {'source_id': '74ea87deb38012873cf8c07e...f19d01a26a098447113e1d7b83efd30c02987114', 'page_number': 1,
    >> 'split_id': 2, 'split_idx_start': 43, '_split_overlap': [{'doc_id': '...', 'range': (23, 53)}, {'doc_id': '...',
    >> 'range': (0, 26)}]}), Document(id=..., content: 'also a third sentence. It also contains a fourth sentence. ',
    >> meta: {'source_id': '...', 'page_number': 1, 'split_id': 3, 'split_idx_start': 73, '_split_overlap':
    >> [{'doc_id': '...', 'range': (30, 56)}, {'doc_id': '...', 'range': (0, 33)}]}), Document(id=..., content:
    >> 'also contains a fourth sentence. And a fifth sentence. And ', meta: {'source_id': '...', 'page_number': 1,
    >> 'split_id': 4, 'split_idx_start': 99, '_split_overlap': [{'doc_id': '...', 'range': (26, 59)},
    >> {'doc_id': '...', 'range': (0, 26)}]}), Document(id=..., content: 'And a fifth sentence. And a sixth sentence.
    >> And a ', meta: {'source_id': '...', 'page_number': 1, 'split_id': 5, 'split_idx_start': 132,
    >> '_split_overlap': [{'doc_id': '...', 'range': (33, 59)}, {'doc_id': '...', 'range': (0, 24)}]})]]}}}}
    ```
    """

    def __init__(self, document_store: DocumentStore, window_size: int = 3):
        """
        Creates a new SentenceWindowRetriever component.

        :param document_store: The Document Store to retrieve the surrounding documents from.
        :param window_size: The number of documents to retrieve before and after the relevant one.
                For example, `window_size: 2` fetches 2 preceding and 2 following documents.
        """
        if window_size < 1:
            raise ValueError("The window_size parameter must be greater than 0.")

        self.window_size = window_size
        self.document_store = document_store

        warnings.warn(
            "The output of `context_documents` will change in the next release. Instead of a "
            "List[List[Document]], the output will be a List[Document], where the documents are ordered by "
            "`split_idx_start`.",
            DeprecationWarning,
        )

    @staticmethod
    def merge_documents_text(documents: List[Document]) -> str:
        """
        Merge a list of document text into a single string.

        This functions concatenates the textual content of a list of documents into a single string, eliminating any
        overlapping content.

        :param documents: List of Documents to merge.
        """
        sorted_docs = sorted(documents, key=lambda doc: doc.meta["split_idx_start"])
        merged_text = ""
        last_idx_end = 0
        for doc in sorted_docs:
            start = doc.meta["split_idx_start"]  # start of the current content

            # if the start of the current content is before the end of the last appended content, adjust it
            start = max(start, last_idx_end)

            # append the non-overlapping part to the merged text
            merged_text += doc.content[start - doc.meta["split_idx_start"] :]  # type: ignore

            # update the last end index
            last_idx_end = doc.meta["split_idx_start"] + len(doc.content)  # type: ignore

        return merged_text

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        docstore = self.document_store.to_dict()
        return default_to_dict(self, document_store=docstore, window_size=self.window_size)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SentenceWindowRetriever":
        """
        Deserializes the component from a dictionary.

        :returns:
            Deserialized component.
        """
        # deserialize the document store
        deserialize_document_store_in_init_params_inplace(data)

        # deserialize the component
        return default_from_dict(cls, data)

    @component.output_types(context_windows=List[str], context_documents=List[Document])
    def run(self, retrieved_documents: List[Document], window_size: Optional[int] = None):
        """
        Based on the `source_id` and on the `doc.meta['split_id']` get surrounding documents from the document store.

        Implements the logic behind the sentence-window technique, retrieving the surrounding documents of a given
        document from the document store.

        :param retrieved_documents: List of retrieved documents from the previous retriever.
        :param window_size: The number of documents to retrieve before and after the relevant one. This will overwrite
                            the `window_size` parameter set in the constructor.
        :returns:
            A dictionary with the following keys:
                - `context_windows`: A list of strings, where each string represents the concatenated text from the
                                     context window of the corresponding document in `retrieved_documents`.
                - `context_documents`: A list `Document` objects, containing the retrieved documents plus the context
                                      document surrounding them. The documents are sorted by the `split_idx_start`
                                      meta field.

        """
        window_size = window_size or self.window_size

        if window_size < 1:
            raise ValueError("The window_size parameter must be greater than 0.")

        if not all("split_id" in doc.meta for doc in retrieved_documents):
            raise ValueError("The retrieved documents must have 'split_id' in the metadata.")

        if not all("source_id" in doc.meta for doc in retrieved_documents):
            raise ValueError("The retrieved documents must have 'source_id' in the metadata.")

        context_text = []
        context_documents = []
        for doc in retrieved_documents:
            source_id = doc.meta["source_id"]
            split_id = doc.meta["split_id"]
            min_before = min(list(range(split_id - 1, split_id - window_size - 1, -1)))
            max_after = max(list(range(split_id + 1, split_id + window_size + 1, 1)))
            context_docs = self.document_store.filter_documents(
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.source_id", "operator": "==", "value": source_id},
                        {"field": "meta.split_id", "operator": ">=", "value": min_before},
                        {"field": "meta.split_id", "operator": "<=", "value": max_after},
                    ],
                }
            )
            context_text.append(self.merge_documents_text(context_docs))
            context_docs_sorted = sorted(context_docs, key=lambda doc: doc.meta["split_idx_start"])
            context_documents.extend(context_docs_sorted)

        return {"context_windows": context_text, "context_documents": context_documents}
