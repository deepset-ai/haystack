# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.document_stores.types import DocumentStore
from haystack.utils import deserialize_document_store_in_init_params_inplace

logger = logging.getLogger(__name__)


@component
class SentenceWindowRetriever:
    """
    Retrieves neighboring documents from a DocumentStore to provide context for query results.

    This component is intended to be used after a Retriever (e.g., BM25Retriever, EmbeddingRetriever).
    It enhances retrieved results by fetching adjacent document chunks to give
    additional context for the user.

    The documents must include metadata indicating their origin and position:
    - `source_id` is used to group sentence chunks belonging to the same original document.
    - `split_id` represents the position/order of the chunk within the document.

    The number of adjacent documents to include on each side of the retrieved document can be configured using the
    `window_size` parameter. You can also specify which metadata fields to use for source and split ID
    via `source_id_meta_field` and `split_id_meta_field`.

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

    def __init__(
        self,
        document_store: DocumentStore,
        window_size: int = 3,
        *,
        source_id_meta_field: str = "source_id",
        split_id_meta_field: str = "split_id",
        raise_on_missing_meta_fields: bool = True,
    ):
        """
        Creates a new SentenceWindowRetriever component.

        :param document_store: The Document Store to retrieve the surrounding documents from.
        :param window_size: The number of documents to retrieve before and after the relevant one.
                For example, `window_size: 2` fetches 2 preceding and 2 following documents.
        :param source_id_meta_field: The metadata field that contains the source ID of the document.
        :param split_id_meta_field: The metadata field that contains the split ID of the document.
        :param raise_on_missing_meta_fields: If True, raises an error if the documents do not contain the required
            metadata fields. If False, it will skip retrieving the context for documents that are missing
            the required metadata fields, but will still include the original document in the results.
        """
        if window_size < 1:
            raise ValueError("The window_size parameter must be greater than 0.")

        self.window_size = window_size
        self.document_store = document_store
        self.source_id_meta_field = source_id_meta_field
        self.split_id_meta_field = split_id_meta_field
        self.raise_on_missing_meta_fields = raise_on_missing_meta_fields

    @staticmethod
    def merge_documents_text(documents: List[Document]) -> str:
        """
        Merge a list of document text into a single string.

        This functions concatenates the textual content of a list of documents into a single string, eliminating any
        overlapping content.

        :param documents: List of Documents to merge.
        """
        if any("split_idx_start" not in doc.meta for doc in documents):
            # If any of the documents is missing the 'split_idx_start' metadata we just concatenate their content.
            return "".join(doc.content for doc in documents if doc.content)

        sorted_docs = sorted(documents, key=lambda doc: doc.meta["split_idx_start"])
        merged_text = ""
        last_idx_end = 0
        for doc in sorted_docs:
            if doc.content is None:
                continue

            start = doc.meta.get("split_idx_start", 0)  # start of the current content

            # if the start of the current content is before the end of the last appended content, adjust it
            start = max(start, last_idx_end)

            # append the non-overlapping part to the merged text
            merged_text += doc.content[start - int(doc.meta["split_idx_start"]) :]

            # update the last end index
            last_idx_end = int(doc.meta["split_idx_start"]) + len(doc.content)

        return merged_text

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        docstore = self.document_store.to_dict()
        return default_to_dict(
            self,
            document_store=docstore,
            window_size=self.window_size,
            source_id_meta_field=self.source_id_meta_field,
            split_id_meta_field=self.split_id_meta_field,
            raise_on_missing_meta_fields=self.raise_on_missing_meta_fields,
        )

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

        if (
            not all(self.split_id_meta_field in doc.meta for doc in retrieved_documents)
            and self.raise_on_missing_meta_fields
        ):
            raise ValueError(f"The retrieved documents must have '{self.split_id_meta_field}' in their metadata.")

        if (
            not all(self.source_id_meta_field in doc.meta for doc in retrieved_documents)
            and self.raise_on_missing_meta_fields
        ):
            raise ValueError(f"The retrieved documents must have '{self.source_id_meta_field}' in their metadata.")

        context_text = []
        context_documents = []
        for doc in retrieved_documents:
            source_id = doc.meta.get(self.source_id_meta_field)
            split_id = doc.meta.get(self.split_id_meta_field)

            if source_id is None or split_id is None:
                logger.warning(
                    "Document {doc_id} is missing required metadata fields to be used with "
                    "SentenceWindowRetriever: {source_id} or {split_id}. Skipping context retrieval for this document.",
                    doc_id=doc.id,
                    source_id=source_id,
                    split_id=split_id,
                )
                context_text.append(doc.content or "")
                context_documents.append(doc)
                continue

            min_before = min(list(range(split_id - 1, split_id - window_size - 1, -1)))
            max_after = max(list(range(split_id + 1, split_id + window_size + 1, 1)))
            context_docs = self.document_store.filter_documents(
                {
                    "operator": "AND",
                    "conditions": [
                        {"field": f"meta.{self.source_id_meta_field}", "operator": "==", "value": source_id},
                        {"field": f"meta.{self.split_id_meta_field}", "operator": ">=", "value": min_before},
                        {"field": f"meta.{self.split_id_meta_field}", "operator": "<=", "value": max_after},
                    ],
                }
            )
            context_text.append(self.merge_documents_text(context_docs))
            context_docs_sorted = sorted(context_docs, key=lambda doc: doc.meta[self.split_id_meta_field])
            context_documents.extend(context_docs_sorted)

        return {"context_windows": context_text, "context_documents": context_documents}
