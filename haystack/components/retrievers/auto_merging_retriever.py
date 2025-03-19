# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Any, Dict, List

from haystack import Document, component, default_to_dict
from haystack.core.serialization import default_from_dict
from haystack.document_stores.types import DocumentStore
from haystack.utils import deserialize_document_store_in_init_params_inplace


@component
class AutoMergingRetriever:
    """
    A retriever which returns parent documents of the matched leaf nodes documents, based on a threshold setting.

    The AutoMergingRetriever assumes you have a hierarchical tree structure of documents, where the leaf nodes
    are indexed in a document store. See the HierarchicalDocumentSplitter for more information on how to create
    such a structure. During retrieval, if the number of matched leaf documents below the same parent is
    higher than a defined threshold, the retriever will return the parent document instead of the individual leaf
    documents.

    The rational is, given that a paragraph is split into multiple chunks represented as leaf documents, and if for
    a given query, multiple chunks are matched, the whole paragraph might be more informative than the individual
    chunks alone.

    Currently the AutoMergingRetriever can only be used by the following DocumentStores:
    - [AstraDB](https://haystack.deepset.ai/integrations/astradb)
    - [ElasticSearch](https://haystack.deepset.ai/docs/latest/documentstore/elasticsearch)
    - [OpenSearch](https://haystack.deepset.ai/docs/latest/documentstore/opensearch)
    - [PGVector](https://haystack.deepset.ai/docs/latest/documentstore/pgvector)
    - [Qdrant](https://haystack.deepset.ai/docs/latest/documentstore/qdrant)

    ```python
    from haystack import Document
    from haystack.components.preprocessors import HierarchicalDocumentSplitter
    from haystack.components.retrievers.auto_merging_retriever import AutoMergingRetriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    # create a hierarchical document structure with 3 levels, where the parent document has 3 children
    text = "The sun rose early in the morning. It cast a warm glow over the trees. Birds began to sing."
    original_document = Document(content=text)
    builder = HierarchicalDocumentSplitter(block_sizes=[10, 3], split_overlap=0, split_by="word")
    docs = builder.run([original_document])["documents"]

    # store level-1 parent documents and initialize the retriever
    doc_store_parents = InMemoryDocumentStore()
    for doc in docs["documents"]:
        if doc.meta["children_ids"] and doc.meta["level"] == 1:
            doc_store_parents.write_documents([doc])
    retriever = AutoMergingRetriever(doc_store_parents, threshold=0.5)

    # assume we retrieved 2 leaf docs from the same parent, the parent document should be returned,
    # since it has 3 children and the threshold=0.5, and we retrieved 2 children (2/3 > 0.66(6))
    leaf_docs = [doc for doc in docs["documents"] if not doc.meta["children_ids"]]
    docs = retriever.run(leaf_docs[4:6])
    >> {'documents': [Document(id=538..),
    >> content: 'warm glow over the trees. Birds began to sing.',
    >> meta: {'block_size': 10, 'parent_id': '835..', 'children_ids': ['c17...', '3ff...', '352...'], 'level': 1, 'source_id': '835...',
    >> 'page_number': 1, 'split_id': 1, 'split_idx_start': 45})]}
    ```
    """  # noqa: E501

    def __init__(self, document_store: DocumentStore, threshold: float = 0.5):
        """
        Initialize the AutoMergingRetriever.

        :param document_store: DocumentStore from which to retrieve the parent documents
        :param threshold: Threshold to decide whether the parent instead of the individual documents is returned
        """

        if not 0 < threshold < 1:
            raise ValueError("The threshold parameter must be between 0 and 1.")

        self.document_store = document_store
        self.threshold = threshold

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        docstore = self.document_store.to_dict()
        return default_to_dict(self, document_store=docstore, threshold=self.threshold)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AutoMergingRetriever":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary with serialized data.
        :returns:
            An instance of the component.
        """
        deserialize_document_store_in_init_params_inplace(data)
        return default_from_dict(cls, data)

    @staticmethod
    def _check_valid_documents(matched_leaf_documents: List[Document]):
        # check if the matched leaf documents have the required meta fields
        if not all(doc.meta.get("__parent_id") for doc in matched_leaf_documents):
            raise ValueError("The matched leaf documents do not have the required meta field '__parent_id'")

        if not all(doc.meta.get("__level") for doc in matched_leaf_documents):
            raise ValueError("The matched leaf documents do not have the required meta field '__level'")

        if not all(doc.meta.get("__block_size") for doc in matched_leaf_documents):
            raise ValueError("The matched leaf documents do not have the required meta field '__block_size'")

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]):
        """
        Run the AutoMergingRetriever.

        Recursively groups documents by their parents and merges them if they meet the threshold,
        continuing up the hierarchy until no more merges are possible.

        :param documents: List of leaf documents that were matched by a retriever
        :returns:
            List of documents (could be a mix of different hierarchy levels)
        """

        AutoMergingRetriever._check_valid_documents(documents)

        def _get_parent_doc(parent_id: str) -> Document:
            parent_docs = self.document_store.filter_documents({"field": "id", "operator": "==", "value": parent_id})
            if len(parent_docs) != 1:
                raise ValueError(f"Expected 1 parent document with id {parent_id}, found {len(parent_docs)}")

            parent_doc = parent_docs[0]
            if not parent_doc.meta.get("__children_ids"):
                raise ValueError(f"Parent document with id {parent_id} does not have any children.")

            return parent_doc

        def _try_merge_level(docs_to_merge: List[Document], docs_to_return: List[Document]) -> List[Document]:
            parent_doc_id_to_child_docs: Dict[str, List[Document]] = defaultdict(list)  # to group documents by parent

            for doc in docs_to_merge:
                if doc.meta.get("__parent_id"):  # only docs that have parents
                    parent_doc_id_to_child_docs[doc.meta["__parent_id"]].append(doc)
                else:
                    docs_to_return.append(doc)  # keep docs that have no parents

            # Process each parent group
            merged_docs = []
            for parent_doc_id, child_docs in parent_doc_id_to_child_docs.items():
                parent_doc = _get_parent_doc(parent_doc_id)

                # Calculate merge score
                score = len(child_docs) / len(parent_doc.meta["__children_ids"])
                if score > self.threshold:
                    merged_docs.append(parent_doc)  # Merge into parent
                else:
                    docs_to_return.extend(child_docs)  # Keep children separate

            # if no new merges were made, we're done
            if not merged_docs:
                return merged_docs + docs_to_return

            # Recursively try to merge the next level
            return _try_merge_level(merged_docs, docs_to_return)

        return {"documents": _try_merge_level(documents, [])}
