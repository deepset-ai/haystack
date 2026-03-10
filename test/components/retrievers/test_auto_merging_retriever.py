# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document, Pipeline
from haystack.components.preprocessors import HierarchicalDocumentSplitter
from haystack.components.retrievers import InMemoryBM25Retriever
from haystack.components.retrievers.auto_merging_retriever import AutoMergingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore


class TestAutoMergingRetriever:
    def test_init_default(self):
        retriever = AutoMergingRetriever(InMemoryDocumentStore())
        assert retriever.threshold == 0.5

    def test_init_with_parameters(self):
        retriever = AutoMergingRetriever(InMemoryDocumentStore(), threshold=0.7)
        assert retriever.threshold == 0.7

    def test_init_with_invalid_threshold(self):
        with pytest.raises(ValueError):
            AutoMergingRetriever(InMemoryDocumentStore(), threshold=-2)

    def test_run_missing_parent_id(self):
        docs = [Document(content="test", meta={"__level": 1, "__block_size": 10})]
        retriever = AutoMergingRetriever(InMemoryDocumentStore())
        with pytest.raises(
            ValueError, match="The matched leaf documents do not have the required meta field '__parent_id'"
        ):
            retriever.run(documents=docs)

    def test_run_missing_level(self):
        docs = [Document(content="test", meta={"__parent_id": "parent1", "__block_size": 10})]

        retriever = AutoMergingRetriever(InMemoryDocumentStore())
        with pytest.raises(
            ValueError, match="The matched leaf documents do not have the required meta field '__level'"
        ):
            retriever.run(documents=docs)

    def test_run_missing_block_size(self):
        docs = [Document(content="test", meta={"__parent_id": "parent1", "__level": 1})]

        retriever = AutoMergingRetriever(InMemoryDocumentStore())
        with pytest.raises(
            ValueError, match="The matched leaf documents do not have the required meta field '__block_size'"
        ):
            retriever.run(documents=docs)

    def test_run_mixed_valid_and_invalid_documents(self):
        docs = [
            Document(content="valid", meta={"__parent_id": "parent1", "__level": 1, "__block_size": 10}),
            Document(content="invalid", meta={"__level": 1, "__block_size": 10}),
        ]
        retriever = AutoMergingRetriever(InMemoryDocumentStore())
        with pytest.raises(
            ValueError, match="The matched leaf documents do not have the required meta field '__parent_id'"
        ):
            retriever.run(documents=docs)

    def test_to_dict(self):
        retriever = AutoMergingRetriever(InMemoryDocumentStore(), threshold=0.7)
        expected = retriever.to_dict()
        assert expected["type"] == "haystack.components.retrievers.auto_merging_retriever.AutoMergingRetriever"
        assert expected["init_parameters"]["threshold"] == 0.7
        assert (
            expected["init_parameters"]["document_store"]["type"]
            == "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore"
        )

    def test_from_dict(self):
        data = {
            "type": "haystack.components.retrievers.auto_merging_retriever.AutoMergingRetriever",
            "init_parameters": {
                "document_store": {
                    "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                    "init_parameters": {
                        "bm25_tokenization_regex": "(?u)\\b\\w\\w+\\b",
                        "bm25_algorithm": "BM25L",
                        "bm25_parameters": {},
                        "embedding_similarity_function": "dot_product",
                        "index": "6b122bb4-211b-465e-804d-77c5857bf4c5",
                    },
                },
                "threshold": 0.7,
            },
        }
        retriever = AutoMergingRetriever.from_dict(data)
        assert retriever.threshold == 0.7

    def test_serialization_deserialization_pipeline(self):
        pipeline = Pipeline()
        doc_store_parents = InMemoryDocumentStore()
        bm_25_retriever = InMemoryBM25Retriever(doc_store_parents)
        auto_merging_retriever = AutoMergingRetriever(doc_store_parents, threshold=0.5)

        pipeline.add_component(name="bm_25_retriever", instance=bm_25_retriever)
        pipeline.add_component(name="auto_merging_retriever", instance=auto_merging_retriever)
        pipeline.connect("bm_25_retriever.documents", "auto_merging_retriever.documents")
        pipeline_dict = pipeline.to_dict()

        new_pipeline = Pipeline.from_dict(pipeline_dict)
        assert new_pipeline == pipeline

    def test_run_parent_not_found(self):
        doc_store = InMemoryDocumentStore()
        retriever = AutoMergingRetriever(doc_store, threshold=0.5)

        # a leaf document with a non-existent parent_id
        leaf_doc = Document(
            content="test", meta={"__parent_id": "non_existent_parent", "__level": 1, "__block_size": 10}
        )

        with pytest.raises(ValueError, match="Expected 1 parent document with id non_existent_parent, found 0"):
            retriever.run([leaf_doc])

    def test_run_parent_without_children_metadata(self):
        """Test case where a parent document exists but doesn't have the __children_ids metadata field"""
        doc_store = InMemoryDocumentStore()

        # Create and store a parent document without __children_ids metadata
        parent_doc = Document(
            content="parent content",
            id="parent1",
            meta={
                "__level": 1,  # Add other required metadata
                "__block_size": 10,
            },
        )
        doc_store.write_documents([parent_doc])

        retriever = AutoMergingRetriever(doc_store, threshold=0.5)

        # Create a leaf document that points to this parent
        leaf_doc = Document(content="leaf content", meta={"__parent_id": "parent1", "__level": 2, "__block_size": 5})

        with pytest.raises(ValueError, match="Parent document with id parent1 does not have any children"):
            retriever.run([leaf_doc])

    def test_run_empty_documents(self):
        retriever = AutoMergingRetriever(InMemoryDocumentStore())
        assert retriever.run([]) == {"documents": []}

    def test_run_return_parent_document(self):
        text = "The sun rose early in the morning. It cast a warm glow over the trees. Birds began to sing."

        docs = [Document(content=text)]
        builder = HierarchicalDocumentSplitter(block_sizes={10, 3}, split_overlap=0, split_by="word")
        docs = builder.run(docs)

        # store all non-leaf documents
        doc_store_parents = InMemoryDocumentStore()
        for doc in docs["documents"]:
            if doc.meta["__children_ids"]:
                doc_store_parents.write_documents([doc])
        retriever = AutoMergingRetriever(doc_store_parents, threshold=0.5)

        # assume we retrieved 2 leaf docs from the same parent, the parent document should be returned,
        # since it has 3 children and the threshold=0.5, and we retrieved 2 children (2/3 > 0.66(6))
        leaf_docs = [doc for doc in docs["documents"] if not doc.meta["__children_ids"]]
        docs = retriever.run(leaf_docs[4:6])
        assert len(docs["documents"]) == 1
        assert docs["documents"][0].content == "warm glow over the trees. Birds began to sing."
        assert len(docs["documents"][0].meta["__children_ids"]) == 3

    def test_run_return_leafs_document(self):
        docs = [Document(content="The monarch of the wild blue yonder rises from the eastern side of the horizon.")]
        builder = HierarchicalDocumentSplitter(block_sizes={10, 3}, split_overlap=0, split_by="word")
        docs = builder.run(docs)

        doc_store_parents = InMemoryDocumentStore()
        for doc in docs["documents"]:
            if doc.meta["__level"] == 1:
                doc_store_parents.write_documents([doc])

        leaf_docs = [doc for doc in docs["documents"] if not doc.meta["__children_ids"]]
        retriever = AutoMergingRetriever(doc_store_parents, threshold=0.6)
        result = retriever.run([leaf_docs[4]])

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "eastern side of "
        assert result["documents"][0].meta["__parent_id"] == docs["documents"][2].id

    def test_run_return_leafs_document_different_parents(self):
        docs = [Document(content="The monarch of the wild blue yonder rises from the eastern side of the horizon.")]
        builder = HierarchicalDocumentSplitter(block_sizes={10, 3}, split_overlap=0, split_by="word")
        docs = builder.run(docs)

        doc_store_parents = InMemoryDocumentStore()
        for doc in docs["documents"]:
            if doc.meta["__level"] == 1:
                doc_store_parents.write_documents([doc])

        leaf_docs = [doc for doc in docs["documents"] if not doc.meta["__children_ids"]]
        retriever = AutoMergingRetriever(doc_store_parents, threshold=0.6)
        result = retriever.run([leaf_docs[4], leaf_docs[3]])

        assert len(result["documents"]) == 2
        assert result["documents"][0].meta["__parent_id"] != result["documents"][1].meta["__parent_id"]

    def test_run_go_up_hierarchy_multiple_levels(self):
        """
        Test if the retriever can go up the hierarchy multiple levels to find the parent document.

        Simulate a scenario where we have 4 leaf-documents that matched some initial query. The leaf-documents
        are continuously merged up the hierarchy until the threshold is no longer met.
        In this case it goes from the 4th level in the hierarchy up the 1st level.
        """
        text = "The sun rose early in the morning. It cast a warm glow over the trees. Birds began to sing."

        docs = [Document(content=text)]
        builder = HierarchicalDocumentSplitter(block_sizes={6, 4, 2, 1}, split_overlap=0, split_by="word")
        docs = builder.run(docs)

        # store all non-leaf documents
        doc_store_parents = InMemoryDocumentStore()
        for doc in docs["documents"]:
            if doc.meta["__children_ids"]:
                doc_store_parents.write_documents([doc])
        retriever = AutoMergingRetriever(doc_store_parents, threshold=0.4)

        # simulate a scenario where we have 4 leaf-documents that matched some initial query
        retrieved_leaf_docs = [d for d in docs["documents"] if d.content in {"The ", "sun ", "rose ", "early "}]

        result = retriever.run(retrieved_leaf_docs)

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "The sun rose early in the "

    def test_run_go_up_hierarchy_multiple_levels_hit_root_document(self):
        """
        Test case where we go up hierarchy until the root document, so the root document is returned.

        It's the only document in the hierarchy which has no parent.
        """
        text = "The sun rose early in the morning. It cast a warm glow over the trees. Birds began to sing."

        docs = [Document(content=text)]
        builder = HierarchicalDocumentSplitter(block_sizes={6, 4}, split_overlap=0, split_by="word")
        docs = builder.run(docs)

        # store all non-leaf documents
        doc_store_parents = InMemoryDocumentStore()
        for doc in docs["documents"]:
            if doc.meta["__children_ids"]:
                doc_store_parents.write_documents([doc])
        retriever = AutoMergingRetriever(doc_store_parents, threshold=0.1)  # set a low threshold to hit root document

        # simulate a scenario where we have 4 leaf-documents that matched some initial query
        retrieved_leaf_docs = [
            d
            for d in docs["documents"]
            if d.content in {"The sun rose early ", "in the ", "morning. It cast a ", "over the trees. Birds "}
        ]

        result = retriever.run(retrieved_leaf_docs)

        assert len(result["documents"]) == 1
        assert result["documents"][0].meta["__level"] == 0  # hit root document
