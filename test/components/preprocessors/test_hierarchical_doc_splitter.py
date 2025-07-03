# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document, Pipeline
from haystack.components.preprocessors import HierarchicalDocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore


class TestHierarchicalDocumentSplitter:
    def test_init_with_default_params(self):
        builder = HierarchicalDocumentSplitter(block_sizes={100, 200, 300})
        assert builder.block_sizes == [300, 200, 100]
        assert builder.split_overlap == 0
        assert builder.split_by == "word"

    def test_init_with_custom_params(self):
        builder = HierarchicalDocumentSplitter(block_sizes={100, 200, 300}, split_overlap=25, split_by="word")
        assert builder.block_sizes == [300, 200, 100]
        assert builder.split_overlap == 25
        assert builder.split_by == "word"

    def test_to_dict(self):
        builder = HierarchicalDocumentSplitter(block_sizes={100, 200, 300}, split_overlap=25, split_by="word")
        expected = builder.to_dict()
        assert expected == {
            "type": "haystack.components.preprocessors.hierarchical_document_splitter.HierarchicalDocumentSplitter",
            "init_parameters": {"block_sizes": [300, 200, 100], "split_overlap": 25, "split_by": "word"},
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.preprocessors.hierarchical_document_splitter.HierarchicalDocumentSplitter",
            "init_parameters": {"block_sizes": [10, 5, 2], "split_overlap": 0, "split_by": "word"},
        }

        builder = HierarchicalDocumentSplitter.from_dict(data)
        assert builder.block_sizes == [10, 5, 2]
        assert builder.split_overlap == 0
        assert builder.split_by == "word"

    def test_run(self):
        builder = HierarchicalDocumentSplitter(block_sizes={10, 5, 2}, split_overlap=0, split_by="word")
        text = "one two three four five six seven eight nine ten"
        doc = Document(content=text)
        output = builder.run([doc])
        docs = output["documents"]
        builder.run([doc])

        assert len(docs) == 9
        assert docs[0].content == "one two three four five six seven eight nine ten"

        # level 1 - root node
        assert docs[0].meta["__level"] == 0
        assert len(docs[0].meta["__children_ids"]) == 2

        # level 2 -left branch
        assert docs[1].meta["__parent_id"] == docs[0].id
        assert docs[1].meta["__level"] == 1
        assert len(docs[1].meta["__children_ids"]) == 3

        # level 2 - right branch
        assert docs[2].meta["__parent_id"] == docs[0].id
        assert docs[2].meta["__level"] == 1
        assert len(docs[2].meta["__children_ids"]) == 3

        # level 3 - left branch - leaf nodes
        assert docs[3].meta["__parent_id"] == docs[1].id
        assert docs[4].meta["__parent_id"] == docs[1].id
        assert docs[5].meta["__parent_id"] == docs[1].id
        assert docs[3].meta["__level"] == 2
        assert docs[4].meta["__level"] == 2
        assert docs[5].meta["__level"] == 2
        assert len(docs[3].meta["__children_ids"]) == 0
        assert len(docs[4].meta["__children_ids"]) == 0
        assert len(docs[5].meta["__children_ids"]) == 0

        # level 3 - right branch - leaf nodes
        assert docs[6].meta["__parent_id"] == docs[2].id
        assert docs[7].meta["__parent_id"] == docs[2].id
        assert docs[8].meta["__parent_id"] == docs[2].id
        assert docs[6].meta["__level"] == 2
        assert docs[7].meta["__level"] == 2
        assert docs[8].meta["__level"] == 2
        assert len(docs[6].meta["__children_ids"]) == 0
        assert len(docs[7].meta["__children_ids"]) == 0
        assert len(docs[8].meta["__children_ids"]) == 0

    def test_to_dict_in_pipeline(self):
        pipeline = Pipeline()
        hierarchical_doc_builder = HierarchicalDocumentSplitter(block_sizes={10, 5, 2})
        doc_store = InMemoryDocumentStore()
        doc_writer = DocumentWriter(document_store=doc_store)
        pipeline.add_component(name="hierarchical_doc_splitter", instance=hierarchical_doc_builder)
        pipeline.add_component(name="doc_writer", instance=doc_writer)
        pipeline.connect("hierarchical_doc_splitter", "doc_writer")
        expected = pipeline.to_dict()

        assert expected.keys() == {
            "connections",
            "connection_type_validation",
            "components",
            "max_runs_per_component",
            "metadata",
        }

        assert expected["components"].keys() == {"hierarchical_doc_splitter", "doc_writer"}

        assert expected["components"]["hierarchical_doc_splitter"] == {
            "type": "haystack.components.preprocessors.hierarchical_document_splitter.HierarchicalDocumentSplitter",
            "init_parameters": {"block_sizes": [10, 5, 2], "split_overlap": 0, "split_by": "word"},
        }

    def test_from_dict_in_pipeline(self):
        data = {
            "metadata": {},
            "max_runs_per_component": 100,
            "components": {
                "hierarchical_document_splitter": {
                    "type": "haystack.components.preprocessors.hierarchical_document_splitter.HierarchicalDocumentSplitter",  # noqa: E501
                    "init_parameters": {"block_sizes": [10, 5, 2], "split_overlap": 0, "split_by": "word"},
                },
                "doc_writer": {
                    "type": "haystack.components.writers.document_writer.DocumentWriter",
                    "init_parameters": {
                        "document_store": {
                            "type": "haystack.document_stores.in_memory.document_store.InMemoryDocumentStore",
                            "init_parameters": {
                                "bm25_tokenization_regex": "(?u)\\b\\w\\w+\\b",
                                "bm25_algorithm": "BM25L",
                                "bm25_parameters": {},
                                "embedding_similarity_function": "dot_product",
                                "index": "f32ad5bf-43cb-4035-9823-1de1ae9853c1",
                            },
                        },
                        "policy": "NONE",
                    },
                },
            },
            "connections": [{"sender": "hierarchical_document_splitter.documents", "receiver": "doc_writer.documents"}],
        }

        assert Pipeline.from_dict(data)

    @pytest.mark.integration
    def test_example_in_pipeline(self):
        pipeline = Pipeline()
        hierarchical_doc_builder = HierarchicalDocumentSplitter(
            block_sizes={10, 5, 2}, split_overlap=0, split_by="word"
        )
        doc_store = InMemoryDocumentStore()
        doc_writer = DocumentWriter(document_store=doc_store)

        pipeline.add_component(name="hierarchical_doc_splitter", instance=hierarchical_doc_builder)
        pipeline.add_component(name="doc_writer", instance=doc_writer)
        pipeline.connect("hierarchical_doc_splitter.documents", "doc_writer")

        text = "one two three four five six seven eight nine ten"
        doc = Document(content=text)
        docs = pipeline.run({"hierarchical_doc_splitter": {"documents": [doc]}})

        assert docs["doc_writer"]["documents_written"] == 9
        assert len(doc_store.storage.values()) == 9

    def test_serialization_deserialization_pipeline(self):
        pipeline = Pipeline()
        hierarchical_doc_builder = HierarchicalDocumentSplitter(
            block_sizes={10, 5, 2}, split_overlap=0, split_by="word"
        )
        doc_store = InMemoryDocumentStore()
        doc_writer = DocumentWriter(document_store=doc_store)

        pipeline.add_component(name="hierarchical_doc_splitter", instance=hierarchical_doc_builder)
        pipeline.add_component(name="doc_writer", instance=doc_writer)
        pipeline.connect("hierarchical_doc_splitter.documents", "doc_writer")
        pipeline_dict = pipeline.to_dict()

        new_pipeline = Pipeline.from_dict(pipeline_dict)
        assert new_pipeline == pipeline

    def test_split_by_sentence_assure_warm_up_was_called(self):
        pipeline = Pipeline()
        hierarchical_doc_builder = HierarchicalDocumentSplitter(
            block_sizes={10, 5, 2}, split_overlap=0, split_by="sentence"
        )
        doc_store = InMemoryDocumentStore()
        doc_writer = DocumentWriter(document_store=doc_store)

        pipeline.add_component(name="hierarchical_doc_splitter", instance=hierarchical_doc_builder)
        pipeline.add_component(name="doc_writer", instance=doc_writer)
        pipeline.connect("hierarchical_doc_splitter.documents", "doc_writer")

        text = "This is one sentence. This is another sentence. This is the third sentence."
        doc = Document(content=text)
        docs = pipeline.run({"hierarchical_doc_splitter": {"documents": [doc]}})

        assert docs["doc_writer"]["documents_written"] == 3
        assert len(doc_store.storage.values()) == 3

    def test_hierarchical_splitter_multiple_block_sizes(self):
        # Test with three different block sizes
        doc = Document(
            content="This is a simple test document with multiple sentences. It should be split into various sizes. "
            "This helps test the hierarchy."
        )

        # Using three block sizes: 10, 5, 2 words
        splitter = HierarchicalDocumentSplitter(block_sizes={10, 5, 2}, split_overlap=0, split_by="word")
        result = splitter.run([doc])

        documents = result["documents"]

        # Verify root document
        assert len(documents) > 1
        root = documents[0]
        assert root.meta["__level"] == 0
        assert root.meta["__parent_id"] is None

        # Verify level 1 documents (block_size=10)
        level_1_docs = [d for d in documents if d.meta["__level"] == 1]
        for doc in level_1_docs:
            assert doc.meta["__block_size"] == 10
            assert doc.meta["__parent_id"] == root.id

        # Verify level 2 documents (block_size=5)
        level_2_docs = [d for d in documents if d.meta["__level"] == 2]
        for doc in level_2_docs:
            assert doc.meta["__block_size"] == 5
            assert doc.meta["__parent_id"] in [d.id for d in level_1_docs]

        # Verify level 3 documents (block_size=2)
        level_3_docs = [d for d in documents if d.meta["__level"] == 3]
        for doc in level_3_docs:
            assert doc.meta["__block_size"] == 2
            assert doc.meta["__parent_id"] in [d.id for d in level_2_docs]

        # Verify children references
        for doc in documents:
            if doc.meta["__children_ids"]:
                child_ids = doc.meta["__children_ids"]
                children = [d for d in documents if d.id in child_ids]
                for child in children:
                    assert child.meta["__parent_id"] == doc.id
                    assert child.meta["__level"] == doc.meta["__level"] + 1
