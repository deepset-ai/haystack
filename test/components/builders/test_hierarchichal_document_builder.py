# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from haystack import Document, Pipeline
from haystack.components.builders.hierarchical_doc_builder import HierarchicalDocumentBuilder
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore


class TestHierarchicalDocumentBuilder:
    def test_init_with_default_params(self):
        builder = HierarchicalDocumentBuilder(block_sizes=[100, 200, 300])
        assert builder.block_sizes == [300, 200, 100]
        assert builder.split_overlap == 0
        assert builder.split_by == "word"

    def test_init_with_custom_params(self):
        builder = HierarchicalDocumentBuilder(block_sizes=[100, 200, 300], split_overlap=25, split_by="word")
        assert builder.block_sizes == [300, 200, 100]
        assert builder.split_overlap == 25
        assert builder.split_by == "word"

    def test_init_with_duplicate_block_sizes(self):
        try:
            HierarchicalDocumentBuilder(block_sizes=[100, 200, 200])
        except ValueError as e:
            assert str(e) == "block_sizes must not contain duplicates"

    def test_to_dict(self):
        builder = HierarchicalDocumentBuilder(block_sizes=[100, 200, 300], split_overlap=25, split_by="word")
        expected = builder.to_dict()
        assert expected == {
            "type": "haystack.components.builders.hierarchical_doc_builder.HierarchicalDocumentBuilder",
            "init_parameters": {"block_sizes": [300, 200, 100], "split_overlap": 25, "split_by": "word"},
        }

    def test_from_dict(self):
        data = {
            "type": "haystack.components.builders.hierarchical_doc_builder.HierarchicalDocumentBuilder",
            "init_parameters": {"block_sizes": [10, 5, 2], "split_overlap": 0, "split_by": "word"},
        }

        builder = HierarchicalDocumentBuilder.from_dict(data)
        assert builder.block_sizes == [10, 5, 2]
        assert builder.split_overlap == 0
        assert builder.split_by == "word"

    def test_run(self):
        builder = HierarchicalDocumentBuilder(block_sizes=[10, 5, 2], split_overlap=0, split_by="word")
        text = "one two three four five six seven eight nine ten"
        doc = Document(content=text)
        output = builder.run([doc])
        docs = output["documents"]

        assert len(docs) == 9
        assert docs[0].content == "one two three four five six seven eight nine ten"

        # level 1 - root node
        assert docs[0].meta["level"] == 1
        assert len(docs[0].meta["children_ids"]) == 2

        # level 2 -left branch
        assert docs[1].meta["parent_id"] == docs[0].id
        assert docs[1].meta["level"] == 2
        assert len(docs[1].meta["children_ids"]) == 3

        # level 2 - right branch
        assert docs[2].meta["parent_id"] == docs[0].id
        assert docs[2].meta["level"] == 2
        assert len(docs[2].meta["children_ids"]) == 3

        # level 3 - left branch - leaf nodes
        assert docs[3].meta["parent_id"] == docs[1].id
        assert docs[4].meta["parent_id"] == docs[1].id
        assert docs[5].meta["parent_id"] == docs[1].id
        assert docs[3].meta["level"] == 3
        assert docs[4].meta["level"] == 3
        assert docs[5].meta["level"] == 3
        assert len(docs[3].meta["children_ids"]) == 0
        assert len(docs[4].meta["children_ids"]) == 0
        assert len(docs[5].meta["children_ids"]) == 0

        # level 3 - right branch - leaf nodes
        assert docs[6].meta["parent_id"] == docs[2].id
        assert docs[7].meta["parent_id"] == docs[2].id
        assert docs[8].meta["parent_id"] == docs[2].id
        assert docs[6].meta["level"] == 3
        assert docs[7].meta["level"] == 3
        assert docs[8].meta["level"] == 3
        assert len(docs[6].meta["children_ids"]) == 0
        assert len(docs[7].meta["children_ids"]) == 0
        assert len(docs[8].meta["children_ids"]) == 0

    def test_to_dict_in_pipeline(self):
        pipeline = Pipeline()
        hierarchical_doc_builder = HierarchicalDocumentBuilder(block_sizes=[10, 5, 2])
        doc_store = InMemoryDocumentStore()
        doc_writer = DocumentWriter(document_store=doc_store)
        pipeline.add_component(name="hierarchical_doc_builder", instance=hierarchical_doc_builder)
        pipeline.add_component(name="doc_writer", instance=doc_writer)
        pipeline.connect("hierarchical_doc_builder", "doc_writer")
        expected = pipeline.to_dict()

        assert expected.keys() == {"metadata", "max_loops_allowed", "components", "connections"}
        assert expected["components"].keys() == {"hierarchical_doc_builder", "doc_writer"}
        assert expected["components"]["hierarchical_doc_builder"] == {
            "type": "haystack.components.builders.hierarchical_doc_builder.HierarchicalDocumentBuilder",
            "init_parameters": {"block_sizes": [10, 5, 2], "split_overlap": 0, "split_by": "word"},
        }

    def test_from_dict_in_pipeline(self):
        data = {
            "metadata": {},
            "max_loops_allowed": 100,
            "components": {
                "hierarchical_doc_builder": {
                    "type": "haystack.components.builders.hierarchical_doc_builder.HierarchicalDocumentBuilder",
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
            "connections": [{"sender": "hierarchical_doc_builder.documents", "receiver": "doc_writer.documents"}],
        }

        assert Pipeline.from_dict(data)

    @pytest.mark.integration
    def test_example_in_pipeline(self):
        pipeline = Pipeline()
        hierarchical_doc_builder = HierarchicalDocumentBuilder(block_sizes=[10, 5, 2], split_overlap=0, split_by="word")
        doc_store = InMemoryDocumentStore()
        doc_writer = DocumentWriter(document_store=doc_store)

        pipeline.add_component(name="hierarchical_doc_builder", instance=hierarchical_doc_builder)
        pipeline.add_component(name="doc_writer", instance=doc_writer)
        pipeline.connect("hierarchical_doc_builder.documents", "doc_writer")

        text = "one two three four five six seven eight nine ten"
        doc = Document(content=text)
        docs = pipeline.run({"hierarchical_doc_builder": {"documents": [doc]}})

        assert docs["doc_writer"]["documents_written"] == 9
        assert len(doc_store.storage.values()) == 9
