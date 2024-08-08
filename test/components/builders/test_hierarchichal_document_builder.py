# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from haystack import Document
from haystack.components.builders.hierarchical_doc_builder import HierarchicalDocumentBuilder


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

        builder = HierarchicalDocumentBuilder.from_dict(data["init_parameters"])
        assert builder.block_sizes == [10, 5, 2]
        assert builder.split_overlap == 0
        assert builder.split_by == "word"

    def test_run(self):
        builder = HierarchicalDocumentBuilder(block_sizes=[10, 5, 2], split_overlap=0, split_by="word")
        text = "one two three four five six seven eight nine ten"
        doc = Document(content=text)
        docs = builder.run([doc])

        assert len(docs) == 9
        assert docs[0].content == "one two three four five six seven eight nine ten"

        # root node
        assert len(docs[0].children_ids) == 2

        # level 1
        assert len(docs[1].children_ids) == 3  # left branch
        assert len(docs[2].children_ids) == 3  # right branch

        # level 2 - leaf nodes - left branch
        assert len(docs[3].children_ids) == 0
        assert len(docs[4].children_ids) == 0
        assert len(docs[5].children_ids) == 0
        assert docs[3].parent_id == docs[1].id
        assert docs[4].parent_id == docs[1].id
        assert docs[5].parent_id == docs[1].id
        assert docs[3].level == 3
        assert docs[4].level == 3
        assert docs[5].level == 3

        # level 2 - leaf nodes - right branch
        assert len(docs[6].children_ids) == 0
        assert len(docs[7].children_ids) == 0
        assert len(docs[8].children_ids) == 0
        assert docs[6].parent_id == docs[2].id
        assert docs[7].parent_id == docs[2].id
        assert docs[8].parent_id == docs[2].id
        assert docs[6].level == 3
        assert docs[7].level == 3
        assert docs[8].level == 3

    def test_example_in_pipeline(self):
        pass
