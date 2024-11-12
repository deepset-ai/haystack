# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict

from haystack import Pipeline
from haystack.dataclasses import Document

from haystack.components.rankers.meta_field_grouping_ranker import MetaFieldGroupingRanker

DOC_LIST = [
    # regular
    Document(content="Javascript is a popular language", meta={"group": "42", "split_id": 7, "subgroup": "subB"}),
    Document(content="A chromosome is a package of DNA", meta={"group": "314", "split_id": 2, "subgroup": "subC"}),
    Document(content="DNA carries genetic information", meta={"group": "314", "split_id": 1, "subgroup": "subE"}),
    Document(content="Blue whales have a big heart", meta={"group": "11", "split_id": 8, "subgroup": "subF"}),
    Document(content="Python is a popular  language", meta={"group": "42", "split_id": 4, "subgroup": "subB"}),
    Document(content="bla bla bla bla", meta={"split_id": 8, "subgroup": "subG"}),
    Document(content="Java is a popular programming language", meta={"group": "42", "split_id": 3, "subgroup": "subB"}),
    Document(content="An octopus has three hearts", meta={"group": "11", "split_id": 2, "subgroup": "subD"}),
    # without split id
    Document(content="without split id", meta={"group": "11"}),
    Document(content="without split id2", meta={"group": "22", "subgroup": "subI"}),
    Document(content="without split id3", meta={"group": "11"}),
    # with list values in the metadata
    Document(content="list values", meta={"value_list": ["11"], "split_id": 8, "sub_value_list": ["subF"]}),
    Document(content="list values2", meta={"value_list": ["12"], "split_id": 3, "sub_value_list": ["subX"]}),
    Document(content="list values3", meta={"value_list": ["12"], "split_id": 8, "sub_value_list": ["subX"]}),
]


class TestMetaFieldGroupingRanker:
    def test_init_default(self) -> None:
        """
        Test the default initialization of the MetaFieldGroupingRanker component.
        """
        sample_ranker = MetaFieldGroupingRanker(group_by="group", sort_docs_by=None)
        result = sample_ranker.run(documents=[])
        assert "documents" in result
        assert result["documents"] == []

    def test_run_group_by_only(self) -> None:
        """
        Test the MetaFieldGroupingRanker component with only the 'group_by' parameter. No subgroup or sorting is done.
        """
        sample_ranker = MetaFieldGroupingRanker(group_by="group")
        result = sample_ranker.run(documents=DOC_LIST)
        assert "documents" in result
        assert len(DOC_LIST) == len(result["documents"])
        assert result["documents"][0].meta["split_id"] == 7 and result["documents"][0].meta["group"] == "42"
        assert result["documents"][1].meta["split_id"] == 4 and result["documents"][1].meta["group"] == "42"
        assert result["documents"][2].meta["split_id"] == 3 and result["documents"][2].meta["group"] == "42"
        assert result["documents"][3].meta["split_id"] == 2 and result["documents"][3].meta["group"] == "314"
        assert result["documents"][4].meta["split_id"] == 1 and result["documents"][4].meta["group"] == "314"
        assert result["documents"][5].meta["split_id"] == 8 and result["documents"][5].meta["group"] == "11"
        assert result["documents"][6].meta["split_id"] == 2 and result["documents"][6].meta["group"] == "11"
        assert result["documents"][7].content == "without split id" and result["documents"][7].meta["group"] == "11"
        assert result["documents"][8].content == "without split id3" and result["documents"][8].meta["group"] == "11"
        assert result["documents"][9].content == "without split id2" and result["documents"][9].meta["group"] == "22"
        assert result["documents"][10].content == "bla bla bla bla"

    def test_with_group_subgroup_and_sorting(self) -> None:
        """
        Test the MetaFieldGroupingRanker component with all parameters set, i.e.: grouping by 'group', subgrouping by 'subgroup',
        and sorting by 'split_id'.
        """
        ranker = MetaFieldGroupingRanker(group_by="group", subgroup_by="subgroup", sort_docs_by="split_id")
        result = ranker.run(documents=DOC_LIST)

        assert "documents" in result
        assert len(DOC_LIST) == len(result["documents"])
        assert (
            result["documents"][0].meta["subgroup"] == "subB"
            and result["documents"][0].meta["group"] == "42"
            and result["documents"][0].meta["split_id"] == 3
        )
        assert (
            result["documents"][1].meta["subgroup"] == "subB"
            and result["documents"][1].meta["group"] == "42"
            and result["documents"][1].meta["split_id"] == 4
        )
        assert (
            result["documents"][2].meta["subgroup"] == "subB"
            and result["documents"][2].meta["group"] == "42"
            and result["documents"][2].meta["split_id"] == 7
        )
        assert result["documents"][3].meta["subgroup"] == "subC" and result["documents"][3].meta["group"] == "314"
        assert result["documents"][4].meta["subgroup"] == "subE" and result["documents"][4].meta["group"] == "314"
        assert result["documents"][5].meta["subgroup"] == "subF" and result["documents"][6].meta["group"] == "11"
        assert result["documents"][6].meta["subgroup"] == "subD" and result["documents"][5].meta["group"] == "11"
        assert result["documents"][7].content == "without split id" and result["documents"][7].meta["group"] == "11"
        assert result["documents"][8].content == "without split id3" and result["documents"][8].meta["group"] == "11"
        assert result["documents"][9].content == "without split id2" and result["documents"][9].meta["group"] == "22"
        assert result["documents"][10].content == "bla bla bla bla"

    def test_run_with_lists(self) -> None:
        """
        Test if the MetaFieldGroupingRanker component can handle list values in the metadata.
        """
        ranker = MetaFieldGroupingRanker(group_by="value_list", subgroup_by="subvaluelist", sort_docs_by="split_id")
        result = ranker.run(documents=DOC_LIST)
        assert "documents" in result
        assert len(DOC_LIST) == len(result["documents"])
        assert result["documents"][0].content == "list values" and result["documents"][0].meta["value_list"] == ["11"]
        assert result["documents"][1].content == "list values2" and result["documents"][1].meta["value_list"] == ["12"]
        assert result["documents"][2].content == "list values3" and result["documents"][2].meta["value_list"] == ["12"]

    def test_run_empty_input(self) -> None:
        """
        Test the behavior of the MetaFieldGroupingRanker component with an empty list of documents.
        """
        sample_ranker = MetaFieldGroupingRanker(group_by="group")
        result = sample_ranker.run(documents=[])
        assert "documents" in result
        assert result["documents"] == []

    def test_run_missing_metadata_keys(self) -> None:
        """
        Test the behavior of the MetaFieldGroupingRanker component when some documents are missing the required metadata keys.
        """
        docs_with_missing_keys = [
            Document(content="Document without group", meta={"split_id": 1, "subgroup": "subA"}),
            Document(content="Document without subgroup", meta={"group": "42", "split_id": 2}),
            Document(content="Document with all keys", meta={"group": "42", "split_id": 3, "subgroup": "subB"}),
        ]
        sample_ranker = MetaFieldGroupingRanker(group_by="group", subgroup_by="subgroup", sort_docs_by="split_id")
        result = sample_ranker.run(documents=docs_with_missing_keys)
        assert "documents" in result
        assert len(result["documents"]) == 3
        assert result["documents"][0].meta["group"] == "42"
        assert result["documents"][1].meta["group"] == "42"
        assert result["documents"][2].content == "Document without group"

    def test_run_metadata_with_different_data_types(self) -> None:
        """
        Test the behavior of the MetaFieldGroupingRanker component when the metadata values have different data types.
        """
        docs_with_mixed_data_types = [
            Document(content="Document with string group", meta={"group": "42", "split_id": 1, "subgroup": "subA"}),
            Document(content="Document with number group", meta={"group": 42, "split_id": 2, "subgroup": "subB"}),
            Document(content="Document with boolean group", meta={"group": True, "split_id": 3, "subgroup": "subC"}),
        ]
        sample_ranker = MetaFieldGroupingRanker(group_by="group", subgroup_by="subgroup", sort_docs_by="split_id")
        result = sample_ranker.run(documents=docs_with_mixed_data_types)
        assert "documents" in result
        assert len(result["documents"]) == 3
        assert result["documents"][0].meta["group"] == "42"
        assert result["documents"][1].meta["group"] == 42
        assert result["documents"][2].meta["group"] is True

    def test_run_duplicate_documents(self) -> None:
        """
        Test the behavior of the MetaFieldGroupingRanker component when the input contains duplicate documents.
        """
        docs_with_duplicates = [
            Document(content="Duplicate 1", meta={"group": "42", "split_id": 1, "subgroup": "subA"}),
            Document(content="Duplicate 1", meta={"group": "42", "split_id": 1, "subgroup": "subA"}),
            Document(content="Unique document", meta={"group": "42", "split_id": 2, "subgroup": "subB"}),
        ]
        sample_ranker = MetaFieldGroupingRanker(group_by="group", subgroup_by="subgroup", sort_docs_by="split_id")
        result = sample_ranker.run(documents=docs_with_duplicates)
        assert "documents" in result
        assert len(result["documents"]) == 3
        assert result["documents"][0].content == "Duplicate 1"
        assert result["documents"][1].content == "Duplicate 1"
        assert result["documents"][2].content == "Unique document"

    def test_run_in_pipeline_dumps_and_loads(self) -> None:
        """
        Test if the MetaFieldGroupingRanker component can be dumped to a YAML string and reloaded from it.
        """
        ranker = MetaFieldGroupingRanker(group_by="group", sort_docs_by="split_id")
        result_single = ranker.run(documents=DOC_LIST)
        pipeline = Pipeline()
        pipeline.add_component("ranker", ranker)
        pipeline_yaml_str = pipeline.dumps()
        pipeline_reloaded = Pipeline().loads(pipeline_yaml_str)
        result: Dict[str, Any] = pipeline_reloaded.run(data={"documents": DOC_LIST})
        result = result["ranker"]
        assert result_single == result
