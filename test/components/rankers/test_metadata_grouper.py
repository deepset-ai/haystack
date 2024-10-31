from typing import Any, Dict

from haystack import Pipeline
from haystack.dataclasses import Document

from haystack.components.rankers.metadata_grouper import MetaDataGrouper

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


class TestDeepsetMetadataGrouper:
    def test_init_default(self) -> None:
        sample_meta_aggregator = MetaDataGrouper(group_by="group", sort_docs_by=None)
        result = sample_meta_aggregator.run(documents=[])
        assert "documents" in result
        assert result["documents"] == []

    def test_run_group_by_only(self) -> None:
        sample_meta_aggregator = MetaDataGrouper(group_by="group")
        result = sample_meta_aggregator.run(documents=DOC_LIST)
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

    def test_run(self) -> None:
        meta_aggregator = MetaDataGrouper(group_by="group", subgroup_by="subgroup", sort_docs_by="split_id")
        result = meta_aggregator.run(documents=DOC_LIST)

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
        meta_aggregator = MetaDataGrouper(group_by="value_list", subgroup_by="subvaluelist", sort_docs_by="split_id")
        result = meta_aggregator.run(documents=DOC_LIST)
        assert "documents" in result
        assert len(DOC_LIST) == len(result["documents"])
        assert result["documents"][0].content == "list values" and result["documents"][0].meta["value_list"] == ["11"]
        assert result["documents"][1].content == "list values2" and result["documents"][1].meta["value_list"] == ["12"]
        assert result["documents"][2].content == "list values3" and result["documents"][2].meta["value_list"] == ["12"]

    def test_run_in_pipeline_dumps_and_loads(self) -> None:
        meta_aggregator = MetaDataGrouper(group_by="group", sort_docs_by="split_id")
        result_single = meta_aggregator.run(documents=DOC_LIST)
        pipeline = Pipeline()
        pipeline.add_component("meta_aggregator", meta_aggregator)
        pipeline_yaml_str = pipeline.dumps()
        pipeline_reloaded = Pipeline().loads(pipeline_yaml_str)
        result: Dict[str, Any] = pipeline_reloaded.run(data={"documents": DOC_LIST})
        result = result["meta_aggregator"]
        assert result_single == result
