from pathlib import Path

import json
import pytest

from haystack.pipelines import Pipeline, RootNode, DocumentSearchPipeline
from haystack.nodes import FARMReader, BM25Retriever, JoinDocuments

from ..conftest import MockRetriever as BaseMockRetriever, MockReader


class MockRetriever(BaseMockRetriever):
    def retrieve(self, *args, **kwargs):
        top_k = None
        if "top_k" in kwargs.keys():
            top_k = kwargs["top_k"]
        elif len(args) > 0:
            top_k = args[-1]

        if top_k and not isinstance(top_k, int):
            raise ValueError("TEST ERROR!")


@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_node_names_validation(document_store_with_docs, tmp_path):
    pipeline = Pipeline()
    pipeline.add_node(
        component=BM25Retriever(document_store=document_store_with_docs), name="Retriever", inputs=["Query"]
    )
    pipeline.add_node(
        component=FARMReader(model_name_or_path="deepset/minilm-uncased-squad2", num_processes=0),
        name="Reader",
        inputs=["Retriever"],
    )

    with pytest.raises(ValueError) as exc_info:
        pipeline.run(
            query="Who lives in Berlin?",
            params={
                "Reader": {"top_k": 3},
                "non-existing-node": {"top_k": 10},
                "top_k": 5,
                "non-existing-global_param": "wrong",
            },
            debug=True,
        )
    exception_raised = str(exc_info.value)
    assert "non-existing-node" in exception_raised
    assert "non-existing-global_param" in exception_raised
    assert "Reader" not in exception_raised
    assert "top_k" not in exception_raised


@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_debug_attributes_global(document_store_with_docs, tmp_path):
    bm25_retriever = BM25Retriever(document_store=document_store_with_docs)
    reader = FARMReader(model_name_or_path="deepset/minilm-uncased-squad2", num_processes=0)

    pipeline = Pipeline()
    pipeline.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
    pipeline.add_node(component=reader, name="Reader", inputs=["BM25Retriever"])

    prediction = pipeline.run(
        query="Who lives in Berlin?", params={"BM25Retriever": {"top_k": 10}, "Reader": {"top_k": 3}}, debug=True
    )
    assert "_debug" in prediction.keys()
    assert "BM25Retriever" in prediction["_debug"].keys()
    assert "Reader" in prediction["_debug"].keys()
    assert "input" in prediction["_debug"]["BM25Retriever"].keys()
    assert "output" in prediction["_debug"]["BM25Retriever"].keys()
    assert "exec_time_ms" in prediction["_debug"]["BM25Retriever"].keys()
    assert "input" in prediction["_debug"]["Reader"].keys()
    assert "output" in prediction["_debug"]["Reader"].keys()
    assert "exec_time_ms" in prediction["_debug"]["Reader"].keys()
    assert prediction["_debug"]["BM25Retriever"]["input"]
    assert prediction["_debug"]["BM25Retriever"]["output"]
    assert prediction["_debug"]["BM25Retriever"]["exec_time_ms"] is not None
    assert prediction["_debug"]["Reader"]["input"]
    assert prediction["_debug"]["Reader"]["output"]
    assert prediction["_debug"]["Reader"]["exec_time_ms"] is not None

    # Avoid circular reference: easiest way to detect those is to use json.dumps
    json.dumps(prediction, default=str)


@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_debug_attributes_per_node(document_store_with_docs, tmp_path):
    bm25_retriever = BM25Retriever(document_store=document_store_with_docs)
    reader = FARMReader(model_name_or_path="deepset/minilm-uncased-squad2", num_processes=0)

    pipeline = Pipeline()
    pipeline.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
    pipeline.add_node(component=reader, name="Reader", inputs=["BM25Retriever"])

    prediction = pipeline.run(
        query="Who lives in Berlin?", params={"BM25Retriever": {"top_k": 10, "debug": True}, "Reader": {"top_k": 3}}
    )
    assert "_debug" in prediction.keys()
    assert "BM25Retriever" in prediction["_debug"].keys()
    assert "Reader" not in prediction["_debug"].keys()
    assert "input" in prediction["_debug"]["BM25Retriever"].keys()
    assert "output" in prediction["_debug"]["BM25Retriever"].keys()
    assert prediction["_debug"]["BM25Retriever"]["input"]
    assert prediction["_debug"]["BM25Retriever"]["output"]

    # Avoid circular reference: easiest way to detect those is to use json.dumps
    json.dumps(prediction, default=str)


@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_debug_attributes_for_join_nodes(document_store_with_docs, tmp_path):
    bm25_retriever_1 = BM25Retriever(document_store=document_store_with_docs)
    bm25_retriever_2 = BM25Retriever(document_store=document_store_with_docs)

    pipeline = Pipeline()
    pipeline.add_node(component=bm25_retriever_1, name="BM25Retriever1", inputs=["Query"])
    pipeline.add_node(component=bm25_retriever_2, name="BM25Retriever2", inputs=["Query"])
    pipeline.add_node(component=JoinDocuments(), name="JoinDocuments", inputs=["BM25Retriever1", "BM25Retriever2"])

    prediction = pipeline.run(query="Who lives in Berlin?", debug=True)
    assert "_debug" in prediction.keys()
    assert "BM25Retriever1" in prediction["_debug"].keys()
    assert "BM25Retriever2" in prediction["_debug"].keys()
    assert "JoinDocuments" in prediction["_debug"].keys()
    assert "input" in prediction["_debug"]["BM25Retriever1"].keys()
    assert "output" in prediction["_debug"]["BM25Retriever1"].keys()
    assert "input" in prediction["_debug"]["BM25Retriever2"].keys()
    assert "output" in prediction["_debug"]["BM25Retriever2"].keys()
    assert "input" in prediction["_debug"]["JoinDocuments"].keys()
    assert "output" in prediction["_debug"]["JoinDocuments"].keys()
    assert prediction["_debug"]["BM25Retriever1"]["input"]
    assert prediction["_debug"]["BM25Retriever1"]["output"]
    assert prediction["_debug"]["BM25Retriever2"]["input"]
    assert prediction["_debug"]["BM25Retriever2"]["output"]
    assert prediction["_debug"]["JoinDocuments"]["input"]
    assert prediction["_debug"]["JoinDocuments"]["output"]

    # Avoid circular reference: easiest way to detect those is to use json.dumps
    json.dumps(prediction, default=str)


@pytest.mark.parametrize("document_store_with_docs", ["memory"], indirect=True)
def test_global_debug_attributes_override_node_ones(document_store_with_docs, tmp_path):
    bm25_retriever = BM25Retriever(document_store=document_store_with_docs)
    reader = FARMReader(model_name_or_path="deepset/minilm-uncased-squad2", num_processes=0)

    pipeline = Pipeline()
    pipeline.add_node(component=bm25_retriever, name="BM25Retriever", inputs=["Query"])
    pipeline.add_node(component=reader, name="Reader", inputs=["BM25Retriever"])

    prediction = pipeline.run(
        query="Who lives in Berlin?",
        params={"BM25Retriever": {"top_k": 10, "debug": True}, "Reader": {"top_k": 3, "debug": True}},
        debug=False,
    )
    assert "_debug" not in prediction.keys()

    prediction = pipeline.run(
        query="Who lives in Berlin?",
        params={"BM25Retriever": {"top_k": 10, "debug": False}, "Reader": {"top_k": 3, "debug": False}},
        debug=True,
    )
    assert "_debug" in prediction.keys()
    assert "BM25Retriever" in prediction["_debug"].keys()
    assert "Reader" in prediction["_debug"].keys()
    assert "input" in prediction["_debug"]["BM25Retriever"].keys()
    assert "output" in prediction["_debug"]["BM25Retriever"].keys()
    assert "input" in prediction["_debug"]["Reader"].keys()
    assert "output" in prediction["_debug"]["Reader"].keys()
    assert prediction["_debug"]["BM25Retriever"]["input"]
    assert prediction["_debug"]["BM25Retriever"]["output"]
    assert prediction["_debug"]["Reader"]["input"]
    assert prediction["_debug"]["Reader"]["output"]


@pytest.mark.unit
def test_missing_top_level_arg():
    pipeline = Pipeline()
    pipeline.add_node(component=MockRetriever(), name="Retriever", inputs=["Query"])
    pipeline.add_node(component=MockReader(), name="Reader", inputs=["Retriever"])

    with pytest.raises(Exception) as exc:
        pipeline.run(params={"Retriever": {"top_k": 10}})
    assert "Must provide a 'query' parameter" in str(exc.value)


@pytest.mark.unit
def test_unexpected_top_level_arg():
    pipeline = Pipeline()
    pipeline.add_node(component=MockRetriever(), name="Retriever", inputs=["Query"])
    pipeline.add_node(component=MockReader(), name="Reader", inputs=["Retriever"])

    with pytest.raises(Exception) as exc:
        pipeline.run(invalid_query="Who made the PDF specification?", params={"Retriever": {"top_k": 10}})
    assert "run() got an unexpected keyword argument 'invalid_query'" in str(exc.value)


@pytest.mark.unit
def test_unexpected_node_arg():
    pipeline = Pipeline()
    pipeline.add_node(component=MockRetriever(), name="Retriever", inputs=["Query"])
    pipeline.add_node(component=MockReader(), name="Reader", inputs=["Retriever"])

    with pytest.raises(Exception) as exc:
        pipeline.run(query="Who made the PDF specification?", params={"Retriever": {"invalid": 10}})
    assert "Invalid parameter 'invalid' for the node 'Retriever'" in str(exc.value)


@pytest.mark.unit
def test_debug_info_propagation():
    class A(RootNode):
        def run(self):
            test = "A"
            return {"test": test, "_debug": {"debug_key_a": "debug_value_a"}}, "output_1"

    class B(RootNode):
        def run(self, test):
            test += "B"
            return {"test": test, "_debug": "debug_value_b"}, "output_1"

    class C(RootNode):
        def run(self, test):
            test += "C"
            return {"test": test}, "output_1"

    class D(RootNode):
        def run(self, test, _debug):
            test += "C"
            assert _debug["B"]["runtime"] == "debug_value_b"
            return {"test": test}, "output_1"

    pipeline = Pipeline()
    pipeline.add_node(name="A", component=A(), inputs=["Query"])
    pipeline.add_node(name="B", component=B(), inputs=["A"])
    pipeline.add_node(name="C", component=C(), inputs=["B"])
    pipeline.add_node(name="D", component=D(), inputs=["C"])
    output = pipeline.run(query="test")
    assert output["_debug"]["A"]["runtime"]["debug_key_a"] == "debug_value_a"
    assert output["_debug"]["B"]["runtime"] == "debug_value_b"
