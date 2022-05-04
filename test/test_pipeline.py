from pathlib import Path
import os
import json
import platform
import sys
from typing import Tuple

import pytest
from requests import PreparedRequest
import responses
import logging
import yaml
import pandas as pd

from haystack import __version__
from haystack.document_stores.deepsetcloud import DeepsetCloudDocumentStore
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes.other.join_docs import JoinDocuments
from haystack.nodes.base import BaseComponent
from haystack.nodes.retriever.sparse import BM25Retriever
from haystack.pipelines import Pipeline, RootNode
from haystack.pipelines.config import validate_config_strings
from haystack.pipelines.utils import generate_code
from haystack.errors import PipelineConfigError
from haystack.nodes import PreProcessor, TextConverter
from haystack.utils.deepsetcloud import DeepsetCloudError
from haystack import Document, Answer
from haystack.nodes.other.route_documents import RouteDocuments
from haystack.nodes.other.join_answers import JoinAnswers

from .conftest import (
    MOCK_DC,
    DC_API_ENDPOINT,
    DC_API_KEY,
    DC_TEST_INDEX,
    SAMPLES_PATH,
    MockDocumentStore,
    MockRetriever,
    MockNode,
    deepset_cloud_fixture,
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="function")
def reduce_windows_recursion_limit():
    """
    Prevents Windows CI from crashing with Stackoverflow in situations we want to provoke a RecursionError
    """
    is_windows = platform.system() == "Windows"
    default_recursion_limit = sys.getrecursionlimit()
    if is_windows:
        reduced_recursion_limit = default_recursion_limit // 2
        logger.warning(f"Reducing recursion limit to {reduced_recursion_limit}")
        sys.setrecursionlimit(reduced_recursion_limit)
    yield
    if is_windows:
        logger.warning(f"Resetting recursion limit to {default_recursion_limit}")
        sys.setrecursionlimit(default_recursion_limit)


class ParentComponent(BaseComponent):
    outgoing_edges = 1

    def __init__(self, dependent: BaseComponent) -> None:
        super().__init__()

    def run(*args, **kwargs):
        logging.info("ParentComponent run() was called")


class ParentComponent2(BaseComponent):
    outgoing_edges = 1

    def __init__(self, dependent: BaseComponent) -> None:
        super().__init__()

    def run(*args, **kwargs):
        logging.info("ParentComponent2 run() was called")


class ChildComponent(BaseComponent):
    def __init__(self, some_key: str = None) -> None:
        super().__init__()

    def run(*args, **kwargs):
        logging.info("ChildComponent run() was called")


class DummyRetriever(MockRetriever):
    def __init__(self, document_store):
        self.document_store = document_store

    def run(self):
        test = "test"
        return {"test": test}, "output_1"


class JoinNode(RootNode):
    def run(self, output=None, inputs=None):
        if inputs:
            output = ""
            for input_dict in inputs:
                output += input_dict["output"]
        return {"output": output}, "output_1"


#
# Integration tests
#


@pytest.mark.integration
@pytest.mark.elasticsearch
def test_to_code_creates_same_pipelines():
    index_pipeline = Pipeline.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "test.haystack-pipeline.yml", pipeline_name="indexing_pipeline"
    )
    query_pipeline = Pipeline.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "test.haystack-pipeline.yml", pipeline_name="query_pipeline"
    )
    query_pipeline_code = query_pipeline.to_code(pipeline_variable_name="query_pipeline_from_code")
    index_pipeline_code = index_pipeline.to_code(pipeline_variable_name="index_pipeline_from_code")

    exec(query_pipeline_code)
    exec(index_pipeline_code)
    assert locals()["query_pipeline_from_code"] is not None
    assert locals()["index_pipeline_from_code"] is not None
    assert query_pipeline.get_config() == locals()["query_pipeline_from_code"].get_config()
    assert index_pipeline.get_config() == locals()["index_pipeline_from_code"].get_config()


#
# Unit tests
#


def test_get_config_creates_dependent_component():
    child = ChildComponent()
    parent = ParentComponent(dependent=child)
    pipeline = Pipeline()
    pipeline.add_node(component=parent, name="parent", inputs=["Query"])

    expected_pipelines = [{"name": "query", "nodes": [{"name": "parent", "inputs": ["Query"]}]}]
    expected_components = [
        {"name": "parent", "type": "ParentComponent", "params": {"dependent": "ChildComponent"}},
        {"name": "ChildComponent", "type": "ChildComponent", "params": {}},
    ]

    config = pipeline.get_config()
    for expected_pipeline in expected_pipelines:
        assert expected_pipeline in config["pipelines"]
    for expected_component in expected_components:
        assert expected_component in config["components"]


def test_get_config_creates_only_one_dependent_component_referenced_by_multiple_parents():
    child = ChildComponent()
    parent = ParentComponent(dependent=child)
    parent2 = ParentComponent2(dependent=child)
    p_ensemble = Pipeline()
    p_ensemble.add_node(component=parent, name="Parent1", inputs=["Query"])
    p_ensemble.add_node(component=parent2, name="Parent2", inputs=["Query"])
    p_ensemble.add_node(component=JoinDocuments(join_mode="merge"), name="JoinResults", inputs=["Parent1", "Parent2"])

    expected_components = [
        {"name": "Parent1", "type": "ParentComponent", "params": {"dependent": "ChildComponent"}},
        {"name": "ChildComponent", "type": "ChildComponent", "params": {}},
        {"name": "Parent2", "type": "ParentComponent2", "params": {"dependent": "ChildComponent"}},
        {"name": "JoinResults", "type": "JoinDocuments", "params": {"join_mode": "merge"}},
    ]

    expected_pipelines = [
        {
            "name": "query",
            "nodes": [
                {"name": "Parent1", "inputs": ["Query"]},
                {"name": "Parent2", "inputs": ["Query"]},
                {"name": "JoinResults", "inputs": ["Parent1", "Parent2"]},
            ],
        }
    ]

    config = p_ensemble.get_config()
    for expected_pipeline in expected_pipelines:
        assert expected_pipeline in config["pipelines"]
    for expected_component in expected_components:
        assert expected_component in config["components"]


def test_get_config_creates_two_different_dependent_components_of_same_type():
    child_a = ChildComponent(some_key="A")
    child_b = ChildComponent(some_key="B")
    parent = ParentComponent(dependent=child_a)
    parent2 = ParentComponent(dependent=child_b)
    p_ensemble = Pipeline()
    p_ensemble.add_node(component=parent, name="ParentA", inputs=["Query"])
    p_ensemble.add_node(component=parent2, name="ParentB", inputs=["Query"])
    p_ensemble.add_node(component=JoinDocuments(join_mode="merge"), name="JoinResults", inputs=["ParentA", "ParentB"])

    expected_components = [
        {"name": "ParentA", "type": "ParentComponent", "params": {"dependent": "ChildComponent"}},
        {"name": "ChildComponent", "type": "ChildComponent", "params": {"some_key": "A"}},
        {"name": "ParentB", "type": "ParentComponent", "params": {"dependent": "ChildComponent_2"}},
        {"name": "ChildComponent_2", "type": "ChildComponent", "params": {"some_key": "B"}},
        {"name": "JoinResults", "type": "JoinDocuments", "params": {"join_mode": "merge"}},
    ]

    expected_pipelines = [
        {
            "name": "query",
            "nodes": [
                {"name": "ParentA", "inputs": ["Query"]},
                {"name": "ParentB", "inputs": ["Query"]},
                {"name": "JoinResults", "inputs": ["ParentA", "ParentB"]},
            ],
        }
    ]

    config = p_ensemble.get_config()
    for expected_pipeline in expected_pipelines:
        assert expected_pipeline in config["pipelines"]
    for expected_component in expected_components:
        assert expected_component in config["components"]


def test_get_config_reuses_same_dependent_components():
    child = ChildComponent()
    parent = ParentComponent(dependent=child)
    pipeline = Pipeline()
    pipeline.add_node(component=parent, name="parent", inputs=["Query"])
    pipeline.add_node(component=child, name="child", inputs=["parent"])
    config = pipeline.get_config()

    expected_pipelines = [
        {"name": "query", "nodes": [{"name": "parent", "inputs": ["Query"]}, {"name": "child", "inputs": ["parent"]}]}
    ]
    expected_components = [
        {"name": "parent", "type": "ParentComponent", "params": {"dependent": "child"}},
        {"name": "child", "type": "ChildComponent", "params": {}},
    ]

    config = pipeline.get_config()
    for expected_pipeline in expected_pipelines:
        assert expected_pipeline in config["pipelines"]
    for expected_component in expected_components:
        assert expected_component in config["components"]


def test_get_config_creates_different_components_if_instances_differ():
    child_a = ChildComponent()
    child_b = ChildComponent()
    child_c = ChildComponent()
    parent = ParentComponent(dependent=child_a)
    parent2 = ParentComponent(dependent=child_b)
    p_ensemble = Pipeline()
    p_ensemble.add_node(component=parent, name="ParentA", inputs=["Query"])
    p_ensemble.add_node(component=parent2, name="ParentB", inputs=["Query"])
    p_ensemble.add_node(component=child_c, name="Child", inputs=["Query"])

    expected_components = [
        {"name": "ParentA", "type": "ParentComponent", "params": {"dependent": "ChildComponent"}},
        {"name": "ChildComponent", "type": "ChildComponent", "params": {}},
        {"name": "ParentB", "type": "ParentComponent", "params": {"dependent": "ChildComponent_2"}},
        {"name": "ChildComponent_2", "type": "ChildComponent", "params": {}},
        {"name": "Child", "type": "ChildComponent", "params": {}},
    ]

    expected_pipelines = [
        {
            "name": "query",
            "nodes": [
                {"name": "ParentA", "inputs": ["Query"]},
                {"name": "ParentB", "inputs": ["Query"]},
                {"name": "Child", "inputs": ["Query"]},
            ],
        }
    ]

    config = p_ensemble.get_config()
    for expected_pipeline in expected_pipelines:
        assert expected_pipeline in config["pipelines"]
    for expected_component in expected_components:
        assert expected_component in config["components"]


def test_get_config_reuses_same_unnamed_dependent_components():
    child = ChildComponent()
    parent = ParentComponent(dependent=child)
    parent2 = ParentComponent(dependent=child)
    p_ensemble = Pipeline()
    p_ensemble.add_node(component=parent, name="ParentA", inputs=["Query"])
    p_ensemble.add_node(component=parent2, name="ParentB", inputs=["Query"])

    expected_components = [
        {"name": "ParentA", "type": "ParentComponent", "params": {"dependent": "ChildComponent"}},
        {"name": "ChildComponent", "type": "ChildComponent", "params": {}},
        {"name": "ParentB", "type": "ParentComponent", "params": {"dependent": "ChildComponent"}},
    ]

    expected_pipelines = [
        {"name": "query", "nodes": [{"name": "ParentA", "inputs": ["Query"]}, {"name": "ParentB", "inputs": ["Query"]}]}
    ]

    config = p_ensemble.get_config()
    for expected_pipeline in expected_pipelines:
        assert expected_pipeline in config["pipelines"]
    for expected_component in expected_components:
        assert expected_component in config["components"]


def test_get_config_multi_level_dependencies():
    child = ChildComponent()
    intermediate = ParentComponent(dependent=child)
    parent = ParentComponent(dependent=intermediate)
    p_ensemble = Pipeline()
    p_ensemble.add_node(component=parent, name="Parent", inputs=["Query"])

    expected_components = [
        {"name": "Parent", "type": "ParentComponent", "params": {"dependent": "ParentComponent"}},
        {"name": "ChildComponent", "type": "ChildComponent", "params": {}},
        {"name": "ParentComponent", "type": "ParentComponent", "params": {"dependent": "ChildComponent"}},
    ]

    expected_pipelines = [{"name": "query", "nodes": [{"name": "Parent", "inputs": ["Query"]}]}]

    config = p_ensemble.get_config()
    for expected_pipeline in expected_pipelines:
        assert expected_pipeline in config["pipelines"]
    for expected_component in expected_components:
        assert expected_component in config["components"]


def test_get_config_multi_level_dependencies_of_same_type():
    child = ChildComponent()
    second_intermediate = ParentComponent(dependent=child)
    intermediate = ParentComponent(dependent=second_intermediate)
    parent = ParentComponent(dependent=intermediate)
    p_ensemble = Pipeline()
    p_ensemble.add_node(component=parent, name="ParentComponent", inputs=["Query"])

    expected_components = [
        {"name": "ParentComponent_3", "type": "ParentComponent", "params": {"dependent": "ChildComponent"}},
        {"name": "ParentComponent_2", "type": "ParentComponent", "params": {"dependent": "ParentComponent_3"}},
        {"name": "ParentComponent", "type": "ParentComponent", "params": {"dependent": "ParentComponent_2"}},
        {"name": "ChildComponent", "type": "ChildComponent", "params": {}},
    ]

    expected_pipelines = [{"name": "query", "nodes": [{"name": "ParentComponent", "inputs": ["Query"]}]}]

    config = p_ensemble.get_config()
    for expected_pipeline in expected_pipelines:
        assert expected_pipeline in config["pipelines"]
    for expected_component in expected_components:
        assert expected_component in config["components"]


def test_get_config_component_with_superclass_arguments():
    class CustomBaseDocumentStore(MockDocumentStore):
        def __init__(self, base_parameter: str):
            self.base_parameter = base_parameter

    class CustomDocumentStore(CustomBaseDocumentStore):
        def __init__(self, sub_parameter: int):
            super().__init__(base_parameter="something")
            self.sub_parameter = sub_parameter

    class CustomRetriever(MockRetriever):
        def __init__(self, document_store):
            super().__init__()
            self.document_store = document_store

    document_store = CustomDocumentStore(sub_parameter=10)
    retriever = CustomRetriever(document_store=document_store)
    pipeline = Pipeline()
    pipeline.add_node(retriever, name="Retriever", inputs=["Query"])

    pipeline.get_config()
    assert pipeline.get_document_store().sub_parameter == 10
    assert pipeline.get_document_store().base_parameter == "something"


def test_get_config_custom_node_with_params():
    class CustomNode(MockNode):
        def __init__(self, param: int):
            super().__init__()
            self.param = param

    pipeline = Pipeline()
    pipeline.add_node(CustomNode(param=10), name="custom_node", inputs=["Query"])

    assert len(pipeline.get_config()["components"]) == 1
    assert pipeline.get_config()["components"][0]["params"] == {"param": 10}


def test_get_config_custom_node_with_positional_params():
    class CustomNode(MockNode):
        def __init__(self, param: int = 1):
            super().__init__()
            self.param = param

    pipeline = Pipeline()
    pipeline.add_node(CustomNode(10), name="custom_node", inputs=["Query"])

    assert len(pipeline.get_config()["components"]) == 1
    assert pipeline.get_config()["components"][0]["params"] == {"param": 10}


def test_generate_code_simple_pipeline():
    config = {
        "version": "ignore",
        "components": [
            {
                "name": "retri",
                "type": "BM25Retriever",
                "params": {"document_store": "ElasticsearchDocumentStore", "top_k": 20},
            },
            {
                "name": "ElasticsearchDocumentStore",
                "type": "ElasticsearchDocumentStore",
                "params": {"index": "my-index"},
            },
        ],
        "pipelines": [{"name": "query", "nodes": [{"name": "retri", "inputs": ["Query"]}]}],
    }

    code = generate_code(pipeline_config=config, pipeline_variable_name="p", generate_imports=False)
    assert code == (
        'elasticsearch_document_store = ElasticsearchDocumentStore(index="my-index")\n'
        "retri = BM25Retriever(document_store=elasticsearch_document_store, top_k=20)\n"
        "\n"
        "p = Pipeline()\n"
        'p.add_node(component=retri, name="retri", inputs=["Query"])'
    )


def test_generate_code_imports():
    pipeline_config = {
        "version": "ignore",
        "components": [
            {"name": "DocumentStore", "type": "ElasticsearchDocumentStore"},
            {"name": "retri", "type": "BM25Retriever", "params": {"document_store": "DocumentStore"}},
            {"name": "retri2", "type": "TfidfRetriever", "params": {"document_store": "DocumentStore"}},
        ],
        "pipelines": [
            {
                "name": "Query",
                "nodes": [{"name": "retri", "inputs": ["Query"]}, {"name": "retri2", "inputs": ["Query"]}],
            }
        ],
    }

    code = generate_code(pipeline_config=pipeline_config, pipeline_variable_name="p", generate_imports=True)
    assert code == (
        "from haystack.document_stores import ElasticsearchDocumentStore\n"
        "from haystack.nodes import BM25Retriever, TfidfRetriever\n"
        "from haystack.pipelines import Pipeline\n"
        "\n"
        "document_store = ElasticsearchDocumentStore()\n"
        "retri = BM25Retriever(document_store=document_store)\n"
        "retri_2 = TfidfRetriever(document_store=document_store)\n"
        "\n"
        "p = Pipeline()\n"
        'p.add_node(component=retri, name="retri", inputs=["Query"])\n'
        'p.add_node(component=retri_2, name="retri2", inputs=["Query"])'
    )


def test_generate_code_imports_no_pipeline_cls():
    pipeline_config = {
        "version": "ignore",
        "components": [
            {"name": "DocumentStore", "type": "ElasticsearchDocumentStore"},
            {"name": "retri", "type": "BM25Retriever", "params": {"document_store": "DocumentStore"}},
        ],
        "pipelines": [{"name": "Query", "nodes": [{"name": "retri", "inputs": ["Query"]}]}],
    }

    code = generate_code(
        pipeline_config=pipeline_config,
        pipeline_variable_name="p",
        generate_imports=True,
        add_pipeline_cls_import=False,
    )
    assert code == (
        "from haystack.document_stores import ElasticsearchDocumentStore\n"
        "from haystack.nodes import BM25Retriever\n"
        "\n"
        "document_store = ElasticsearchDocumentStore()\n"
        "retri = BM25Retriever(document_store=document_store)\n"
        "\n"
        "p = Pipeline()\n"
        'p.add_node(component=retri, name="retri", inputs=["Query"])'
    )


def test_generate_code_comment():
    pipeline_config = {
        "version": "ignore",
        "components": [
            {"name": "DocumentStore", "type": "ElasticsearchDocumentStore"},
            {"name": "retri", "type": "BM25Retriever", "params": {"document_store": "DocumentStore"}},
        ],
        "pipelines": [{"name": "Query", "nodes": [{"name": "retri", "inputs": ["Query"]}]}],
    }

    comment = "This is my comment\n...and here is a new line"
    code = generate_code(pipeline_config=pipeline_config, pipeline_variable_name="p", comment=comment)
    assert code == (
        "# This is my comment\n"
        "# ...and here is a new line\n"
        "from haystack.document_stores import ElasticsearchDocumentStore\n"
        "from haystack.nodes import BM25Retriever\n"
        "from haystack.pipelines import Pipeline\n"
        "\n"
        "document_store = ElasticsearchDocumentStore()\n"
        "retri = BM25Retriever(document_store=document_store)\n"
        "\n"
        "p = Pipeline()\n"
        'p.add_node(component=retri, name="retri", inputs=["Query"])'
    )


def test_generate_code_is_component_order_invariant():
    pipeline_config = {
        "version": "ignore",
        "pipelines": [
            {
                "name": "Query",
                "nodes": [
                    {"name": "EsRetriever", "inputs": ["Query"]},
                    {"name": "EmbeddingRetriever", "inputs": ["Query"]},
                    {"name": "JoinResults", "inputs": ["EsRetriever", "EmbeddingRetriever"]},
                ],
            }
        ],
    }

    doc_store = {"name": "ElasticsearchDocumentStore", "type": "ElasticsearchDocumentStore"}
    es_retriever = {
        "name": "EsRetriever",
        "type": "BM25Retriever",
        "params": {"document_store": "ElasticsearchDocumentStore"},
    }
    emb_retriever = {
        "name": "EmbeddingRetriever",
        "type": "EmbeddingRetriever",
        "params": {
            "document_store": "ElasticsearchDocumentStore",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    }
    join_node = {"name": "JoinResults", "type": "JoinDocuments"}

    component_orders = [
        [doc_store, es_retriever, emb_retriever, join_node],
        [es_retriever, emb_retriever, join_node, doc_store],
        [join_node, es_retriever, emb_retriever, doc_store],
    ]

    expected_code = (
        "elasticsearch_document_store = ElasticsearchDocumentStore()\n"
        "es_retriever = BM25Retriever(document_store=elasticsearch_document_store)\n"
        'embedding_retriever = EmbeddingRetriever(document_store=elasticsearch_document_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2")\n'
        "join_results = JoinDocuments()\n"
        "\n"
        "p = Pipeline()\n"
        'p.add_node(component=es_retriever, name="EsRetriever", inputs=["Query"])\n'
        'p.add_node(component=embedding_retriever, name="EmbeddingRetriever", inputs=["Query"])\n'
        'p.add_node(component=join_results, name="JoinResults", inputs=["EsRetriever", "EmbeddingRetriever"])'
    )

    for components in component_orders:
        pipeline_config["components"] = components

        code = generate_code(pipeline_config=pipeline_config, pipeline_variable_name="p", generate_imports=False)
        assert code == expected_code


def test_generate_code_can_handle_weak_cyclic_pipelines():
    config = {
        "version": "ignore",
        "components": [
            {"name": "parent", "type": "ParentComponent", "params": {"dependent": "child"}},
            {"name": "child", "type": "ChildComponent", "params": {}},
        ],
        "pipelines": [
            {
                "name": "query",
                "nodes": [{"name": "parent", "inputs": ["Query"]}, {"name": "child", "inputs": ["parent"]}],
            }
        ],
    }
    code = generate_code(pipeline_config=config, generate_imports=False)
    assert code == (
        "child = ChildComponent()\n"
        "parent = ParentComponent(dependent=child)\n"
        "\n"
        "pipeline = Pipeline()\n"
        'pipeline.add_node(component=parent, name="parent", inputs=["Query"])\n'
        'pipeline.add_node(component=child, name="child", inputs=["parent"])'
    )


@pytest.mark.parametrize("input", ["\btest", " test", "#test", "+test", "\ttest", "\ntest", "test()"])
def test_validate_user_input_invalid(input):
    with pytest.raises(PipelineConfigError, match="is not a valid variable name or value"):
        validate_config_strings(input)


@pytest.mark.parametrize(
    "input",
    [
        "test",
        "testName",
        "test_name",
        "test-name",
        "test-name1234",
        "http://localhost:8000/my-path",
        "C:\\Some\\Windows\\Path\\To\\file.txt",
    ],
)
def test_validate_user_input_valid(input):
    validate_config_strings(input)


def test_validate_pipeline_config_invalid_component_name():
    with pytest.raises(PipelineConfigError, match="is not a valid variable name or value"):
        validate_config_strings({"components": [{"name": "\btest"}]})


def test_validate_pipeline_config_invalid_component_type():
    with pytest.raises(PipelineConfigError, match="is not a valid variable name or value"):
        validate_config_strings({"components": [{"name": "test", "type": "\btest"}]})


def test_validate_pipeline_config_invalid_component_param():
    with pytest.raises(PipelineConfigError, match="is not a valid variable name or value"):
        validate_config_strings({"components": [{"name": "test", "type": "test", "params": {"key": "\btest"}}]})


def test_validate_pipeline_config_invalid_component_param_key():
    with pytest.raises(PipelineConfigError, match="is not a valid variable name or value"):
        validate_config_strings({"components": [{"name": "test", "type": "test", "params": {"\btest": "test"}}]})


def test_validate_pipeline_config_invalid_pipeline_name():
    with pytest.raises(PipelineConfigError, match="is not a valid variable name or value"):
        validate_config_strings({"components": [{"name": "test", "type": "test"}], "pipelines": [{"name": "\btest"}]})


def test_validate_pipeline_config_invalid_pipeline_node_name():
    with pytest.raises(PipelineConfigError, match="is not a valid variable name or value"):
        validate_config_strings(
            {
                "components": [{"name": "test", "type": "test"}],
                "pipelines": [{"name": "test", "type": "test", "nodes": [{"name": "\btest"}]}],
            }
        )


def test_validate_pipeline_config_invalid_pipeline_node_inputs():
    with pytest.raises(PipelineConfigError, match="is not a valid variable name or value"):
        validate_config_strings(
            {
                "components": [{"name": "test", "type": "test"}],
                "pipelines": [{"name": "test", "type": "test", "nodes": [{"name": "test", "inputs": ["\btest"]}]}],
            }
        )


def test_validate_pipeline_config_recursive_config(reduce_windows_recursion_limit):
    pipeline_config = {}
    node = {"config": pipeline_config}
    pipeline_config["node"] = node

    with pytest.raises(PipelineConfigError, match="recursive"):
        validate_config_strings(pipeline_config)


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_load_from_deepset_cloud_query():
    if MOCK_DC:
        with open(SAMPLES_PATH / "dc" / "pipeline_config.json", "r") as f:
            pipeline_config_yaml_response = json.load(f)

        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/{DC_TEST_INDEX}/json",
            json=pipeline_config_yaml_response,
            status=200,
        )

        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/indexes/{DC_TEST_INDEX}/documents-query",
            json=[{"id": "test_doc", "content": "man on hores"}],
            status=200,
        )

    query_pipeline = Pipeline.load_from_deepset_cloud(
        pipeline_config_name=DC_TEST_INDEX, api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
    )
    retriever = query_pipeline.get_node("Retriever")
    document_store = retriever.document_store
    assert isinstance(retriever, BM25Retriever)
    assert isinstance(document_store, DeepsetCloudDocumentStore)
    assert document_store == query_pipeline.get_document_store()

    prediction = query_pipeline.run(query="man on horse", params={})

    assert prediction["query"] == "man on horse"
    assert len(prediction["documents"]) == 1
    assert prediction["documents"][0].id == "test_doc"


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_load_from_deepset_cloud_indexing():
    if MOCK_DC:
        with open(SAMPLES_PATH / "dc" / "pipeline_config.json", "r") as f:
            pipeline_config_yaml_response = json.load(f)

        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/{DC_TEST_INDEX}/json",
            json=pipeline_config_yaml_response,
            status=200,
        )

    indexing_pipeline = Pipeline.load_from_deepset_cloud(
        pipeline_config_name=DC_TEST_INDEX, api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, pipeline_name="indexing"
    )
    document_store = indexing_pipeline.get_node("DocumentStore")
    assert isinstance(document_store, DeepsetCloudDocumentStore)

    with pytest.raises(
        Exception, match=".*NotImplementedError.*DeepsetCloudDocumentStore currently does not support writing documents"
    ):
        indexing_pipeline.run(file_paths=[SAMPLES_PATH / "docs" / "doc_1.txt"])


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_list_pipelines_on_deepset_cloud():
    if MOCK_DC:
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines",
            json={
                "data": [
                    {
                        "name": "test_pipeline_config",
                        "pipeline_id": "2184e0c1-c6ec-40a1-9b28-5d2768e5efa2",
                        "status": "DEPLOYED",
                        "created_at": "2022-02-01T09:57:03.803991+00:00",
                        "deleted": False,
                        "is_default": False,
                        "indexing": {"status": "IN_PROGRESS", "pending_file_count": 4, "total_file_count": 33},
                    }
                ],
                "has_more": False,
                "total": 1,
            },
            status=200,
        )

    pipelines = Pipeline.list_pipelines_on_deepset_cloud(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY)
    assert len(pipelines) == 1
    assert pipelines[0]["name"] == "test_pipeline_config"


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_save_to_deepset_cloud():
    if MOCK_DC:
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_pipeline_config",
            json={
                "name": "test_pipeline_config",
                "pipeline_id": "2184e9c1-c6ec-40a1-9b28-5d2768e5efa2",
                "status": "UNDEPLOYED",
                "created_at": "2022-02-01T09:57:03.803991+00:00",
                "deleted": False,
                "is_default": False,
                "indexing": {"status": "IN_PROGRESS", "pending_file_count": 4, "total_file_count": 33},
            },
            status=200,
        )

        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_pipeline_config_deployed",
            json={
                "name": "test_pipeline_config_deployed",
                "pipeline_id": "8184e0c1-c6ec-40a1-9b28-5d2768e5efa3",
                "status": "DEPLOYED",
                "created_at": "2022-02-09T09:57:03.803991+00:00",
                "deleted": False,
                "is_default": False,
                "indexing": {"status": "INDEXED", "pending_file_count": 0, "total_file_count": 33},
            },
            status=200,
        )

        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_pipeline_config_copy",
            json={"errors": ["Pipeline with the name test_pipeline_config_copy does not exists."]},
            status=404,
        )

        with open(SAMPLES_PATH / "dc" / "pipeline_config.json", "r") as f:
            pipeline_config_yaml_response = json.load(f)

        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/{DC_TEST_INDEX}/json",
            json=pipeline_config_yaml_response,
            status=200,
        )

        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines",
            json={"name": "test_pipeline_config_copy"},
            status=200,
        )

        responses.add(
            method=responses.PUT,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_pipeline_config/yaml",
            json={"name": "test_pipeline_config"},
            status=200,
        )

        responses.add(
            method=responses.PUT,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_pipeline_config_deployed/yaml",
            json={"errors": ["Updating the pipeline yaml is not allowed for pipelines with status: 'DEPLOYED'"]},
            status=406,
        )

    query_pipeline = Pipeline.load_from_deepset_cloud(
        pipeline_config_name=DC_TEST_INDEX, api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
    )

    index_pipeline = Pipeline.load_from_deepset_cloud(
        pipeline_config_name=DC_TEST_INDEX, api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY, pipeline_name="indexing"
    )

    Pipeline.save_to_deepset_cloud(
        query_pipeline=query_pipeline,
        index_pipeline=index_pipeline,
        pipeline_config_name="test_pipeline_config_copy",
        api_endpoint=DC_API_ENDPOINT,
        api_key=DC_API_KEY,
    )

    with pytest.raises(
        ValueError,
        match="Pipeline config 'test_pipeline_config' already exists. Set `overwrite=True` to overwrite pipeline config",
    ):
        Pipeline.save_to_deepset_cloud(
            query_pipeline=query_pipeline,
            index_pipeline=index_pipeline,
            pipeline_config_name="test_pipeline_config",
            api_endpoint=DC_API_ENDPOINT,
            api_key=DC_API_KEY,
        )

    Pipeline.save_to_deepset_cloud(
        query_pipeline=query_pipeline,
        index_pipeline=index_pipeline,
        pipeline_config_name="test_pipeline_config",
        api_endpoint=DC_API_ENDPOINT,
        api_key=DC_API_KEY,
        overwrite=True,
    )

    with pytest.raises(
        ValueError,
        match="Deployed pipeline configs are not allowed to be updated. Please undeploy pipeline config 'test_pipeline_config_deployed' first",
    ):
        Pipeline.save_to_deepset_cloud(
            query_pipeline=query_pipeline,
            index_pipeline=index_pipeline,
            pipeline_config_name="test_pipeline_config_deployed",
            api_endpoint=DC_API_ENDPOINT,
            api_key=DC_API_KEY,
            overwrite=True,
        )


@pytest.mark.integration
@pytest.mark.elasticsearch
@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_save_nonexisting_pipeline_to_deepset_cloud():
    if MOCK_DC:

        def dc_document_store_matcher(request: PreparedRequest) -> Tuple[bool, str]:
            matches = False
            reason = "No DeepsetCloudDocumentStore found."
            request_body = request.body or ""
            json_body = yaml.safe_load(request_body)
            components = json_body["components"]
            for component in components:
                if component["type"].endswith("DocumentStore"):
                    if component["type"] == "DeepsetCloudDocumentStore":
                        matches = True
                    else:
                        matches = False
                        reason = f"Component {component['name']} is of type {component['type']} and not DeepsetCloudDocumentStore"
                        break
            return matches, reason

        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
            json={"errors": ["Pipeline with the name test_pipeline_config_copy does not exists."]},
            status=404,
        )

        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines",
            json={"name": "test_new_non_existing_pipeline"},
            status=201,
            match=[dc_document_store_matcher],
        )

    es_document_store = ElasticsearchDocumentStore()
    es_retriever = BM25Retriever(document_store=es_document_store)
    file_converter = TextConverter()
    preprocessor = PreProcessor()

    query_pipeline = Pipeline()
    query_pipeline.add_node(component=es_retriever, name="Retriever", inputs=["Query"])
    index_pipeline = Pipeline()
    index_pipeline.add_node(component=file_converter, name="FileConverter", inputs=["File"])
    index_pipeline.add_node(component=preprocessor, name="Preprocessor", inputs=["FileConverter"])
    index_pipeline.add_node(component=es_document_store, name="DocumentStore", inputs=["Preprocessor"])

    Pipeline.save_to_deepset_cloud(
        query_pipeline=query_pipeline,
        index_pipeline=index_pipeline,
        pipeline_config_name="test_new_non_existing_pipeline",
        api_endpoint=DC_API_ENDPOINT,
        api_key=DC_API_KEY,
    )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_deploy_on_deepset_cloud_non_existing_pipeline():
    if MOCK_DC:
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
            json={"errors": ["Pipeline with the name test_pipeline_config_copy does not exists."]},
            status=404,
        )

    with pytest.raises(DeepsetCloudError, match="Pipeline config 'test_new_non_existing_pipeline' does not exist."):
        Pipeline.deploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
        )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_undeploy_on_deepset_cloud_non_existing_pipeline():
    if MOCK_DC:
        responses.add(
            method=responses.GET,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
            json={"errors": ["Pipeline with the name test_pipeline_config_copy does not exists."]},
            status=404,
        )

    with pytest.raises(DeepsetCloudError, match="Pipeline config 'test_new_non_existing_pipeline' does not exist."):
        Pipeline.undeploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
        )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_deploy_on_deepset_cloud():
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline/deploy",
            json={"status": "DEPLOYMENT_SCHEDULED"},
            status=200,
        )

        # status will be first undeployed, after deploy() it's in progress twice and the third time deployed
        status_flow = ["UNDEPLOYED", "DEPLOYMENT_IN_PROGRESS", "DEPLOYMENT_IN_PROGRESS", "DEPLOYED"]
        for status in status_flow:
            responses.add(
                method=responses.GET,
                url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
                json={"status": status},
                status=200,
            )

    Pipeline.deploy_on_deepset_cloud(
        pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
    )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_undeploy_on_deepset_cloud():
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline/undeploy",
            json={"status": "UNDEPLOYMENT_SCHEDULED"},
            status=200,
        )

        # status will be first undeployed, after deploy() it's in progress twice and the third time deployed
        status_flow = ["DEPLOYED", "UNDEPLOYMENT_IN_PROGRESS", "UNDEPLOYMENT_IN_PROGRESS", "UNDEPLOYED"]
        for status in status_flow:
            responses.add(
                method=responses.GET,
                url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
                json={"status": status},
                status=200,
            )

    Pipeline.undeploy_on_deepset_cloud(
        pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
    )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_deploy_on_deepset_cloud_sate_already_satisfied():
    if MOCK_DC:
        # status will be first undeployed, after deploy() it's in progress twice and the third time deployed
        status_flow = ["DEPLOYED"]
        for status in status_flow:
            responses.add(
                method=responses.GET,
                url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
                json={"status": status},
                status=200,
            )

    Pipeline.deploy_on_deepset_cloud(
        pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
    )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_undeploy_on_deepset_cloud_sate_already_satisfied():
    if MOCK_DC:
        # status will be first undeployed, after deploy() it's in progress twice and the third time deployed
        status_flow = ["UNDEPLOYED"]
        for status in status_flow:
            responses.add(
                method=responses.GET,
                url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
                json={"status": status},
                status=200,
            )

    Pipeline.undeploy_on_deepset_cloud(
        pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
    )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_deploy_on_deepset_cloud_failed():
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline/deploy",
            json={"status": "DEPLOYMENT_SCHEDULED"},
            status=200,
        )

        # status will be first undeployed, after deploy() it's in progress and the third time undeployed
        status_flow = ["UNDEPLOYED", "DEPLOYMENT_IN_PROGRESS", "UNDEPLOYED"]
        for status in status_flow:
            responses.add(
                method=responses.GET,
                url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
                json={"status": status},
                status=200,
            )

    with pytest.raises(
        DeepsetCloudError, match="Deployment of pipeline config 'test_new_non_existing_pipeline' failed."
    ):
        Pipeline.deploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
        )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_undeploy_on_deepset_cloud_failed():
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline/undeploy",
            json={"status": "UNDEPLOYMENT_SCHEDULED"},
            status=200,
        )

        # status will be first undeployed, after deploy() it's in progress and the third time undeployed
        status_flow = ["DEPLOYED", "UNDEPLOYMENT_IN_PROGRESS", "DEPLOYED"]
        for status in status_flow:
            responses.add(
                method=responses.GET,
                url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
                json={"status": status},
                status=200,
            )

    with pytest.raises(
        DeepsetCloudError, match="Undeployment of pipeline config 'test_new_non_existing_pipeline' failed."
    ):
        Pipeline.undeploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
        )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_deploy_on_deepset_cloud_invalid_initial_state():
    if MOCK_DC:
        status_flow = ["UNDEPLOYMENT_SCHEDULED"]
        for status in status_flow:
            responses.add(
                method=responses.GET,
                url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
                json={"status": status},
                status=200,
            )

    with pytest.raises(
        DeepsetCloudError,
        match="Pipeline config 'test_new_non_existing_pipeline' is in invalid state 'UNDEPLOYMENT_SCHEDULED' to be transitioned to 'DEPLOYED'.",
    ):
        Pipeline.deploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
        )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_undeploy_on_deepset_cloud_invalid_initial_state():
    if MOCK_DC:
        status_flow = ["DEPLOYMENT_SCHEDULED"]
        for status in status_flow:
            responses.add(
                method=responses.GET,
                url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
                json={"status": status},
                status=200,
            )

    with pytest.raises(
        DeepsetCloudError,
        match="Pipeline config 'test_new_non_existing_pipeline' is in invalid state 'DEPLOYMENT_SCHEDULED' to be transitioned to 'UNDEPLOYED'.",
    ):
        Pipeline.undeploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
        )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_deploy_on_deepset_cloud_invalid_state_in_progress():
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline/deploy",
            json={"status": "DEPLOYMENT_SCHEDULED"},
            status=200,
        )

        # status will be first undeployed, after deploy() it's in progress twice and the third time deployed
        status_flow = ["UNDEPLOYED", "UNDEPLOYMENT_IN_PROGRESS"]
        for status in status_flow:
            responses.add(
                method=responses.GET,
                url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
                json={"status": status},
                status=200,
            )
    with pytest.raises(
        DeepsetCloudError,
        match="Deployment of pipline config 'test_new_non_existing_pipeline' aborted. Undeployment was requested.",
    ):
        Pipeline.deploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
        )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_undeploy_on_deepset_cloud_invalid_state_in_progress():
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline/undeploy",
            json={"status": "UNDEPLOYMENT_SCHEDULED"},
            status=200,
        )

        # status will be first undeployed, after deploy() it's in progress twice and the third time deployed
        status_flow = ["DEPLOYED", "DEPLOYMENT_IN_PROGRESS"]
        for status in status_flow:
            responses.add(
                method=responses.GET,
                url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
                json={"status": status},
                status=200,
            )
    with pytest.raises(
        DeepsetCloudError,
        match="Undeployment of pipline config 'test_new_non_existing_pipeline' aborted. Deployment was requested.",
    ):
        Pipeline.undeploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
        )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_deploy_on_deepset_cloud_unknown_state_in_progress():
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline/deploy",
            json={"status": "DEPLOYMENT_SCHEDULED"},
            status=200,
        )

        # status will be first undeployed, after deploy() it's in progress twice and the third time deployed
        status_flow = ["UNDEPLOYED", "ASKDHFASJDF"]
        for status in status_flow:
            responses.add(
                method=responses.GET,
                url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
                json={"status": status},
                status=200,
            )
    with pytest.raises(
        DeepsetCloudError,
        match="Deployment of pipeline config 'test_new_non_existing_pipeline ended in unexpected status: UNKNOWN",
    ):
        Pipeline.deploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
        )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_undeploy_on_deepset_cloud_unknown_state_in_progress():
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline/undeploy",
            json={"status": "UNDEPLOYMENT_SCHEDULED"},
            status=200,
        )

        # status will be first undeployed, after deploy() it's in progress twice and the third time deployed
        status_flow = ["DEPLOYED", "ASKDHFASJDF"]
        for status in status_flow:
            responses.add(
                method=responses.GET,
                url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
                json={"status": status},
                status=200,
            )
    with pytest.raises(
        DeepsetCloudError,
        match="Undeployment of pipeline config 'test_new_non_existing_pipeline ended in unexpected status: UNKNOWN",
    ):
        Pipeline.undeploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
        )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_deploy_on_deepset_cloud_timeout():
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline/deploy",
            json={"status": "DEPLOYMENT_SCHEDULED"},
            status=200,
        )

        # status will be first undeployed, after deploy() it's in progress twice and the third time deployed
        status_flow = ["UNDEPLOYED", "DEPLOYMENT_IN_PROGRESS", "DEPLOYMENT_IN_PROGRESS", "DEPLOYED"]
        for status in status_flow:
            responses.add(
                method=responses.GET,
                url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
                json={"status": status},
                status=200,
            )
    with pytest.raises(
        TimeoutError, match="Transitioning of 'test_new_non_existing_pipeline' to state 'DEPLOYED' timed out."
    ):
        Pipeline.deploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline",
            api_endpoint=DC_API_ENDPOINT,
            api_key=DC_API_KEY,
            timeout=5,
        )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_undeploy_on_deepset_cloud_timeout():
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline/undeploy",
            json={"status": "UNDEPLOYMENT_SCHEDULED"},
            status=200,
        )

        # status will be first undeployed, after deploy() it's in progress twice and the third time deployed
        status_flow = ["DEPLOYED", "UNDEPLOYMENT_IN_PROGRESS", "UNDEPLOYMENT_IN_PROGRESS", "UNDEPLOYED"]
        for status in status_flow:
            responses.add(
                method=responses.GET,
                url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
                json={"status": status},
                status=200,
            )
    with pytest.raises(
        TimeoutError, match="Transitioning of 'test_new_non_existing_pipeline' to state 'UNDEPLOYED' timed out."
    ):
        Pipeline.undeploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline",
            api_endpoint=DC_API_ENDPOINT,
            api_key=DC_API_KEY,
            timeout=5,
        )


def test_graph_validation_invalid_edge():
    docstore = MockDocumentStore()
    retriever = DummyRetriever(document_store=docstore)
    pipeline = Pipeline()
    pipeline.add_node(name="DocStore", component=docstore, inputs=["Query"])

    with pytest.raises(PipelineConfigError, match="DocStore has only 1 outgoing edge"):
        pipeline.add_node(name="Retriever", component=retriever, inputs=["DocStore.output_2"])


def test_graph_validation_non_existing_edge():
    docstore = MockDocumentStore()
    retriever = DummyRetriever(document_store=docstore)
    pipeline = Pipeline()
    pipeline.add_node(name="DocStore", component=docstore, inputs=["Query"])

    with pytest.raises(PipelineConfigError, match="'wrong_edge_label' is not a valid edge name"):
        pipeline.add_node(name="Retriever", component=retriever, inputs=["DocStore.wrong_edge_label"])


def test_graph_validation_invalid_node():
    docstore = MockDocumentStore()
    retriever = DummyRetriever(document_store=docstore)
    pipeline = Pipeline()
    pipeline.add_node(name="DocStore", component=docstore, inputs=["Query"])

    with pytest.raises(PipelineConfigError, match="Cannot find node 'InvalidNode'"):
        pipeline.add_node(name="Retriever", component=retriever, inputs=["InvalidNode"])


def test_graph_validation_invalid_root_node():
    docstore = MockDocumentStore()
    pipeline = Pipeline()

    with pytest.raises(PipelineConfigError, match="one single root node"):
        pipeline.add_node(name="DocStore", component=docstore, inputs=["InvalidNode"])


def test_graph_validation_no_root_node():
    docstore = MockNode()
    pipeline = Pipeline()

    with pytest.raises(PipelineConfigError, match="one single root node"):
        pipeline.add_node(name="Node", component=docstore, inputs=[])


def test_graph_validation_two_root_nodes():
    docstore = MockNode()
    pipeline = Pipeline()

    with pytest.raises(PipelineConfigError, match="one single root node"):
        pipeline.add_node(name="Node", component=docstore, inputs=["Query", "File"])

    with pytest.raises(PipelineConfigError, match="one single root node"):
        pipeline.add_node(name="Node", component=docstore, inputs=["Query", "Query"])


def test_graph_validation_duplicate_node_instance():
    node = MockNode()
    pipeline = Pipeline()
    pipeline.add_node(name="node_a", component=node, inputs=["Query"])

    with pytest.raises(PipelineConfigError, match="You have already added the same instance to the pipeline"):
        pipeline.add_node(name="node_b", component=node, inputs=["node_a"])


def test_graph_validation_duplicate_node():
    node = MockNode()
    other_node = MockNode()
    pipeline = Pipeline()
    pipeline.add_node(name="node", component=node, inputs=["Query"])
    with pytest.raises(PipelineConfigError, match="'node' is already in the pipeline"):
        pipeline.add_node(name="node", component=other_node, inputs=["Query"])


def test_parallel_paths_in_pipeline_graph():
    class A(RootNode):
        def run(self):
            test = "A"
            return {"test": test}, "output_1"

    class B(RootNode):
        def run(self, test):
            test += "B"
            return {"test": test}, "output_1"

    class C(RootNode):
        def run(self, test):
            test += "C"
            return {"test": test}, "output_1"

    class D(RootNode):
        def run(self, test):
            test += "D"
            return {"test": test}, "output_1"

    class E(RootNode):
        def run(self, test):
            test += "E"
            return {"test": test}, "output_1"

    class JoinNode(RootNode):
        def run(self, inputs):
            test = inputs[0]["test"] + inputs[1]["test"]
            return {"test": test}, "output_1"

    pipeline = Pipeline()
    pipeline.add_node(name="A", component=A(), inputs=["Query"])
    pipeline.add_node(name="B", component=B(), inputs=["A"])
    pipeline.add_node(name="C", component=C(), inputs=["B"])
    pipeline.add_node(name="E", component=E(), inputs=["C"])
    pipeline.add_node(name="D", component=D(), inputs=["B"])
    pipeline.add_node(name="F", component=JoinNode(), inputs=["D", "E"])
    output = pipeline.run(query="test")
    assert output["test"] == "ABDABCE"

    pipeline = Pipeline()
    pipeline.add_node(name="A", component=A(), inputs=["Query"])
    pipeline.add_node(name="B", component=B(), inputs=["A"])
    pipeline.add_node(name="C", component=C(), inputs=["B"])
    pipeline.add_node(name="D", component=D(), inputs=["B"])
    pipeline.add_node(name="E", component=JoinNode(), inputs=["C", "D"])
    output = pipeline.run(query="test")
    assert output["test"] == "ABCABD"


def test_parallel_paths_in_pipeline_graph_with_branching():
    class AWithOutput1(RootNode):
        outgoing_edges = 2

        def run(self):
            output = "A"
            return {"output": output}, "output_1"

    class AWithOutput2(RootNode):
        outgoing_edges = 2

        def run(self):
            output = "A"
            return {"output": output}, "output_2"

    class AWithOutputAll(RootNode):
        outgoing_edges = 2

        def run(self):
            output = "A"
            return {"output": output}, "output_all"

    class B(RootNode):
        def run(self, output):
            output += "B"
            return {"output": output}, "output_1"

    class C(RootNode):
        def run(self, output):
            output += "C"
            return {"output": output}, "output_1"

    class D(RootNode):
        def run(self, output):
            output += "D"
            return {"output": output}, "output_1"

    class E(RootNode):
        def run(self, output):
            output += "E"
            return {"output": output}, "output_1"

    class JoinNode(RootNode):
        def run(self, output=None, inputs=None):
            if inputs:
                output = ""
                for input_dict in inputs:
                    output += input_dict["output"]
            return {"output": output}, "output_1"

    pipeline = Pipeline()
    pipeline.add_node(name="A", component=AWithOutput1(), inputs=["Query"])
    pipeline.add_node(name="B", component=B(), inputs=["A.output_1"])
    pipeline.add_node(name="C", component=C(), inputs=["A.output_2"])
    pipeline.add_node(name="D", component=E(), inputs=["B"])
    pipeline.add_node(name="E", component=D(), inputs=["B"])
    pipeline.add_node(name="F", component=JoinNode(), inputs=["D", "E", "C"])
    output = pipeline.run(query="test")
    assert output["output"] == "ABEABD"

    pipeline = Pipeline()
    pipeline.add_node(name="A", component=AWithOutput2(), inputs=["Query"])
    pipeline.add_node(name="B", component=B(), inputs=["A.output_1"])
    pipeline.add_node(name="C", component=C(), inputs=["A.output_2"])
    pipeline.add_node(name="D", component=E(), inputs=["B"])
    pipeline.add_node(name="E", component=D(), inputs=["B"])
    pipeline.add_node(name="F", component=JoinNode(), inputs=["D", "E", "C"])
    output = pipeline.run(query="test")
    assert output["output"] == "AC"

    pipeline = Pipeline()
    pipeline.add_node(name="A", component=AWithOutputAll(), inputs=["Query"])
    pipeline.add_node(name="B", component=B(), inputs=["A.output_1"])
    pipeline.add_node(name="C", component=C(), inputs=["A.output_2"])
    pipeline.add_node(name="D", component=E(), inputs=["B"])
    pipeline.add_node(name="E", component=D(), inputs=["B"])
    pipeline.add_node(name="F", component=JoinNode(), inputs=["D", "E", "C"])
    output = pipeline.run(query="test")
    assert output["output"] == "ACABEABD"


def test_pipeline_components():
    class Node(BaseComponent):
        outgoing_edges = 1

        def run(self):
            test = "test"
            return {"test": test}, "output_1"

    a = Node()
    b = Node()
    c = Node()
    d = Node()
    e = Node()
    pipeline = Pipeline()
    pipeline.add_node(name="A", component=a, inputs=["Query"])
    pipeline.add_node(name="B", component=b, inputs=["A"])
    pipeline.add_node(name="C", component=c, inputs=["B"])
    pipeline.add_node(name="D", component=d, inputs=["C"])
    pipeline.add_node(name="E", component=e, inputs=["D"])
    assert len(pipeline.components) == 5
    assert pipeline.components["A"] == a
    assert pipeline.components["B"] == b
    assert pipeline.components["C"] == c
    assert pipeline.components["D"] == d
    assert pipeline.components["E"] == e


def test_pipeline_get_document_store_from_components():
    doc_store = MockDocumentStore()
    pipeline = Pipeline()
    pipeline.add_node(name="A", component=doc_store, inputs=["File"])

    assert doc_store == pipeline.get_document_store()


def test_pipeline_get_document_store_from_components_multiple_doc_stores():
    doc_store_a = MockDocumentStore()
    doc_store_b = MockDocumentStore()
    pipeline = Pipeline()
    pipeline.add_node(name="A", component=doc_store_a, inputs=["File"])
    pipeline.add_node(name="B", component=doc_store_b, inputs=["File"])

    with pytest.raises(Exception, match="Multiple Document Stores found in Pipeline"):
        pipeline.get_document_store()


def test_pipeline_get_document_store_from_retriever():
    doc_store = MockDocumentStore()
    retriever = DummyRetriever(document_store=doc_store)
    pipeline = Pipeline()
    pipeline.add_node(name="A", component=retriever, inputs=["Query"])

    assert doc_store == pipeline.get_document_store()


def test_pipeline_get_document_store_from_dual_retriever():
    doc_store = MockDocumentStore()
    retriever_a = DummyRetriever(document_store=doc_store)
    retriever_b = DummyRetriever(document_store=doc_store)
    join_node = JoinNode()
    pipeline = Pipeline()
    pipeline.add_node(name="A", component=retriever_a, inputs=["Query"])
    pipeline.add_node(name="B", component=retriever_b, inputs=["Query"])
    pipeline.add_node(name="C", component=join_node, inputs=["A", "B"])

    assert doc_store == pipeline.get_document_store()


def test_pipeline_get_document_store_multiple_doc_stores_from_dual_retriever():
    doc_store_a = MockDocumentStore()
    doc_store_b = MockDocumentStore()
    retriever_a = DummyRetriever(document_store=doc_store_a)
    retriever_b = DummyRetriever(document_store=doc_store_b)
    join_node = JoinNode()
    pipeline = Pipeline()
    pipeline.add_node(name="A", component=retriever_a, inputs=["Query"])
    pipeline.add_node(name="B", component=retriever_b, inputs=["Query"])
    pipeline.add_node(name="C", component=join_node, inputs=["A", "B"])

    with pytest.raises(Exception, match="Multiple Document Stores found in Pipeline"):
        pipeline.get_document_store()


#
# RouteDocuments tests
#


def test_routedocuments_by_content_type():
    docs = [
        Document(content="text document", content_type="text"),
        Document(
            content=pd.DataFrame(columns=["col 1", "col 2"], data=[["row 1", "row 1"], ["row 2", "row 2"]]),
            content_type="table",
        ),
    ]
    route_documents = RouteDocuments()
    result, _ = route_documents.run(documents=docs)
    assert len(result["output_1"]) == 1
    assert len(result["output_2"]) == 1
    assert result["output_1"][0].content_type == "text"
    assert result["output_2"][0].content_type == "table"


def test_routedocuments_by_metafield(test_docs_xs):
    docs = [Document.from_dict(doc) if isinstance(doc, dict) else doc for doc in test_docs_xs]
    route_documents = RouteDocuments(split_by="meta_field", metadata_values=["test1", "test3", "test5"])
    result, _ = route_documents.run(docs)
    assert len(result["output_1"]) == 1
    assert len(result["output_2"]) == 1
    assert len(result["output_3"]) == 1
    assert result["output_1"][0].meta["meta_field"] == "test1"
    assert result["output_2"][0].meta["meta_field"] == "test3"
    assert result["output_3"][0].meta["meta_field"] == "test5"


#
# JoinAnswers tests
#


@pytest.mark.parametrize("join_mode", ["concatenate", "merge"])
def test_joinanswers(join_mode):
    inputs = [{"answers": [Answer(answer="answer 1", score=0.7)]}, {"answers": [Answer(answer="answer 2", score=0.8)]}]

    join_answers = JoinAnswers(join_mode=join_mode)
    result, _ = join_answers.run(inputs)
    assert len(result["answers"]) == 2
    assert result["answers"] == sorted(result["answers"], reverse=True)

    result, _ = join_answers.run(inputs, top_k_join=1)
    assert len(result["answers"]) == 1
    assert result["answers"][0].answer == "answer 2"
