from pathlib import Path

import os
import json
from unittest.mock import Mock

import pandas as pd
import pytest
import responses

from haystack import __version__, Document, Answer, JoinAnswers
from haystack.document_stores.base import BaseDocumentStore
from haystack.document_stores.deepsetcloud import DeepsetCloudDocumentStore
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.nodes.other.join_docs import JoinDocuments
from haystack.nodes.base import BaseComponent
from haystack.nodes.retriever.base import BaseRetriever
from haystack.nodes.retriever.sparse import ElasticsearchRetriever
from haystack.pipelines import Pipeline, DocumentSearchPipeline, RootNode, ExtractiveQAPipeline
from haystack.pipelines.config import _validate_user_input, validate_config
from haystack.pipelines.utils import generate_code
from haystack.nodes import DensePassageRetriever, EmbeddingRetriever, RouteDocuments

from conftest import MOCK_DC, DC_API_ENDPOINT, DC_API_KEY, DC_TEST_INDEX, SAMPLES_PATH, deepset_cloud_fixture


class ParentComponent(BaseComponent):
    outgoing_edges = 1

    def __init__(self, dependent: BaseComponent) -> None:
        super().__init__()
        self.set_config(dependent=dependent)


class ParentComponent2(BaseComponent):
    outgoing_edges = 1

    def __init__(self, dependent: BaseComponent) -> None:
        super().__init__()
        self.set_config(dependent=dependent)


class ChildComponent(BaseComponent):
    def __init__(self, some_key: str = None) -> None:
        super().__init__()
        self.set_config(some_key=some_key)


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
def test_load_and_save_yaml(document_store, tmp_path):
    # test correct load of indexing pipeline from yaml
    pipeline = Pipeline.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "test_pipeline.yaml", pipeline_name="indexing_pipeline"
    )
    pipeline.run(file_paths=SAMPLES_PATH / "pdf" / "sample_pdf_1.pdf")
    # test correct load of query pipeline from yaml
    pipeline = Pipeline.load_from_yaml(SAMPLES_PATH / "pipeline" / "test_pipeline.yaml", pipeline_name="query_pipeline")
    prediction = pipeline.run(
        query="Who made the PDF specification?", params={"ESRetriever": {"top_k": 10}, "Reader": {"top_k": 3}}
    )
    assert prediction["query"] == "Who made the PDF specification?"
    assert prediction["answers"][0].answer == "Adobe Systems"
    assert "_debug" not in prediction.keys()

    # test invalid pipeline name
    with pytest.raises(Exception):
        Pipeline.load_from_yaml(path=SAMPLES_PATH / "pipeline" / "test_pipeline.yaml", pipeline_name="invalid")
    # test config export
    pipeline.save_to_yaml(tmp_path / "test.yaml")
    with open(tmp_path / "test.yaml", "r", encoding="utf-8") as stream:
        saved_yaml = stream.read()
    expected_yaml = f"""
        components:
        - name: ESRetriever
          params:
            document_store: ElasticsearchDocumentStore
          type: ElasticsearchRetriever
        - name: ElasticsearchDocumentStore
          params:
            index: haystack_test
            label_index: haystack_test_label
          type: ElasticsearchDocumentStore
        - name: Reader
          params:
            model_name_or_path: deepset/roberta-base-squad2
            no_ans_boost: -10
            num_processes: 0
          type: FARMReader
        pipelines:
        - name: query
          nodes:
          - inputs:
            - Query
            name: ESRetriever
          - inputs:
            - ESRetriever
            name: Reader
          type: Pipeline
        version: {__version__}
    """
    assert saved_yaml.replace(" ", "").replace("\n", "") == expected_yaml.replace(" ", "").replace("\n", "")


@pytest.mark.elasticsearch
@pytest.mark.parametrize("document_store", ["elasticsearch"], indirect=True)
def test_load_and_save_yaml_prebuilt_pipelines(document_store, tmp_path):
    # populating index
    pipeline = Pipeline.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "test_pipeline.yaml", pipeline_name="indexing_pipeline"
    )
    pipeline.run(file_paths=SAMPLES_PATH / "pdf" / "sample_pdf_1.pdf")
    # test correct load of query pipeline from yaml
    pipeline = ExtractiveQAPipeline.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "test_pipeline.yaml", pipeline_name="query_pipeline"
    )
    prediction = pipeline.run(
        query="Who made the PDF specification?", params={"ESRetriever": {"top_k": 10}, "Reader": {"top_k": 3}}
    )
    assert prediction["query"] == "Who made the PDF specification?"
    assert prediction["answers"][0].answer == "Adobe Systems"
    assert "_debug" not in prediction.keys()

    # test invalid pipeline name
    with pytest.raises(Exception):
        ExtractiveQAPipeline.load_from_yaml(
            path=SAMPLES_PATH / "pipeline" / "test_pipeline.yaml", pipeline_name="invalid"
        )
    # test config export
    pipeline.save_to_yaml(tmp_path / "test.yaml")
    with open(tmp_path / "test.yaml", "r", encoding="utf-8") as stream:
        saved_yaml = stream.read()
    expected_yaml = f"""
        components:
        - name: ESRetriever
          params:
            document_store: ElasticsearchDocumentStore
          type: ElasticsearchRetriever
        - name: ElasticsearchDocumentStore
          params:
            index: haystack_test
            label_index: haystack_test_label
          type: ElasticsearchDocumentStore
        - name: Reader
          params:
            model_name_or_path: deepset/roberta-base-squad2
            no_ans_boost: -10
            num_processes: 0
          type: FARMReader
        pipelines:
        - name: query
          nodes:
          - inputs:
            - Query
            name: ESRetriever
          - inputs:
            - ESRetriever
            name: Reader
          type: Pipeline
        version: {__version__}
    """
    assert saved_yaml.replace(" ", "").replace("\n", "") == expected_yaml.replace(" ", "").replace("\n", "")


def test_load_tfidfretriever_yaml(tmp_path):
    documents = [
        {
            "content": "A Doc specifically talking about haystack. Haystack can be used to scale QA models to large document collections."
        }
    ]
    pipeline = Pipeline.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "test_pipeline_tfidfretriever.yaml", pipeline_name="query_pipeline"
    )
    with pytest.raises(Exception) as exc_info:
        pipeline.run(
            query="What can be used to scale QA models to large document collections?",
            params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}},
        )
    exception_raised = str(exc_info.value)
    assert "Retrieval requires dataframe df and tf-idf matrix" in exception_raised

    pipeline.get_node(name="Retriever").document_store.write_documents(documents=documents)
    prediction = pipeline.run(
        query="What can be used to scale QA models to large document collections?",
        params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}},
    )
    assert prediction["query"] == "What can be used to scale QA models to large document collections?"
    assert prediction["answers"][0].answer == "haystack"


@pytest.mark.elasticsearch
def test_to_code_creates_same_pipelines():
    index_pipeline = Pipeline.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "test_pipeline.yaml", pipeline_name="indexing_pipeline"
    )
    query_pipeline = Pipeline.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "test_pipeline.yaml", pipeline_name="query_pipeline"
    )
    query_pipeline_code = query_pipeline.to_code(pipeline_variable_name="query_pipeline_from_code")
    index_pipeline_code = index_pipeline.to_code(pipeline_variable_name="index_pipeline_from_code")
    exec(query_pipeline_code)
    exec(index_pipeline_code)
    assert locals()["query_pipeline_from_code"] is not None
    assert locals()["index_pipeline_from_code"] is not None
    assert query_pipeline.get_config() == locals()["query_pipeline_from_code"].get_config()
    assert index_pipeline.get_config() == locals()["index_pipeline_from_code"].get_config()


def test_get_config_creates_dependent_component():
    child = ChildComponent()
    parent = ParentComponent(dependent=child)
    pipeline = Pipeline()
    pipeline.add_node(component=parent, name="parent", inputs=["Query"])

    expected_pipelines = [{"name": "query", "type": "Pipeline", "nodes": [{"name": "parent", "inputs": ["Query"]}]}]

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
            "type": "Pipeline",
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
            "type": "Pipeline",
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


def test_generate_code_simple_pipeline():
    config = {
        "components": [
            {
                "name": "retri",
                "type": "ElasticsearchRetriever",
                "params": {"document_store": "ElasticsearchDocumentStore", "top_k": 20},
            },
            {
                "name": "ElasticsearchDocumentStore",
                "type": "ElasticsearchDocumentStore",
                "params": {"index": "my-index"},
            },
        ],
        "pipelines": [{"name": "query", "type": "Pipeline", "nodes": [{"name": "retri", "inputs": ["Query"]}]}],
    }

    code = generate_code(pipeline_config=config, pipeline_variable_name="p", generate_imports=False)
    assert code == (
        'elasticsearch_document_store = ElasticsearchDocumentStore(index="my-index")\n'
        "retri = ElasticsearchRetriever(document_store=elasticsearch_document_store, top_k=20)\n"
        "\n"
        "p = Pipeline()\n"
        'p.add_node(component=retri, name="retri", inputs=["Query"])'
    )


def test_generate_code_imports():
    pipeline_config = {
        "components": [
            {
                "name": "DocumentStore",
                "type": "ElasticsearchDocumentStore",
            },
            {"name": "retri", "type": "ElasticsearchRetriever", "params": {"document_store": "DocumentStore"}},
            {"name": "retri2", "type": "EmbeddingRetriever", "params": {"document_store": "DocumentStore"}},
        ],
        "pipelines": [
            {
                "name": "Query",
                "type": "Pipeline",
                "nodes": [{"name": "retri", "inputs": ["Query"]}, {"name": "retri2", "inputs": ["Query"]}],
            }
        ],
    }

    code = generate_code(pipeline_config=pipeline_config, pipeline_variable_name="p", generate_imports=True)
    assert code == (
        "from haystack.document_stores import ElasticsearchDocumentStore\n"
        "from haystack.nodes import ElasticsearchRetriever, EmbeddingRetriever\n"
        "from haystack.pipelines import Pipeline\n"
        "\n"
        "document_store = ElasticsearchDocumentStore()\n"
        "retri = ElasticsearchRetriever(document_store=document_store)\n"
        "retri_2 = EmbeddingRetriever(document_store=document_store)\n"
        "\n"
        "p = Pipeline()\n"
        'p.add_node(component=retri, name="retri", inputs=["Query"])\n'
        'p.add_node(component=retri_2, name="retri2", inputs=["Query"])'
    )


def test_generate_code_imports_no_pipeline_cls():
    pipeline_config = {
        "components": [
            {
                "name": "DocumentStore",
                "type": "ElasticsearchDocumentStore",
            },
            {"name": "retri", "type": "ElasticsearchRetriever", "params": {"document_store": "DocumentStore"}},
        ],
        "pipelines": [{"name": "Query", "type": "Pipeline", "nodes": [{"name": "retri", "inputs": ["Query"]}]}],
    }

    code = generate_code(
        pipeline_config=pipeline_config,
        pipeline_variable_name="p",
        generate_imports=True,
        add_pipeline_cls_import=False,
    )
    assert code == (
        "from haystack.document_stores import ElasticsearchDocumentStore\n"
        "from haystack.nodes import ElasticsearchRetriever\n"
        "\n"
        "document_store = ElasticsearchDocumentStore()\n"
        "retri = ElasticsearchRetriever(document_store=document_store)\n"
        "\n"
        "p = Pipeline()\n"
        'p.add_node(component=retri, name="retri", inputs=["Query"])'
    )


def test_generate_code_comment():
    pipeline_config = {
        "components": [
            {
                "name": "DocumentStore",
                "type": "ElasticsearchDocumentStore",
            },
            {"name": "retri", "type": "ElasticsearchRetriever", "params": {"document_store": "DocumentStore"}},
        ],
        "pipelines": [{"name": "Query", "type": "Pipeline", "nodes": [{"name": "retri", "inputs": ["Query"]}]}],
    }

    comment = "This is my comment\n...and here is a new line"
    code = generate_code(pipeline_config=pipeline_config, pipeline_variable_name="p", comment=comment)
    assert code == (
        "# This is my comment\n"
        "# ...and here is a new line\n"
        "from haystack.document_stores import ElasticsearchDocumentStore\n"
        "from haystack.nodes import ElasticsearchRetriever\n"
        "from haystack.pipelines import Pipeline\n"
        "\n"
        "document_store = ElasticsearchDocumentStore()\n"
        "retri = ElasticsearchRetriever(document_store=document_store)\n"
        "\n"
        "p = Pipeline()\n"
        'p.add_node(component=retri, name="retri", inputs=["Query"])'
    )


def test_generate_code_is_component_order_invariant():
    pipeline_config = {
        "pipelines": [
            {
                "name": "Query",
                "type": "Pipeline",
                "nodes": [
                    {"name": "EsRetriever", "inputs": ["Query"]},
                    {"name": "EmbeddingRetriever", "inputs": ["Query"]},
                    {"name": "JoinResults", "inputs": ["EsRetriever", "EmbeddingRetriever"]},
                ],
            }
        ]
    }

    doc_store = {
        "name": "ElasticsearchDocumentStore",
        "type": "ElasticsearchDocumentStore",
    }
    es_retriever = {
        "name": "EsRetriever",
        "type": "ElasticsearchRetriever",
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
    join_node = {
        "name": "JoinResults",
        "type": "JoinDocuments",
    }

    component_orders = [
        [doc_store, es_retriever, emb_retriever, join_node],
        [es_retriever, emb_retriever, join_node, doc_store],
        [join_node, es_retriever, emb_retriever, doc_store],
    ]

    expected_code = (
        "elasticsearch_document_store = ElasticsearchDocumentStore()\n"
        "es_retriever = ElasticsearchRetriever(document_store=elasticsearch_document_store)\n"
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


@pytest.mark.parametrize("input", ["\btest", " test", "#test", "+test", "\ttest", "\ntest", "test()"])
def test_validate_user_input_invalid(input):
    with pytest.raises(ValueError, match="is not a valid config variable name"):
        _validate_user_input(input)


@pytest.mark.parametrize(
    "input", ["test", "testName", "test_name", "test-name", "test-name1234", "http://localhost:8000/my-path"]
)
def test_validate_user_input_valid(input):
    _validate_user_input(input)


def test_validate_pipeline_config_invalid_component_name():
    with pytest.raises(ValueError, match="is not a valid config variable name"):
        validate_config({"components": [{"name": "\btest"}]})


def test_validate_pipeline_config_invalid_component_type():
    with pytest.raises(ValueError, match="is not a valid config variable name"):
        validate_config({"components": [{"name": "test", "type": "\btest"}]})


def test_validate_pipeline_config_invalid_component_param():
    with pytest.raises(ValueError, match="is not a valid config variable name"):
        validate_config({"components": [{"name": "test", "type": "test", "params": {"key": "\btest"}}]})


def test_validate_pipeline_config_invalid_component_param_key():
    with pytest.raises(ValueError, match="is not a valid config variable name"):
        validate_config({"components": [{"name": "test", "type": "test", "params": {"\btest": "test"}}]})


def test_validate_pipeline_config_invalid_pipeline_name():
    with pytest.raises(ValueError, match="is not a valid config variable name"):
        validate_config(
            {
                "components": [
                    {
                        "name": "test",
                        "type": "test",
                    }
                ],
                "pipelines": [{"name": "\btest"}],
            }
        )


def test_validate_pipeline_config_invalid_pipeline_type():
    with pytest.raises(ValueError, match="is not a valid config variable name"):
        validate_config(
            {
                "components": [
                    {
                        "name": "test",
                        "type": "test",
                    }
                ],
                "pipelines": [{"name": "test", "type": "\btest"}],
            }
        )


def test_validate_pipeline_config_invalid_pipeline_node_name():
    with pytest.raises(ValueError, match="is not a valid config variable name"):
        validate_config(
            {
                "components": [
                    {
                        "name": "test",
                        "type": "test",
                    }
                ],
                "pipelines": [{"name": "test", "type": "test", "nodes": [{"name": "\btest"}]}],
            }
        )


def test_validate_pipeline_config_invalid_pipeline_node_inputs():
    with pytest.raises(ValueError, match="is not a valid config variable name"):
        validate_config(
            {
                "components": [
                    {
                        "name": "test",
                        "type": "test",
                    }
                ],
                "pipelines": [{"name": "test", "type": "test", "nodes": [{"name": "test", "inputs": ["\btest"]}]}],
            }
        )


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
    assert isinstance(retriever, ElasticsearchRetriever)
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


# @pytest.mark.slow
# @pytest.mark.elasticsearch
# @pytest.mark.parametrize(
#     "retriever_with_docs, document_store_with_docs",
#     [("elasticsearch", "elasticsearch")],
#     indirect=True,
# )
@pytest.mark.parametrize(
    "retriever_with_docs,document_store_with_docs",
    [
        ("dpr", "elasticsearch"),
        ("dpr", "faiss"),
        ("dpr", "memory"),
        ("dpr", "milvus1"),
        ("embedding", "elasticsearch"),
        ("embedding", "faiss"),
        ("embedding", "memory"),
        ("embedding", "milvus1"),
        ("elasticsearch", "elasticsearch"),
        ("es_filter_only", "elasticsearch"),
        ("tfidf", "memory"),
    ],
    indirect=True,
)
def test_graph_creation(retriever_with_docs, document_store_with_docs):
    pipeline = Pipeline()
    pipeline.add_node(name="ES", component=retriever_with_docs, inputs=["Query"])

    with pytest.raises(AssertionError):
        pipeline.add_node(name="Reader", component=retriever_with_docs, inputs=["ES.output_2"])

    with pytest.raises(AssertionError):
        pipeline.add_node(name="Reader", component=retriever_with_docs, inputs=["ES.wrong_edge_label"])

    with pytest.raises(Exception):
        pipeline.add_node(name="Reader", component=retriever_with_docs, inputs=["InvalidNode"])

    with pytest.raises(Exception):
        pipeline = Pipeline()
        pipeline.add_node(name="ES", component=retriever_with_docs, inputs=["InvalidNode"])


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
    class DummyDocumentStore(BaseDocumentStore):
        pass

    doc_store = DummyDocumentStore()
    pipeline = Pipeline()
    pipeline.add_node(name="A", component=doc_store, inputs=["File"])

    assert doc_store == pipeline.get_document_store()


def test_pipeline_get_document_store_from_components_multiple_doc_stores():
    class DummyDocumentStore(BaseDocumentStore):
        pass

    doc_store_a = DummyDocumentStore()
    doc_store_b = DummyDocumentStore()
    pipeline = Pipeline()
    pipeline.add_node(name="A", component=doc_store_a, inputs=["File"])
    pipeline.add_node(name="B", component=doc_store_b, inputs=["File"])

    with pytest.raises(Exception, match="Multiple Document Stores found in Pipeline"):
        pipeline.get_document_store()


def test_pipeline_get_document_store_from_retriever():
    class DummyRetriever(BaseRetriever):
        def __init__(self, document_store):
            self.document_store = document_store

        def run(self):
            test = "test"
            return {"test": test}, "output_1"

    class DummyDocumentStore(BaseDocumentStore):
        pass

    doc_store = DummyDocumentStore()
    retriever = DummyRetriever(document_store=doc_store)
    pipeline = Pipeline()
    pipeline.add_node(name="A", component=retriever, inputs=["Query"])

    assert doc_store == pipeline.get_document_store()


def test_pipeline_get_document_store_from_dual_retriever():
    class DummyRetriever(BaseRetriever):
        def __init__(self, document_store):
            self.document_store = document_store

        def run(self):
            test = "test"
            return {"test": test}, "output_1"

    class DummyDocumentStore(BaseDocumentStore):
        pass

    class JoinNode(RootNode):
        def run(self, output=None, inputs=None):
            if inputs:
                output = ""
                for input_dict in inputs:
                    output += input_dict["output"]
            return {"output": output}, "output_1"

    doc_store = DummyDocumentStore()
    retriever_a = DummyRetriever(document_store=doc_store)
    retriever_b = DummyRetriever(document_store=doc_store)
    pipeline = Pipeline()
    pipeline.add_node(name="A", component=retriever_a, inputs=["Query"])
    pipeline.add_node(name="B", component=retriever_b, inputs=["Query"])
    pipeline.add_node(name="C", component=JoinNode(), inputs=["A", "B"])

    assert doc_store == pipeline.get_document_store()


def test_pipeline_get_document_store_multiple_doc_stores_from_dual_retriever():
    class DummyRetriever(BaseRetriever):
        def __init__(self, document_store):
            self.document_store = document_store

        def run(self):
            test = "test"
            return {"test": test}, "output_1"

    class DummyDocumentStore(BaseDocumentStore):
        pass

    class JoinNode(RootNode):
        def run(self, output=None, inputs=None):
            if inputs:
                output = ""
                for input_dict in inputs:
                    output += input_dict["output"]
            return {"output": output}, "output_1"

    doc_store_a = DummyDocumentStore()
    doc_store_b = DummyDocumentStore()
    retriever_a = DummyRetriever(document_store=doc_store_a)
    retriever_b = DummyRetriever(document_store=doc_store_b)
    pipeline = Pipeline()
    pipeline.add_node(name="A", component=retriever_a, inputs=["Query"])
    pipeline.add_node(name="B", component=retriever_b, inputs=["Query"])
    pipeline.add_node(name="C", component=JoinNode(), inputs=["A", "B"])

    with pytest.raises(Exception, match="Multiple Document Stores found in Pipeline"):
        pipeline.get_document_store()


def test_existing_faiss_document_store():
    clean_faiss_document_store()

    pipeline = Pipeline.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "test_pipeline_faiss_indexing.yaml", pipeline_name="indexing_pipeline"
    )
    pipeline.run(file_paths=SAMPLES_PATH / "pdf" / "sample_pdf_1.pdf")

    new_document_store = pipeline.get_document_store()
    new_document_store.save("existing_faiss_document_store")

    # test correct load of query pipeline from yaml
    pipeline = Pipeline.load_from_yaml(
        SAMPLES_PATH / "pipeline" / "test_pipeline_faiss_retrieval.yaml", pipeline_name="query_pipeline"
    )

    retriever = pipeline.get_node("DPRRetriever")
    existing_document_store = retriever.document_store
    faiss_index = existing_document_store.faiss_indexes["document"]
    assert faiss_index.ntotal == 2

    prediction = pipeline.run(query="Who made the PDF specification?", params={"DPRRetriever": {"top_k": 10}})

    assert prediction["query"] == "Who made the PDF specification?"
    assert len(prediction["documents"]) == 2
    clean_faiss_document_store()


@pytest.mark.slow
@pytest.mark.parametrize("retriever_with_docs", ["elasticsearch", "dpr", "embedding"], indirect=True)
@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_documentsearch_es_authentication(retriever_with_docs, document_store_with_docs: ElasticsearchDocumentStore):
    if isinstance(retriever_with_docs, (DensePassageRetriever, EmbeddingRetriever)):
        document_store_with_docs.update_embeddings(retriever=retriever_with_docs)
    mock_client = Mock(wraps=document_store_with_docs.client)
    document_store_with_docs.client = mock_client
    auth_headers = {"Authorization": "Basic YWRtaW46cm9vdA=="}
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    prediction = pipeline.run(
        query="Who lives in Berlin?",
        params={"Retriever": {"top_k": 10, "headers": auth_headers}},
    )
    assert prediction is not None
    assert len(prediction["documents"]) == 5
    mock_client.search.assert_called_once()
    args, kwargs = mock_client.search.call_args
    assert "headers" in kwargs
    assert kwargs["headers"] == auth_headers


@pytest.mark.slow
@pytest.mark.parametrize("retriever_with_docs", ["tfidf"], indirect=True)
def test_documentsearch_document_store_authentication(retriever_with_docs, document_store_with_docs):
    mock_client = None
    if isinstance(document_store_with_docs, ElasticsearchDocumentStore):
        es_document_store: ElasticsearchDocumentStore = document_store_with_docs
        mock_client = Mock(wraps=es_document_store.client)
        es_document_store.client = mock_client
    auth_headers = {"Authorization": "Basic YWRtaW46cm9vdA=="}
    pipeline = DocumentSearchPipeline(retriever=retriever_with_docs)
    if not mock_client:
        with pytest.raises(Exception):
            prediction = pipeline.run(
                query="Who lives in Berlin?",
                params={"Retriever": {"top_k": 10, "headers": auth_headers}},
            )
    else:
        prediction = pipeline.run(
            query="Who lives in Berlin?",
            params={"Retriever": {"top_k": 10, "headers": auth_headers}},
        )
        assert prediction is not None
        assert len(prediction["documents"]) == 5
        mock_client.count.assert_called_once()
        args, kwargs = mock_client.count.call_args
        assert "headers" in kwargs
        assert kwargs["headers"] == auth_headers


def test_route_documents_by_content_type():
    # Test routing by content_type
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


def test_route_documents_by_metafield(test_docs_xs):
    # Test routing by metadata field
    docs = [Document.from_dict(doc) if isinstance(doc, dict) else doc for doc in test_docs_xs]
    route_documents = RouteDocuments(split_by="meta_field", metadata_values=["test1", "test3", "test5"])
    result, _ = route_documents.run(docs)
    assert len(result["output_1"]) == 1
    assert len(result["output_2"]) == 1
    assert len(result["output_3"]) == 1
    assert result["output_1"][0].meta["meta_field"] == "test1"
    assert result["output_2"][0].meta["meta_field"] == "test3"
    assert result["output_3"][0].meta["meta_field"] == "test5"


@pytest.mark.parametrize("join_mode", ["concatenate", "merge"])
def test_join_answers(join_mode):
    inputs = [{"answers": [Answer(answer="answer 1", score=0.7)]}, {"answers": [Answer(answer="answer 2", score=0.8)]}]

    join_answers = JoinAnswers(join_mode=join_mode)
    result, _ = join_answers.run(inputs)
    assert len(result["answers"]) == 2
    assert result["answers"] == sorted(result["answers"], reverse=True)

    result, _ = join_answers.run(inputs, top_k_join=1)
    assert len(result["answers"]) == 1
    assert result["answers"][0].answer == "answer 2"


def clean_faiss_document_store():
    if Path("existing_faiss_document_store").exists():
        os.remove("existing_faiss_document_store")
    if Path("existing_faiss_document_store.json").exists():
        os.remove("existing_faiss_document_store.json")
    if Path("faiss_document_store.db").exists():
        os.remove("faiss_document_store.db")
