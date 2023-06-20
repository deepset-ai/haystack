import ssl
import json
import platform
import sys
from typing import Tuple
from copy import deepcopy
from unittest import mock

import pytest
from requests import PreparedRequest
import responses
import logging
import yaml

from haystack import __version__
from haystack.document_stores.deepsetcloud import DeepsetCloudDocumentStore
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.document_stores.memory import InMemoryDocumentStore
from haystack.nodes.other.join_docs import JoinDocuments
from haystack.nodes.base import BaseComponent
from haystack.nodes.retriever.sparse import BM25Retriever
from haystack.nodes.retriever.sparse import FilterRetriever
from haystack.pipelines import (
    Pipeline,
    RootNode,
    GenerativeQAPipeline,
    FAQPipeline,
    ExtractiveQAPipeline,
    SearchSummarizationPipeline,
    TranslationWrapperPipeline,
    RetrieverQuestionGenerationPipeline,
    QuestionAnswerGenerationPipeline,
    DocumentSearchPipeline,
    QuestionGenerationPipeline,
    MostSimilarDocumentsPipeline,
)
from haystack.pipelines.config import get_component_definitions
from haystack.pipelines.utils import generate_code
from haystack.errors import PipelineConfigError
from haystack.nodes import PreProcessor, TextConverter
from haystack.utils.deepsetcloud import DeepsetCloudError
from haystack import Answer

from ..conftest import (
    MOCK_DC,
    DC_API_ENDPOINT,
    DC_API_KEY,
    DC_TEST_INDEX,
    MockDocumentStore,
    MockSeq2SegGenerator,
    MockRetriever,
    MockNode,
    deepset_cloud_fixture,
    MockReader,
    MockSummarizer,
    MockTranslator,
    MockQuestionGenerator,
)

logger = logging.getLogger(__name__)


@pytest.fixture
def reduce_windows_recursion_limit():
    """
    Prevents Windows CI from crashing with Stackoverflow in situations we want to provoke a RecursionError
    """
    is_windows = platform.system() == "Windows"
    default_recursion_limit = sys.getrecursionlimit()
    if is_windows:
        reduced_recursion_limit = default_recursion_limit // 2
        logger.warning("Reducing recursion limit to %s", reduced_recursion_limit)
        sys.setrecursionlimit(reduced_recursion_limit)
    yield
    if is_windows:
        logger.warning("Resetting recursion limit to %s", default_recursion_limit)
        sys.setrecursionlimit(default_recursion_limit)


class ParentComponent(BaseComponent):
    outgoing_edges = 1

    def __init__(self, dependent: BaseComponent) -> None:
        super().__init__()

    def run(*args, **kwargs):
        logger.info("ParentComponent run() was called")

    def run_batch(*args, **kwargs):
        pass


class ParentComponent2(BaseComponent):
    outgoing_edges = 1

    def __init__(self, dependent: BaseComponent) -> None:
        super().__init__()

    def run(*args, **kwargs):
        logger.info("ParentComponent2 run() was called")

    def run_batch(*args, **kwargs):
        pass


class ChildComponent(BaseComponent):
    outgoing_edges = 0

    def __init__(self, some_key: str = None) -> None:
        super().__init__()

    def run(*args, **kwargs):
        logger.info("ChildComponent run() was called")

    def run_batch(*args, **kwargs):
        pass


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
def test_to_code_creates_same_pipelines(samples_path):
    index_pipeline = Pipeline.load_from_yaml(
        samples_path / "pipeline" / "test.haystack-pipeline.yml", pipeline_name="indexing_pipeline"
    )
    query_pipeline = Pipeline.load_from_yaml(
        samples_path / "pipeline" / "test.haystack-pipeline.yml", pipeline_name="query_pipeline"
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_get_config_custom_node_with_params():
    class CustomNode(MockNode):
        def __init__(self, param: int):
            super().__init__()
            self.param = param

    pipeline = Pipeline()
    pipeline.add_node(CustomNode(param=10), name="custom_node", inputs=["Query"])

    assert len(pipeline.get_config()["components"]) == 1
    assert pipeline.get_config()["components"][0]["params"] == {"param": 10}


@pytest.mark.unit
def test_get_config_custom_node_with_positional_params():
    class CustomNode(MockNode):
        def __init__(self, param: int = 1):
            super().__init__()
            self.param = param

    pipeline = Pipeline()
    pipeline.add_node(CustomNode(10), name="custom_node", inputs=["Query"])

    assert len(pipeline.get_config()["components"]) == 1
    assert pipeline.get_config()["components"][0]["params"] == {"param": 10}


@pytest.mark.unit
def test_get_config_multi_output_node():
    class MultiOutputNode(BaseComponent):
        outgoing_edges = 2

        def run(self, *a, **k):
            pass

        def run_batch(self, *a, **k):
            pass

    pipeline = Pipeline()
    pipeline.add_node(MultiOutputNode(), name="multi_output_node", inputs=["Query"])
    pipeline.add_node(MockNode(), name="fork_1", inputs=["multi_output_node.output_1"])
    pipeline.add_node(MockNode(), name="fork_2", inputs=["multi_output_node.output_2"])
    pipeline.add_node(JoinNode(), name="join_node", inputs=["fork_1", "fork_2"])

    config = pipeline.get_config()
    assert len(config["components"]) == 4
    assert len(config["pipelines"]) == 1
    nodes = config["pipelines"][0]["nodes"]
    assert len(nodes) == 4

    assert nodes[0]["name"] == "multi_output_node"
    assert len(nodes[0]["inputs"]) == 1
    assert "Query" in nodes[0]["inputs"]

    assert nodes[1]["name"] == "fork_1"
    assert len(nodes[1]["inputs"]) == 1
    assert "multi_output_node.output_1" in nodes[1]["inputs"]

    assert nodes[2]["name"] == "fork_2"
    assert len(nodes[2]["inputs"]) == 1
    assert "multi_output_node.output_2" in nodes[2]["inputs"]

    assert nodes[3]["name"] == "join_node"
    assert len(nodes[3]["inputs"]) == 2
    assert "fork_1" in nodes[3]["inputs"]
    assert "fork_2" in nodes[3]["inputs"]


@pytest.mark.unit
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


@pytest.mark.unit
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
        'document_store.name = "DocumentStore"\n'
        "retri = BM25Retriever(document_store=document_store)\n"
        "retri_2 = TfidfRetriever(document_store=document_store)\n"
        "\n"
        "p = Pipeline()\n"
        'p.add_node(component=retri, name="retri", inputs=["Query"])\n'
        'p.add_node(component=retri_2, name="retri2", inputs=["Query"])'
    )


@pytest.mark.unit
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
        'document_store.name = "DocumentStore"\n'
        "retri = BM25Retriever(document_store=document_store)\n"
        "\n"
        "p = Pipeline()\n"
        'p.add_node(component=retri, name="retri", inputs=["Query"])'
    )


@pytest.mark.unit
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
        'document_store.name = "DocumentStore"\n'
        "retri = BM25Retriever(document_store=document_store)\n"
        "\n"
        "p = Pipeline()\n"
        'p.add_node(component=retri, name="retri", inputs=["Query"])'
    )


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_pipeline_classify_type(tmp_path):
    pipe = GenerativeQAPipeline(generator=MockSeq2SegGenerator(), retriever=MockRetriever())
    assert pipe.get_type().startswith("GenerativeQAPipeline")

    pipe = FAQPipeline(retriever=MockRetriever())
    assert pipe.get_type().startswith("FAQPipeline")

    pipe = ExtractiveQAPipeline(reader=MockReader(), retriever=MockRetriever())
    assert pipe.get_type().startswith("ExtractiveQAPipeline")

    search_pipe = SearchSummarizationPipeline(summarizer=MockSummarizer(), retriever=MockRetriever())
    assert search_pipe.get_type().startswith("SearchSummarizationPipeline")

    pipe = RetrieverQuestionGenerationPipeline(retriever=MockRetriever(), question_generator=MockQuestionGenerator())
    assert pipe.get_type().startswith("RetrieverQuestionGenerationPipeline")

    qag_pipe = QuestionAnswerGenerationPipeline(question_generator=MockQuestionGenerator(), reader=MockReader())
    assert qag_pipe.get_type().startswith("QuestionAnswerGenerationPipeline")

    pipe = DocumentSearchPipeline(retriever=MockRetriever())
    assert pipe.get_type().startswith("DocumentSearchPipeline")

    pipe = QuestionGenerationPipeline(question_generator=MockQuestionGenerator())
    assert pipe.get_type().startswith("QuestionGenerationPipeline")

    pipe = TranslationWrapperPipeline(
        input_translator=MockTranslator(), output_translator=MockTranslator(), pipeline=qag_pipe
    )
    pipe.get_type().startswith("TranslationWrapperPipeline")

    pipe = MostSimilarDocumentsPipeline(document_store=MockDocumentStore())
    assert pipe.get_type().startswith("MostSimilarDocumentsPipeline")

    # previously misclassified as "UnknownPipeline"
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
               version: ignore
               components:
               - name: document_store
                 type: MockDocumentStore
               - name: retriever
                 type: MockRetriever
               - name: retriever_2
                 type: MockRetriever
               pipelines:
               - name: my_pipeline
                 nodes:
                 - name: retriever
                   inputs:
                   - Query
                 - name: retriever_2
                   inputs:
                   - Query
                 - name: document_store
                   inputs:
                   - retriever

           """
        )
    pipe = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    # two retrievers but still a DocumentSearchPipeline
    assert pipe.get_type().startswith("DocumentSearchPipeline")

    # previously misclassified as "UnknownPipeline"
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
               version: ignore
               components:
               - name: document_store
                 type: MockDocumentStore
               - name: retriever
                 type: MockRetriever
               - name: retriever_2
                 type: MockRetriever
               - name: retriever_3
                 type: MockRetriever
               pipelines:
               - name: my_pipeline
                 nodes:
                 - name: retriever
                   inputs:
                   - Query
                 - name: retriever_2
                   inputs:
                   - Query
                 - name: retriever_3
                   inputs:
                   - Query
                 - name: document_store
                   inputs:
                   - retriever

           """
        )
    pipe = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    # three retrievers but still a DocumentSearchPipeline
    assert pipe.get_type().startswith("DocumentSearchPipeline")

    # previously misclassified as "UnknownPipeline"
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
               version: ignore
               components:
               - name: document_store
                 type: MockDocumentStore
               - name: retriever
                 type: BM25Retriever
               pipelines:
               - name: my_pipeline
                 nodes:
                 - name: retriever
                   inputs:
                   - Query
                 - name: document_store
                   inputs:
                   - retriever

           """
        )
    pipe = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    # BM25Retriever used - still a DocumentSearchPipeline
    assert pipe.get_type().startswith("DocumentSearchPipeline")


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_load_from_deepset_cloud_query(samples_path):
    if MOCK_DC:
        with open(samples_path / "dc" / "pipeline_config.json", "r") as f:
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
    assert document_store.name == "DocumentStore"

    prediction = query_pipeline.run(query="man on horse", params={})

    assert prediction["query"] == "man on horse"
    assert len(prediction["documents"]) == 1
    assert prediction["documents"][0].id == "test_doc"


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_load_from_deepset_cloud_indexing(caplog, samples_path):
    if MOCK_DC:
        with open(samples_path / "dc" / "pipeline_config.json", "r") as f:
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

    with caplog.at_level(logging.INFO):
        indexing_pipeline.run(file_paths=[samples_path / "docs" / "doc_1.txt"])
        assert "Note that DeepsetCloudDocumentStore does not support write operations." in caplog.text
        assert "Input to write_documents: {" in caplog.text


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_list_pipelines_on_deepset_cloud():
    pipelines = Pipeline.list_pipelines_on_deepset_cloud(api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY)
    assert len(pipelines) == 1
    assert pipelines[0]["name"] == DC_TEST_INDEX


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_save_to_deepset_cloud(samples_path):
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

        with open(samples_path / "dc" / "pipeline_config.json", "r") as f:
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
def test_deploy_on_deepset_cloud(caplog):
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

    with caplog.at_level(logging.INFO):
        pipeline_url = f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline/search"
        Pipeline.deploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
        )
        assert "Pipeline config 'test_new_non_existing_pipeline' successfully deployed." in caplog.text
        assert pipeline_url in caplog.text
        assert "curl" in caplog.text


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_deploy_on_deepset_cloud_no_curl_message(caplog):
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

    with caplog.at_level(logging.INFO):
        pipeline_url = f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline/search"
        Pipeline.deploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline",
            api_endpoint=DC_API_ENDPOINT,
            api_key=DC_API_KEY,
            show_curl_message=False,
        )
        assert "Pipeline config 'test_new_non_existing_pipeline' successfully deployed." in caplog.text
        assert pipeline_url in caplog.text
        assert "curl" not in caplog.text


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
        match="Deployment of pipeline config 'test_new_non_existing_pipeline' aborted. Undeployment was requested.",
    ):
        Pipeline.deploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
        )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_failed_deploy_on_deepset_cloud():
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline/deploy",
            json={"status": "DEPLOYMENT_SCHEDULED"},
            status=200,
        )

        # status will be first undeployed, after deploy() it's in progress twice and the third time deployment failed
        status_flow = ["UNDEPLOYED", "DEPLOYMENT_IN_PROGRESS", "DEPLOYMENT_IN_PROGRESS", "DEPLOYMENT_FAILED"]
        for status in status_flow:
            responses.add(
                method=responses.GET,
                url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
                json={"status": status},
                status=200,
            )
    with pytest.raises(
        DeepsetCloudError,
        match=f"Deployment of pipeline config 'test_new_non_existing_pipeline' failed. "
        "This might be caused by an exception in deepset Cloud or a runtime error in the pipeline. "
        "You can try to run this pipeline locally first.",
    ):
        Pipeline.deploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
        )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_unexpected_failed_deploy_on_deepset_cloud():
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline/deploy",
            json={"status": "DEPLOYMENT_SCHEDULED"},
            status=200,
        )

        # status will be first undeployed, after deploy() it's in deployment failed
        status_flow = ["UNDEPLOYED", "DEPLOYMENT_FAILED"]
        for status in status_flow:
            responses.add(
                method=responses.GET,
                url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
                json={"status": status},
                status=200,
            )
    with pytest.raises(
        DeepsetCloudError,
        match=f"Deployment of pipeline config 'test_new_non_existing_pipeline' failed. "
        "This might be caused by an exception in deepset Cloud or a runtime error in the pipeline. "
        "You can try to run this pipeline locally first.",
    ):
        Pipeline.deploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
        )


@pytest.mark.usefixtures(deepset_cloud_fixture.__name__)
@responses.activate
def test_deploy_on_deepset_cloud_with_failed_start_state(caplog):
    if MOCK_DC:
        responses.add(
            method=responses.POST,
            url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline/deploy",
            json={"status": "DEPLOYMENT_SCHEDULED"},
            status=200,
        )

        # status will be first in failed (but not invalid) state, after deploy() it's in progress twice and third time deployed
        status_flow = ["DEPLOYMENT_FAILED", "DEPLOYMENT_IN_PROGRESS", "DEPLOYMENT_IN_PROGRESS", "DEPLOYED"]
        for status in status_flow:
            responses.add(
                method=responses.GET,
                url=f"{DC_API_ENDPOINT}/workspaces/default/pipelines/test_new_non_existing_pipeline",
                json={"status": status},
                status=200,
            )

    with caplog.at_level(logging.WARNING):
        Pipeline.deploy_on_deepset_cloud(
            pipeline_config_name="test_new_non_existing_pipeline", api_endpoint=DC_API_ENDPOINT, api_key=DC_API_KEY
        )
        assert (
            "Pipeline config 'test_new_non_existing_pipeline' is in a failed state 'PipelineStatus.DEPLOYMENT_FAILED'."
            in caplog.text
        )
        assert "This might be caused by a previous error during (un)deployment." in caplog.text


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
        match="Undeployment of pipeline config 'test_new_non_existing_pipeline' aborted. Deployment was requested.",
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


@pytest.mark.unit
def test_graph_validation_invalid_edge():
    docstore = MockDocumentStore()
    retriever = DummyRetriever(document_store=docstore)
    pipeline = Pipeline()
    pipeline.add_node(name="DocStore", component=docstore, inputs=["Query"])

    with pytest.raises(PipelineConfigError, match="DocStore has only 1 outgoing edge"):
        pipeline.add_node(name="Retriever", component=retriever, inputs=["DocStore.output_2"])


@pytest.mark.unit
def test_graph_validation_non_existing_edge():
    docstore = MockDocumentStore()
    retriever = DummyRetriever(document_store=docstore)
    pipeline = Pipeline()
    pipeline.add_node(name="DocStore", component=docstore, inputs=["Query"])

    with pytest.raises(PipelineConfigError, match="'wrong_edge_label' is not a valid edge name"):
        pipeline.add_node(name="Retriever", component=retriever, inputs=["DocStore.wrong_edge_label"])


@pytest.mark.unit
def test_graph_validation_invalid_node():
    docstore = MockDocumentStore()
    retriever = DummyRetriever(document_store=docstore)
    pipeline = Pipeline()
    pipeline.add_node(name="DocStore", component=docstore, inputs=["Query"])

    with pytest.raises(PipelineConfigError, match="Cannot find node 'InvalidNode'"):
        pipeline.add_node(name="Retriever", component=retriever, inputs=["InvalidNode"])


@pytest.mark.unit
def test_graph_validation_invalid_root_node():
    docstore = MockDocumentStore()
    pipeline = Pipeline()

    with pytest.raises(PipelineConfigError, match="one single root node"):
        pipeline.add_node(name="DocStore", component=docstore, inputs=["InvalidNode"])


@pytest.mark.unit
def test_graph_validation_no_root_node():
    docstore = MockNode()
    pipeline = Pipeline()

    with pytest.raises(PipelineConfigError, match="one single root node"):
        pipeline.add_node(name="Node", component=docstore, inputs=[])


@pytest.mark.unit
def test_graph_validation_two_root_nodes():
    docstore = MockNode()
    pipeline = Pipeline()

    with pytest.raises(PipelineConfigError, match="one single root node"):
        pipeline.add_node(name="Node", component=docstore, inputs=["Query", "File"])

    with pytest.raises(PipelineConfigError, match="one single root node"):
        pipeline.add_node(name="Node", component=docstore, inputs=["Query", "Query"])


@pytest.mark.unit
def test_graph_validation_duplicate_node_instance():
    node = MockNode()
    pipeline = Pipeline()
    pipeline.add_node(name="node_a", component=node, inputs=["Query"])

    with pytest.raises(PipelineConfigError, match="You have already added the same instance to the pipeline"):
        pipeline.add_node(name="node_b", component=node, inputs=["node_a"])


@pytest.mark.unit
def test_graph_validation_duplicate_node():
    node = MockNode()
    other_node = MockNode()
    pipeline = Pipeline()
    pipeline.add_node(name="node", component=node, inputs=["Query"])
    with pytest.raises(PipelineConfigError, match="'node' is already in the pipeline"):
        pipeline.add_node(name="node", component=other_node, inputs=["Query"])


# See https://github.com/deepset-ai/haystack/issues/2568
@pytest.mark.unit
def test_pipeline_nodes_can_have_uncopiable_objects_as_args():
    class DummyNode(MockNode):
        def __init__(self, uncopiable: ssl.SSLContext):
            self.uncopiable = uncopiable

    node = DummyNode(uncopiable=ssl.SSLContext())
    pipeline = Pipeline()
    pipeline.add_node(component=node, name="node", inputs=["Query"])

    # If the object is getting copied, it will raise TypeError: cannot pickle 'SSLContext' object
    # `get_components_definitions()` should NOT copy objects to allow this usecase
    get_component_definitions(pipeline.get_config())


@pytest.mark.unit
def test_pipeline_env_vars_do_not_modify__component_config(caplog, monkeypatch):
    class DummyNode(MockNode):
        def __init__(self, replaceable: str):
            self.replaceable = replaceable

    monkeypatch.setenv("NODE_PARAMS_REPLACEABLE", "env value")

    node = DummyNode(replaceable="init value")
    pipeline = Pipeline()
    pipeline.add_node(component=node, name="node", inputs=["Query"])

    original_component_config = deepcopy(node._component_config)
    original_pipeline_config = deepcopy(pipeline.get_config())

    no_env_defs = get_component_definitions(pipeline.get_config(), overwrite_with_env_variables=False)

    with caplog.at_level(logging.INFO):
        env_defs = get_component_definitions(pipeline.get_config(), overwrite_with_env_variables=True)
        assert "overwritten with environment variable 'NODE_PARAMS_REPLACEABLE' value '***'." in caplog.text

    new_component_config = deepcopy(node._component_config)
    new_pipeline_config = deepcopy(pipeline.get_config())

    assert no_env_defs != env_defs
    assert no_env_defs["node"]["params"]["replaceable"] == "init value"
    assert env_defs["node"]["params"]["replaceable"] == "env value"

    assert original_component_config == new_component_config
    assert original_component_config["params"]["replaceable"] == "init value"
    assert new_component_config["params"]["replaceable"] == "init value"

    assert original_pipeline_config == new_pipeline_config
    assert original_pipeline_config["components"][0]["params"]["replaceable"] == "init value"
    assert new_pipeline_config["components"][0]["params"]["replaceable"] == "init value"


@pytest.mark.unit
def test_pipeline_env_vars_do_not_modify_pipeline_config(monkeypatch):
    class DummyNode(MockNode):
        def __init__(self, replaceable: str):
            self.replaceable = replaceable

    monkeypatch.setenv("NODE_PARAMS_REPLACEABLE", "env value")

    node = DummyNode(replaceable="init value")
    pipeline = Pipeline()
    pipeline.add_node(component=node, name="node", inputs=["Query"])

    pipeline_config = pipeline.get_config()
    original_pipeline_config = deepcopy(pipeline_config)

    get_component_definitions(pipeline_config, overwrite_with_env_variables=True)

    assert original_pipeline_config == pipeline_config
    assert original_pipeline_config["components"][0]["params"]["replaceable"] == "init value"
    assert pipeline_config["components"][0]["params"]["replaceable"] == "init value"


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
def test_pipeline_components():
    class Node(BaseComponent):
        outgoing_edges = 1

        def run(self):
            test = "test"
            return {"test": test}, "output_1"

        def run_batch(self):
            return

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


@pytest.mark.unit
def test_pipeline_get_document_store_from_components():
    doc_store = MockDocumentStore()
    pipeline = Pipeline()
    pipeline.add_node(name="A", component=doc_store, inputs=["File"])

    assert doc_store == pipeline.get_document_store()


@pytest.mark.unit
def test_pipeline_get_document_store_from_components_multiple_doc_stores():
    doc_store_a = MockDocumentStore()
    doc_store_b = MockDocumentStore()
    pipeline = Pipeline()
    pipeline.add_node(name="A", component=doc_store_a, inputs=["File"])
    pipeline.add_node(name="B", component=doc_store_b, inputs=["File"])

    with pytest.raises(Exception, match="Multiple Document Stores found in Pipeline"):
        pipeline.get_document_store()


@pytest.mark.unit
def test_pipeline_get_document_store_from_retriever():
    doc_store = MockDocumentStore()
    retriever = DummyRetriever(document_store=doc_store)
    pipeline = Pipeline()
    pipeline.add_node(name="A", component=retriever, inputs=["Query"])

    assert doc_store == pipeline.get_document_store()


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_batch_querying_single_query(document_store_with_docs, samples_path):
    query_pipeline = Pipeline.load_from_yaml(
        samples_path / "pipeline" / "test.haystack-pipeline.yml", pipeline_name="query_pipeline"
    )
    query_pipeline.components["ESRetriever"].document_store = document_store_with_docs
    result = query_pipeline.run_batch(queries=["Who lives in Berlin?"])
    assert isinstance(result["answers"], list)
    assert isinstance(result["answers"][0], list)
    assert isinstance(result["answers"][0][0], Answer)
    assert len(result["answers"]) == 1  # Predictions for 1 collection of docs (single query)
    assert len(result["answers"][0]) == 5  # Reader top-k set to 5


@pytest.mark.parametrize("document_store_with_docs", ["elasticsearch"], indirect=True)
def test_batch_querying_multiple_queries(document_store_with_docs, samples_path):
    query_pipeline = Pipeline.load_from_yaml(
        samples_path / "pipeline" / "test.haystack-pipeline.yml", pipeline_name="query_pipeline"
    )
    query_pipeline.components["ESRetriever"].document_store = document_store_with_docs
    result = query_pipeline.run_batch(queries=["Who lives in Berlin?", "Who lives in New York?"])
    # As we have a list of queries as input, this Pipeline will retrieve a list of relevant documents for each of the
    # queries (resulting in a list of lists of documents), apply the reader with each query and their corresponding
    # retrieved documents and return the predicted answers for each document list
    assert isinstance(result["answers"], list)
    assert isinstance(result["answers"][0], list)
    assert isinstance(result["answers"][0][0], Answer)
    assert len(result["answers"]) == 2  # Predictions for 2 collections of documents
    assert len(result["answers"][0]) == 5  # top-k of 5 for collection of docs


@pytest.mark.unit
def test_fix_to_pipeline_execution_when_join_follows_join():
    # wire up 4 retrievers, each with one document
    document_store_1 = InMemoryDocumentStore()
    retriever_1 = FilterRetriever(document_store_1, scale_score=True)
    dicts_1 = [{"content": "Alpha", "score": 0.552}]
    document_store_1.write_documents(dicts_1)

    document_store_2 = InMemoryDocumentStore()
    retriever_2 = FilterRetriever(document_store_2, scale_score=True)
    dicts_2 = [{"content": "Beta", "score": 0.542}]
    document_store_2.write_documents(dicts_2)

    document_store_3 = InMemoryDocumentStore()
    retriever_3 = FilterRetriever(document_store_3, scale_score=True)
    dicts_3 = [{"content": "Gamma", "score": 0.532}]
    document_store_3.write_documents(dicts_3)

    document_store_4 = InMemoryDocumentStore()
    retriever_4 = FilterRetriever(document_store_4, scale_score=True)
    dicts_4 = [{"content": "Delta", "score": 0.512}]
    document_store_4.write_documents(dicts_4)

    # wire up a pipeline of the retrievers, with 4-way join
    pipeline = Pipeline()
    pipeline.add_node(component=retriever_1, name="Retriever1", inputs=["Query"])
    pipeline.add_node(component=retriever_2, name="Retriever2", inputs=["Query"])
    pipeline.add_node(component=retriever_3, name="Retriever3", inputs=["Query"])
    pipeline.add_node(component=retriever_4, name="Retriever4", inputs=["Query"])
    pipeline.add_node(
        component=JoinDocuments(weights=[0.25, 0.25, 0.25, 0.25], join_mode="merge"),
        name="Join",
        inputs=["Retriever1", "Retriever2", "Retriever3", "Retriever4"],
    )

    res = pipeline.run(query="Alpha Beta Gamma Delta")
    documents = res["documents"]
    assert len(documents) == 4  # all four documents should be found

    # wire up a pipeline of the retrievers, with join following join
    pipeline = Pipeline()
    pipeline.add_node(component=retriever_1, name="Retriever1", inputs=["Query"])
    pipeline.add_node(component=retriever_2, name="Retriever2", inputs=["Query"])
    pipeline.add_node(component=retriever_3, name="Retriever3", inputs=["Query"])
    pipeline.add_node(component=retriever_4, name="Retriever4", inputs=["Query"])
    pipeline.add_node(
        component=JoinDocuments(weights=[0.5, 0.5], join_mode="merge"),
        name="Join12",
        inputs=["Retriever1", "Retriever2"],
    )
    pipeline.add_node(
        component=JoinDocuments(weights=[0.5, 0.5], join_mode="merge"),
        name="Join34",
        inputs=["Retriever3", "Retriever4"],
    )
    pipeline.add_node(
        component=JoinDocuments(weights=[0.5, 0.5], join_mode="merge"), name="JoinFinal", inputs=["Join12", "Join34"]
    )

    res = pipeline.run(query="Alpha Beta Gamma Delta")
    documents = res["documents"]
    assert len(documents) == 4  # all four documents should be found


@pytest.mark.unit
def test_update_config_hash():
    fake_configs = {
        "version": "ignore",
        "components": [
            {
                "name": "MyReader",
                "type": "FARMReader",
                "params": {"no_ans_boost": -10, "model_name_or_path": "deepset/roberta-base-squad2"},
            }
        ],
        "pipelines": [
            {
                "name": "my_query_pipeline",
                "nodes": [
                    {"name": "MyRetriever", "inputs": ["Query"]},
                    {"name": "MyReader", "inputs": ["MyRetriever"]},
                ],
            }
        ],
    }
    with mock.patch("haystack.pipelines.base.Pipeline.get_config", return_value=fake_configs):
        test_pipeline = Pipeline()
        assert test_pipeline.config_hash == None
        test_pipeline.update_config_hash()
        assert test_pipeline.config_hash == "a30d3273de0d70e63e8cd91d915255b3"


@pytest.mark.unit
def test_load_from_config_w_param_that_equals_component_name():
    config = {
        "version": "ignore",
        "components": [{"name": "node", "type": "InMemoryDocumentStore", "params": {"index": "node"}}],
        "pipelines": [{"name": "indexing", "nodes": [{"name": "node", "inputs": ["File"]}]}],
    }

    pipeline = Pipeline.load_from_config(pipeline_config=config)
    assert pipeline.components["node"].name == "node"
    assert pipeline.components["node"].index == "node"
