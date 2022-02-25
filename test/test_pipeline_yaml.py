import pytest
import json
import networkx as nx

import haystack
from haystack import __version__, BaseComponent
from haystack.nodes._json_schema import get_json_schema
from haystack.pipelines import Pipeline
from haystack.document_stores import BaseDocumentStore
from haystack.nodes import BaseReader, BaseRetriever, FileTypeClassifier
from haystack.errors import PipelineConfigError

from .conftest import SAMPLES_PATH
from . import test_pipeline_yaml  # For mocking


class MockDocumentStore(BaseDocumentStore):

    def _create_document_field_map(self, *a, **k):
        pass

    def delete_documents(self, *a, **k):
        pass

    def delete_labels(self, *a, **k):
        pass

    def get_all_documents(self, *a, **k):
        pass

    def get_all_documents_generator(self, *a, **k):
        pass

    def get_all_labels(self, *a, **k):
        pass

    def get_document_by_id(self, *a, **k):
        pass

    def get_document_count(self, *a, **k):
        pass
 
    def get_documents_by_id(self, *a, **k):
        pass

    def get_label_count(self, *a, **k):
        pass

    def query_by_embedding(self, *a, **k):
        pass

    def write_documents(self, *a, **k):
        pass

    def write_labels(self, *a, **k):
        pass


class MockReader(BaseReader):
    pass

class MockRetriever(BaseRetriever):
    pass


@pytest.fixture(autouse=True)
def mock_importable_nodes_list(request, monkeypatch):
    # Do not patch integration tests
    if 'integration' in request.keywords:
        return

    monkeypatch.setattr(BaseComponent, "_find_subclasses_in_modules", lambda *a, **k: [
        (test_pipeline_yaml, MockDocumentStore),
        (test_pipeline_yaml, MockReader),
        (test_pipeline_yaml, MockRetriever),
    ])



@pytest.fixture(autouse=True)
def mock_json_schema(request, monkeypatch, tmp_path):
    # Do not patch integration tests
    if 'integration' in request.keywords:
        return

    monkeypatch.setattr(haystack.pipelines.base, "JSON_SCHEMAS_PATH", tmp_path)
    monkeypatch.setattr(BaseComponent, "_find_subclasses_in_modules", lambda *a, **k: [
        (test_pipeline_yaml, MockDocumentStore),
        (test_pipeline_yaml, MockReader),
        (test_pipeline_yaml, MockRetriever),
    ])

    filename = f"haystack-pipeline-{__version__}.schema.json"
    test_schema = get_json_schema(filename=filename)

    with open(tmp_path / filename, "w") as schema_file:
        json.dump(test_schema, schema_file, indent=4)



@pytest.fixture
def mock_110_json_schema(request, monkeypatch, tmp_path):
    # Do not patch integration tests
    if 'integration' in request.keywords:
        return

    monkeypatch.setattr(haystack.nodes._json_schema, "haystack_version", "1.1.0")
    monkeypatch.setattr(haystack.pipelines.base, "JSON_SCHEMAS_PATH", tmp_path)
    monkeypatch.setattr(BaseComponent, "_find_subclasses_in_modules", lambda *a, **k: [
        (test_pipeline_yaml, MockDocumentStore),
        (test_pipeline_yaml, MockReader),
        (test_pipeline_yaml, MockRetriever),
    ])
    
    filename = f"haystack-pipeline-1.1.0.schema.json"
    test_schema = get_json_schema(filename=filename)

    with open(tmp_path / filename, "w") as schema_file:
        json.dump(test_schema, schema_file, indent=4)


#
# Integration
#

@pytest.mark.integration
@pytest.mark.elasticsearch
def test_load_and_save_from_yaml(tmp_path):
    config_path = SAMPLES_PATH / "pipeline" / "test_pipeline.yaml"
    config_version = "1.2.0"  # TODO: might parametrize on "important" schema changes

    # Test the indexing pipeline: 
    # Load it
    indexing_pipeline = Pipeline.load_from_yaml(path=config_path, version=config_version, pipeline_name="indexing_pipeline")

    # Check if it works
    indexing_pipeline.get_document_store().delete_documents()
    assert indexing_pipeline.get_document_store().get_document_count() == 0
    indexing_pipeline.run(file_paths=SAMPLES_PATH / "pdf" / "sample_pdf_1.pdf")
    assert indexing_pipeline.get_document_store().get_document_count() > 0

    # Save it
    new_indexing_config = tmp_path / "test_indexing.yaml"
    indexing_pipeline.save_to_yaml(new_indexing_config)
    
    # Re-load it and compare the resulting pipelines
    new_indexing_pipeline = Pipeline.load_from_yaml(path=new_indexing_config)
    assert nx.is_isomorphic(new_indexing_pipeline.graph, indexing_pipeline.graph)

    # Check that modifying a pipeline modifies the output YAML
    modified_indexing_pipeline = Pipeline.load_from_yaml(path=new_indexing_config)
    modified_indexing_pipeline.add_node(FileTypeClassifier(), name="file_classifier", inputs=["File"])
    assert not nx.is_isomorphic(new_indexing_pipeline.graph, modified_indexing_pipeline.graph)

    # Test the query pipeline:
    # Load it
    query_pipeline = Pipeline.load_from_yaml(path=config_path, pipeline_name="query_pipeline")

    # Check if it works
    prediction = query_pipeline.run(
        query="Who made the PDF specification?", params={"ESRetriever": {"top_k": 10}, "Reader": {"top_k": 3}}
    )
    assert prediction["query"] == "Who made the PDF specification?"
    assert prediction["answers"][0].answer == "Adobe Systems"
    assert "_debug" not in prediction.keys()

    # Save it
    new_query_config = tmp_path / "test_query.yaml"
    query_pipeline.save_to_yaml(new_query_config)

    # Re-load it and compare the resulting pipelines
    new_query_pipeline = Pipeline.load_from_yaml(path=new_query_config)
    assert nx.is_isomorphic(new_query_pipeline.graph, query_pipeline.graph)

    # Check that different pipelines produce different files
    assert not nx.is_isomorphic(new_query_pipeline.graph, new_indexing_pipeline.graph)


#
# Unit 
#

def test_load_yaml_non_existing_file():
    with pytest.raises(PipelineConfigError):
        Pipeline.load_from_yaml(path=SAMPLES_PATH / "pipeline" / "I_dont_exist.yml")


def test_load_yaml_invalid_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write("this is not valid YAML!")
    with pytest.raises(PipelineConfigError):
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")


def test_load_yaml_missing_version(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(f"""
            components:
            - name: docstore
              type: MockDocumentStore
            pipelines:
            - name: query
              nodes:
              - name: docstore
                inputs:
                - Query
        """)
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
        assert "version" in str(e)


def test_load_yaml_custom_version(tmp_path, mock_110_json_schema):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(f"""
            version: 1.1.0
            components:
            - name: docstore
              type: MockDocumentStore
            pipelines:
            - name: query
              nodes:
              - name: docstore
                inputs:
                - Query
        """)
    Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml", version="1.1.0")


def test_load_yaml_non_existing_version(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(f"""
            version: random
            components:
            - name: docstore
              type: MockDocumentStore
            pipelines:
            - name: query
              nodes:
              - name: docstore
                inputs:
                - Query
        """)
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml", version="random")
        assert "version" in str(e) and "random" in str(e)
 

def test_load_yaml_no_components(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(f"""
            version: {__version__}
            components:
            pipelines:
            - name: query
              nodes:
        """)
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
        assert "components" in str(e)


def test_load_yaml_wrong_component(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(f"""
            version: {__version__}
            components:
            - name: docstore
              type: ImaginaryDocumentStore
            pipelines:
            - name: query
              nodes:
              - name: docstore
                inputs:
                - Query
        """)
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
        assert "ImaginaryDocumentStore" in str(e)


def test_load_yaml_no_pipelines(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(f"""
            version: {__version__}
            components:
            - name: docstore
              type: MockDocumentStore
            pipelines:
        """)
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
        assert "pipeline" in str(e)


def test_load_yaml_invalid_pipeline_name(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(f"""
            version: {__version__}
            components:
            - name: docstore
              type: MockDocumentStore
            pipelines:
            - name: query
              nodes:
              - name: docstore
                inputs:
                - Query
        """)
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml", pipeline_name="invalid")
        assert "invalid" in str(e) and "pipeline" in str(e)


def test_load_yaml_pipeline_with_type(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(f"""
            version: {__version__}
            components:
            - name: docstore
              type: MockDocumentStore
            pipelines:
            - name: query
              type: Query
              nodes:
              - name: docstore
                inputs:
                - Query
        """)
    Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")


def test_load_yaml_pipeline_with_wrong_nodes(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(f"""
            version: {__version__}
            components:
            - name: docstore
              type: MockDocumentStore
            pipelines:
            - name: query
              nodes:
              - name: not_existing_node
                inputs:
                - Query
        """)
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
        assert "not_existing_node" in str(e)


def test_load_yaml_pipeline_not_a_dag(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(f"""
            version: {__version__}
            components:
            - name: retriever
              type: MockRetriever
            - name: reader
              type: MockRetriever
            pipelines:
            - name: query
              nodes:
              - name: retriever
                inputs:
                - reader
              - name: reader
                inputs:
                - retriever
        """)
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")



# def test_load_and_save_yaml_prebuilt_pipelines(document_store, tmp_path):
#     # populating index
#     pipeline = Pipeline.load_from_yaml(
#         SAMPLES_PATH / "pipeline" / "test_pipeline.yaml", pipeline_name="indexing_pipeline"
#     )
#     pipeline.run(file_paths=SAMPLES_PATH / "pdf" / "sample_pdf_1.pdf")
#     # test correct load of query pipeline from yaml
#     pipeline = ExtractiveQAPipeline.load_from_yaml(
#         SAMPLES_PATH / "pipeline" / "test_pipeline.yaml", pipeline_name="query_pipeline"
#     )
#     prediction = pipeline.run(
#         query="Who made the PDF specification?", params={"ESRetriever": {"top_k": 10}, "Reader": {"top_k": 3}}
#     )
#     assert prediction["query"] == "Who made the PDF specification?"
#     assert prediction["answers"][0].answer == "Adobe Systems"
#     assert "_debug" not in prediction.keys()

#     # test invalid pipeline name
#     with pytest.raises(Exception):
#         ExtractiveQAPipeline.load_from_yaml(
#             path=SAMPLES_PATH / "pipeline" / "test_pipeline.yaml", pipeline_name="invalid"
#         )
#     # test config export
#     pipeline.save_to_yaml(tmp_path / "test.yaml")
#     with open(tmp_path / "test.yaml", "r", encoding="utf-8") as stream:
#         saved_yaml = stream.read()
#     expected_yaml = f"""
#         components:
#         - name: ESRetriever
#           params:
#             document_store: ElasticsearchDocumentStore
#           type: ElasticsearchRetriever
#         - name: ElasticsearchDocumentStore
#           params:
#             index: haystack_test
#             label_index: haystack_test_label
#           type: ElasticsearchDocumentStore
#         - name: Reader
#           params:
#             model_name_or_path: deepset/roberta-base-squad2
#             no_ans_boost: -10
#             num_processes: 0
#           type: FARMReader
#         pipelines:
#         - name: query
#           nodes:
#           - inputs:
#             - Query
#             name: ESRetriever
#           - inputs:
#             - ESRetriever
#             name: Reader
#           type: Pipeline
#         version: {__version__}
#     """
#     assert saved_yaml.replace(" ", "").replace("\n", "") == expected_yaml.replace(" ", "").replace("\n", "")


# def test_load_tfidfretriever_yaml(tmp_path):
#     documents = [
#         {
#             "content": "A Doc specifically talking about haystack. Haystack can be used to scale QA models to large document collections."
#         }
#     ]
#     pipeline = Pipeline.load_from_yaml(
#         SAMPLES_PATH / "pipeline" / "test_pipeline_tfidfretriever.yaml", pipeline_name="query_pipeline"
#     )
#     with pytest.raises(Exception) as exc_info:
#         pipeline.run(
#             query="What can be used to scale QA models to large document collections?",
#             params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}},
#         )
#     exception_raised = str(exc_info.value)
#     assert "Retrieval requires dataframe df and tf-idf matrix" in exception_raised

#     pipeline.get_node(name="Retriever").document_store.write_documents(documents=documents)
#     prediction = pipeline.run(
#         query="What can be used to scale QA models to large document collections?",
#         params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 3}},
#     )
#     assert prediction["query"] == "What can be used to scale QA models to large document collections?"
#     assert prediction["answers"][0].answer == "haystack"

