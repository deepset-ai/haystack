import networkx as nx
import pytest

from haystack import __version__
from haystack.pipelines import Pipeline
from haystack.nodes import FileTypeClassifier
from haystack.errors import PipelineConfigError, PipelineValidationError

from conftest import SAMPLES_PATH


@pytest.fixture
def pipeline_config_path(tmp_path):
    """
    This fixture ensures that the test YAML file is always loaded with the proper version,
    so that it won't fail validation for that.
    """
    source_path = SAMPLES_PATH / "pipeline" / "test_pipeline.yaml"
    test_path = tmp_path / "tmp.yml"

    config = None
    with open(source_path, "r") as source_file:
        config = source_file.read()
    if config:
        config = config.replace("1.1.0", __version__)
        with open(test_path, "w") as tmp_file:
            tmp_file.write(config)

    yield test_path


#
# Integration
#

@pytest.mark.integration
@pytest.mark.elasticsearch
def test_load_and_save_from_yaml(tmp_path, pipeline_config_path):

    # Test the indexing pipeline: 
    # Load it
    indexing_pipeline = Pipeline.load_from_yaml(path=pipeline_config_path, pipeline_name="indexing_pipeline")

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
    query_pipeline = Pipeline.load_from_yaml(path=pipeline_config_path, pipeline_name="query_pipeline")

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
    with pytest.raises(PipelineValidationError):
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")


def test_load_yaml_invalid_pipeline_name(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(f"""
            version: {__version__}
            components:
            - name: docstore
              type: ElasticsearchDocumentStore
            pipelines:
            - name: query
              nodes:
              - name: docstore
                inputs:
                - Query
        """)
    with pytest.raises(PipelineConfigError):
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml", pipeline_name="invalid")


def test_load_yaml_missing_version(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(f"""
            components:
            - name: docstore
              type: ElasticsearchDocumentStore
            pipelines:
            - name: query
              nodes:
              - name: docstore
                inputs:
                - Query
        """)
    with pytest.raises(PipelineValidationError):
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

