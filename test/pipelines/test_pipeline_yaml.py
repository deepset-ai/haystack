from abc import abstractmethod
import logging
from numpy import mat
import pytest
import json
import logging
import inspect
import networkx as nx
from enum import Enum
from pydantic.dataclasses import dataclass
from typing import Any, Dict, List, Optional

import haystack
from haystack import Pipeline
from haystack.nodes import _json_schema
from haystack.nodes import FileTypeClassifier
from haystack.errors import HaystackError, PipelineConfigError, PipelineSchemaError, DocumentStoreError
from haystack.nodes.base import BaseComponent

from ..conftest import MockNode, MockDocumentStore, MockReader, MockRetriever
from .. import conftest


#
# Fixtures
#


@pytest.fixture(autouse=True)
def mock_json_schema(request, monkeypatch, tmp_path):
    """
    JSON schema with the main version and only mocked nodes.
    """
    # Do not patch integration tests
    if "integration" in request.keywords:
        return

    # Mock the subclasses list to make it very small, containing only mock nodes
    monkeypatch.setattr(
        haystack.nodes._json_schema,
        "find_subclasses_in_modules",
        lambda *a, **k: [(conftest, MockDocumentStore), (conftest, MockReader), (conftest, MockRetriever)],
    )
    # Point the JSON schema path to tmp_path
    monkeypatch.setattr(haystack.nodes._json_schema, "JSON_SCHEMAS_PATH", tmp_path)

    # Generate mock schema in tmp_path
    filename = f"haystack-pipeline-main.schema.json"
    test_schema = _json_schema.get_json_schema(filename=filename, version="ignore")

    with open(tmp_path / filename, "w") as schema_file:
        json.dump(test_schema, schema_file, indent=4)


#
# Integration
#


@pytest.mark.integration
@pytest.mark.elasticsearch
def test_load_and_save_from_yaml(tmp_path, samples_path):
    config_path = samples_path / "pipeline" / "test.haystack-pipeline.yml"

    # Test the indexing pipeline:
    # Load it
    indexing_pipeline = Pipeline.load_from_yaml(path=config_path, pipeline_name="indexing_pipeline")

    # Check if it works
    indexing_pipeline.get_document_store().delete_documents()
    assert indexing_pipeline.get_document_store().get_document_count() == 0
    indexing_pipeline.run(file_paths=samples_path / "pdf" / "sample_pdf_1.pdf")
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


@pytest.mark.unit
def test_load_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: retriever
              type: MockRetriever
            - name: reader
              type: MockReader
            pipelines:
            - name: query
              nodes:
              - name: retriever
                inputs:
                - Query
              - name: reader
                inputs:
                - retriever
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    assert len(pipeline.graph.nodes) == 3
    assert isinstance(pipeline.get_node("retriever"), MockRetriever)
    assert isinstance(pipeline.get_node("reader"), MockReader)


def test_load_yaml_elasticsearch_not_responding(tmp_path):
    # Test if DocumentStoreError is raised if elasticsearch instance is not responding (due to wrong port)
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: ESRetriever
              type: BM25Retriever
              params:
                document_store: DocumentStore
            - name: DocumentStore
              type: ElasticsearchDocumentStore
              params:
                port: 1234
            - name: PDFConverter
              type: PDFToTextConverter
            - name: Preprocessor
              type: PreProcessor
            pipelines:
            - name: query_pipeline
              nodes:
              - name: ESRetriever
                inputs: [Query]
            - name: indexing_pipeline
              nodes:
              - name: PDFConverter
                inputs: [File]
              - name: Preprocessor
                inputs: [PDFConverter]
              - name: ESRetriever
                inputs: [Preprocessor]
              - name: DocumentStore
                inputs: [ESRetriever]
        """
        )
    with pytest.raises(DocumentStoreError):
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml", pipeline_name="indexing_pipeline")


@pytest.mark.unit
def test_load_yaml_non_existing_file(samples_path):
    with pytest.raises(FileNotFoundError):
        Pipeline.load_from_yaml(path=samples_path / "pipeline" / "I_dont_exist.yml")


@pytest.mark.unit
def test_load_yaml_invalid_yaml(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write("this is not valid YAML!")
    with pytest.raises(PipelineConfigError):
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")


@pytest.mark.unit
def test_load_yaml_missing_version(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            """
            components:
            - name: docstore
              type: MockDocumentStore
            pipelines:
            - name: my_pipeline
              nodes:
              - name: docstore
                inputs:
                - Query
        """
        )
    with pytest.raises(PipelineConfigError, match="Validation failed") as e:
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
        assert "version" in str(e)


@pytest.mark.unit
def test_load_yaml_non_existing_version(tmp_path, caplog):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            """
            version: random
            components:
            - name: docstore
              type: MockDocumentStore
            pipelines:
            - name: my_pipeline
              nodes:
              - name: docstore
                inputs:
                - Query
        """
        )
    with caplog.at_level(logging.WARNING):
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
        assert "version 'random'" in caplog.text
        assert f"Haystack {haystack.__version__}" in caplog.text


@pytest.mark.unit
def test_load_yaml_non_existing_version_strict(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            """
            version: random
            components:
            - name: docstore
              type: MockDocumentStore
            pipelines:
            - name: my_pipeline
              nodes:
              - name: docstore
                inputs:
                - Query
        """
        )
    with pytest.raises(PipelineConfigError, match="Cannot load pipeline configuration of version random"):
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml", strict_version_check=True)


@pytest.mark.unit
def test_load_yaml_incompatible_version(tmp_path, caplog):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            """
            version: 1.1.0
            components:
            - name: docstore
              type: MockDocumentStore
            pipelines:
            - name: my_pipeline
              nodes:
              - name: docstore
                inputs:
                - Query
        """
        )
    with caplog.at_level(logging.WARNING):
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
        assert "version '1.1.0'" in caplog.text
        assert f"Haystack {haystack.__version__}" in caplog.text


@pytest.mark.unit
def test_load_yaml_incompatible_version_strict(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            """
            version: 1.1.0
            components:
            - name: docstore
              type: MockDocumentStore
            pipelines:
            - name: my_pipeline
              nodes:
              - name: docstore
                inputs:
                - Query
        """
        )
    with pytest.raises(PipelineConfigError, match="Cannot load pipeline configuration of version 1.1.0"):
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml", strict_version_check=True)


@pytest.mark.unit
def test_load_yaml_no_components(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            pipelines:
            - name: my_pipeline
              nodes:
        """
        )
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
        assert "components" in str(e)


@pytest.mark.unit
def test_load_yaml_wrong_component(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: docstore
              type: ImaginaryDocumentStore
            pipelines:
            - name: my_pipeline
              nodes:
              - name: docstore
                inputs:
                - Query
        """
        )
    with pytest.raises(HaystackError) as e:
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
        assert "ImaginaryDocumentStore" in str(e)


@pytest.mark.unit
def test_load_yaml_custom_component(tmp_path):
    class CustomNode(MockNode):
        def __init__(self, param: int):
            super().__init__()
            self.param = param

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: custom_node
              type: CustomNode
              params:
                param: 1
            pipelines:
            - name: my_pipeline
              nodes:
              - name: custom_node
                inputs:
                - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    assert pipeline.get_node("custom_node").param == 1


@pytest.mark.unit
def test_load_yaml_custom_component_with_null_values(tmp_path):
    class CustomNode(MockNode):
        def __init__(self, param: Optional[str], lst_param: Optional[List[Any]], dict_param: Optional[Dict[str, Any]]):
            super().__init__()
            self.param = param
            self.lst_param = lst_param
            self.dict_param = dict_param

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: custom_node
              type: CustomNode
              params:
                param: null
                lst_param: null
                dict_param: null
            pipelines:
            - name: my_pipeline
              nodes:
              - name: custom_node
                inputs:
                - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    assert len(pipeline.graph.nodes) == 2
    assert pipeline.get_node("custom_node").param is None
    assert pipeline.get_node("custom_node").lst_param is None
    assert pipeline.get_node("custom_node").dict_param is None


@pytest.mark.unit
def test_load_yaml_custom_component_with_no_init(tmp_path):
    class CustomNode(MockNode):
        pass

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: custom_node
              type: CustomNode
            pipelines:
            - name: my_pipeline
              nodes:
              - name: custom_node
                inputs:
                - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    assert isinstance(pipeline.get_node("custom_node"), CustomNode)


@pytest.mark.unit
def test_load_yaml_custom_component_neednt_call_super(tmp_path):
    """This is a side-effect. Here for behavior documentation only"""

    class CustomNode(BaseComponent):
        outgoing_edges = 1

        def __init__(self, param: int):
            self.param = param

        def run(self, *a, **k):
            pass

        def run_batch(self, *a, **k):
            pass

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: custom_node
              type: CustomNode
              params:
                param: 1
            pipelines:
            - name: my_pipeline
              nodes:
              - name: custom_node
                inputs:
                - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    assert isinstance(pipeline.get_node("custom_node"), CustomNode)
    assert pipeline.get_node("custom_node").param == 1


@pytest.mark.unit
def test_load_yaml_custom_component_cant_be_abstract(tmp_path):
    class CustomNode(MockNode):
        @abstractmethod
        def abstract_method(self):
            pass

    assert inspect.isabstract(CustomNode)

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: custom_node
              type: CustomNode
            pipelines:
            - name: my_pipeline
              nodes:
              - name: custom_node
                inputs:
                - Query
        """
        )
    with pytest.raises(PipelineSchemaError, match="abstract"):
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")


@pytest.mark.unit
def test_load_yaml_custom_component_name_can_include_base(tmp_path):
    class BaseCustomNode(MockNode):
        def __init__(self):
            super().__init__()

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: custom_node
              type: BaseCustomNode
            pipelines:
            - name: my_pipeline
              nodes:
              - name: custom_node
                inputs:
                - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    assert isinstance(pipeline.get_node("custom_node"), BaseCustomNode)


@pytest.mark.unit
def test_load_yaml_custom_component_must_subclass_basecomponent(tmp_path):
    class SomeCustomNode:
        def run(self, *a, **k):
            pass

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: custom_node
              type: SomeCustomNode
              params:
                param: 1
            pipelines:
            - name: my_pipeline
              nodes:
              - name: custom_node
                inputs:
                - Query
        """
        )
    with pytest.raises(PipelineSchemaError, match="'SomeCustomNode' not found"):
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")


@pytest.mark.unit
def test_load_yaml_custom_component_referencing_other_node_in_init(tmp_path):
    class OtherNode(MockNode):
        def __init__(self, another_param: str):
            super().__init__()
            self.param = another_param

    class CustomNode(MockNode):
        def __init__(self, other_node: OtherNode):
            super().__init__()
            self.other_node = other_node

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: other_node
              type: OtherNode
              params:
                another_param: value
            - name: custom_node
              type: CustomNode
              params:
                other_node: other_node
            pipelines:
            - name: my_pipeline
              nodes:
              - name: custom_node
                inputs:
                - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    assert isinstance(pipeline.get_node("custom_node"), CustomNode)
    assert isinstance(pipeline.get_node("custom_node").other_node, OtherNode)
    assert pipeline.get_node("custom_node").name == "custom_node"
    assert pipeline.get_node("custom_node").other_node.name == "other_node"


@pytest.mark.unit
def test_load_yaml_custom_component_with_helper_class_in_init(tmp_path):
    """
    This test can work from the perspective of YAML schema validation:
    HelperClass is picked up correctly and everything gets loaded.

    However, for now we decide to disable this feature.
    See haystack/_json_schema.py for details.
    """

    @dataclass  # Makes this test class JSON serializable
    class HelperClass:
        def __init__(self, another_param: str):
            self.param = another_param

    class CustomNode(MockNode):
        def __init__(self, some_exotic_parameter: HelperClass = HelperClass(1)):
            super().__init__()
            self.some_exotic_parameter = some_exotic_parameter

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: custom_node
              type: CustomNode
            pipelines:
            - name: my_pipeline
              nodes:
              - name: custom_node
                inputs:
                - Query
        """
        )
    with pytest.raises(PipelineSchemaError, match="takes object instances as parameters in its __init__ function"):
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")


@pytest.mark.unit
def test_load_yaml_custom_component_with_helper_class_in_yaml(tmp_path):
    """
    This test can work from the perspective of YAML schema validation:
    HelperClass is picked up correctly and everything gets loaded.

    However, for now we decide to disable this feature.
    See haystack/_json_schema.py for details.
    """

    class HelperClass:
        def __init__(self, another_param: str):
            self.param = another_param

    class CustomNode(MockNode):
        def __init__(self, some_exotic_parameter: HelperClass):
            super().__init__()
            self.some_exotic_parameter = some_exotic_parameter

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: custom_node
              type: CustomNode
              params:
                some_exotic_parameter: HelperClass("hello")
            pipelines:
            - name: my_pipeline
              nodes:
              - name: custom_node
                inputs:
                - Query
        """
        )
    pipe = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    assert pipe.get_node("custom_node").some_exotic_parameter == 'HelperClass("hello")'


@pytest.mark.unit
def test_load_yaml_custom_component_with_enum_in_init(tmp_path):
    """
    This test can work from the perspective of YAML schema validation:
    Flags is picked up correctly and everything gets loaded.

    However, for now we decide to disable this feature.
    See haystack/_json_schema.py for details.
    """

    class Flags(Enum):
        FIRST_VALUE = 1
        SECOND_VALUE = 2

    class CustomNode(MockNode):
        def __init__(self, some_exotic_parameter: Flags = None):
            super().__init__()
            self.some_exotic_parameter = some_exotic_parameter

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: custom_node
              type: CustomNode
            pipelines:
            - name: my_pipeline
              nodes:
              - name: custom_node
                inputs:
                - Query
        """
        )
    with pytest.raises(PipelineSchemaError, match="takes object instances as parameters in its __init__ function"):
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")


@pytest.mark.unit
def test_load_yaml_custom_component_with_enum_in_yaml(tmp_path):
    """
    This test can work from the perspective of YAML schema validation:
    Flags is picked up correctly and everything gets loaded.

    However, for now we decide to disable this feature.
    See haystack/_json_schema.py for details.
    """

    class Flags(Enum):
        FIRST_VALUE = 1
        SECOND_VALUE = 2

    class CustomNode(MockNode):
        def __init__(self, some_exotic_parameter: Flags):
            super().__init__()
            self.some_exotic_parameter = some_exotic_parameter

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: custom_node
              type: CustomNode
              params:
                some_exotic_parameter: Flags.SECOND_VALUE
            pipelines:
            - name: my_pipeline
              nodes:
              - name: custom_node
                inputs:
                - Query
        """
        )
    with pytest.raises(PipelineSchemaError, match="takes object instances as parameters in its __init__ function"):
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")


@pytest.mark.unit
def test_load_yaml_custom_component_with_external_constant(tmp_path):
    """
    This is a potential pitfall. The code should work as described here.
    """

    class AnotherClass:
        CLASS_CONSTANT = "str"

    class CustomNode(MockNode):
        def __init__(self, some_exotic_parameter: str):
            super().__init__()
            self.some_exotic_parameter = some_exotic_parameter

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: custom_node
              type: CustomNode
              params:
                some_exotic_parameter: AnotherClass.CLASS_CONSTANT  # Will *NOT* be resolved
            pipelines:
            - name: my_pipeline
              nodes:
              - name: custom_node
                inputs:
                - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    node = pipeline.get_node("custom_node")
    assert node.some_exotic_parameter == "AnotherClass.CLASS_CONSTANT"


@pytest.mark.unit
def test_load_yaml_custom_component_with_superclass(tmp_path):
    class BaseCustomNode(MockNode):
        def __init__(self):
            super().__init__()

    class CustomNode(BaseCustomNode):
        def __init__(self, some_exotic_parameter: str):
            super().__init__()
            self.some_exotic_parameter = some_exotic_parameter

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: custom_node
              type: CustomNode
              params:
                some_exotic_parameter: value
            pipelines:
            - name: my_pipeline
              nodes:
              - name: custom_node
                inputs:
                - Query
        """
        )
    Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")


@pytest.mark.unit
def test_load_yaml_custom_component_with_variadic_args(tmp_path):
    class BaseCustomNode(MockNode):
        def __init__(self, base_parameter: int):
            super().__init__()
            self.base_parameter = base_parameter

    class CustomNode(BaseCustomNode):
        def __init__(self, some_parameter: str, *args):
            super().__init__(*args)
            self.some_parameter = some_parameter

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: custom_node
              type: CustomNode
              params:
                base_parameter: 1
                some_parameter: value
            pipelines:
            - name: my_pipeline
              nodes:
              - name: custom_node
                inputs:
                - Query
        """
        )
    with pytest.raises(PipelineSchemaError, match="variadic"):
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")


@pytest.mark.unit
def test_load_yaml_custom_component_with_variadic_kwargs(tmp_path):
    class BaseCustomNode(MockNode):
        def __init__(self, base_parameter: int):
            super().__init__()
            self.base_parameter = base_parameter

    class CustomNode(BaseCustomNode):
        def __init__(self, some_parameter: str, **kwargs):
            super().__init__(**kwargs)
            self.some_parameter = some_parameter

    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: custom_node
              type: CustomNode
              params:
                base_parameter: 1
                some_parameter: value
            pipelines:
            - name: my_pipeline
              nodes:
              - name: custom_node
                inputs:
                - Query
        """
        )
    with pytest.raises(PipelineSchemaError, match="variadic"):
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")


@pytest.mark.unit
def test_load_yaml_no_pipelines(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: docstore
              type: MockDocumentStore
            pipelines:
        """
        )
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
        assert "pipeline" in str(e)


@pytest.mark.unit
def test_load_yaml_invalid_pipeline_name(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: docstore
              type: MockDocumentStore
            pipelines:
            - name: my_pipeline
              nodes:
              - name: docstore
                inputs:
                - Query
        """
        )
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml", pipeline_name="invalid")
        assert "invalid" in str(e) and "pipeline" in str(e)


@pytest.mark.unit
def test_load_yaml_pipeline_with_wrong_nodes(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: docstore
              type: MockDocumentStore
            pipelines:
            - name: my_pipeline
              nodes:
              - name: not_existing_node
                inputs:
                - Query
        """
        )
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
        assert "not_existing_node" in str(e)


@pytest.mark.unit
def test_load_yaml_pipeline_not_acyclic_graph(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: retriever
              type: MockRetriever
            - name: reader
              type: MockRetriever
            pipelines:
            - name: my_pipeline
              nodes:
              - name: retriever
                inputs:
                - reader
              - name: reader
                inputs:
                - retriever
        """
        )
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
        assert "reader" in str(e) or "retriever" in str(e)
        assert "loop" in str(e)


@pytest.mark.unit
def test_load_yaml_wrong_root(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: retriever
              type: MockRetriever
            pipelines:
            - name: my_pipeline
              nodes:
              - name: retriever
                inputs:
                - Nothing
        """
        )
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
        assert "Nothing" in str(e)
        assert "root" in str(e).lower()


@pytest.mark.unit
def test_load_yaml_two_roots_invalid(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
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
                - File
        """
        )
    with pytest.raises(PipelineConfigError) as e:
        Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    assert "File" in str(e) or "Query" in str(e)


@pytest.mark.unit
def test_load_yaml_two_roots_valid(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
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
        """
        )
    Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")


@pytest.mark.unit
def test_load_yaml_two_roots_in_separate_pipelines(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: node_1
              type: MockNode
            - name: node_2
              type: MockNode
            pipelines:
            - name: pipeline_1
              nodes:
              - name: node_1
                inputs:
                - Query
              - name: node_2
                inputs:
                - Query
            - name: pipeline_2
              nodes:
              - name: node_1
                inputs:
                - File
              - name: node_2
                inputs:
                - File
        """
        )
    Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml", pipeline_name="pipeline_1")
    Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml", pipeline_name="pipeline_2")


@pytest.mark.unit
def test_load_yaml_disconnected_component(tmp_path):
    with open(tmp_path / "tmp_config.yml", "w") as tmp_file:
        tmp_file.write(
            f"""
            version: ignore
            components:
            - name: docstore
              type: MockDocumentStore
            - name: retriever
              type: MockRetriever
            pipelines:
            - name: query
              nodes:
              - name: docstore
                inputs:
                - Query
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    assert len(pipeline.graph.nodes) == 2
    assert isinstance(pipeline.get_document_store(), MockDocumentStore)
    assert not pipeline.get_node("retriever")


@pytest.mark.unit
def test_load_yaml_unusual_chars_in_values(tmp_path):
    class DummyNode(BaseComponent):
        outgoing_edges = 1

        def __init__(self, space_param, non_alphanumeric_param):
            super().__init__()
            self.space_param = space_param
            self.non_alphanumeric_param = non_alphanumeric_param

        def run(self):
            raise NotImplementedError

        def run_batch(self):
            raise NotImplementedError

    with open(tmp_path / "tmp_config.yml", "w", encoding="utf-8") as tmp_file:
        tmp_file.write(
            f"""
            version: '1.9.0'

            components:
                - name: DummyNode
                  type: DummyNode
                  params:
                    space_param: with space
                    non_alphanumeric_param: \[ümlaut\]

            pipelines:
                - name: indexing
                  nodes:
                    - name: DummyNode
                      inputs: [File]
        """
        )
    pipeline = Pipeline.load_from_yaml(path=tmp_path / "tmp_config.yml")
    assert pipeline.components["DummyNode"].space_param == "with space"
    assert pipeline.components["DummyNode"].non_alphanumeric_param == "\\[ümlaut\\]"


@pytest.mark.unit
def test_save_yaml(tmp_path):
    pipeline = Pipeline()
    pipeline.add_node(MockRetriever(), name="retriever", inputs=["Query"])
    pipeline.save_to_yaml(tmp_path / "saved_pipeline.yml")

    with open(tmp_path / "saved_pipeline.yml", "r") as saved_yaml:
        content = saved_yaml.read()

        assert content.count("retriever") == 2
        assert "MockRetriever" in content
        assert "Query" in content
        assert f"version: {haystack.__version__}" in content


@pytest.mark.unit
def test_save_yaml_overwrite(tmp_path):
    pipeline = Pipeline()
    retriever = MockRetriever()
    pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])

    with open(tmp_path / "saved_pipeline.yml", "w") as _:
        pass

    pipeline.save_to_yaml(tmp_path / "saved_pipeline.yml")

    with open(tmp_path / "saved_pipeline.yml", "r") as saved_yaml:
        content = saved_yaml.read()
        assert content != ""


@pytest.mark.unit
@pytest.mark.parametrize("pipeline_file", ["ray.simple.haystack-pipeline.yml", "ray.advanced.haystack-pipeline.yml"])
def test_load_yaml_ray_args_in_pipeline(samples_path, pipeline_file):
    with pytest.raises(PipelineConfigError) as e:
        pipeline = Pipeline.load_from_yaml(
            samples_path / "pipeline" / pipeline_file, pipeline_name="ray_query_pipeline"
        )
