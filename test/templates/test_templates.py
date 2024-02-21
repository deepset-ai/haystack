import sys
import tempfile

import pytest

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.generators import HuggingFaceTGIGenerator
from haystack.core.errors import PipelineValidationError
from haystack.templates.pipelines import PipelineTemplate
from haystack.templates.source import _templateSource, PipelineType


@pytest.fixture
def random_valid_template():
    template = """components:
  generator: {{ generator | tojson }}
  prompt_builder: {{prompt_builder}}

connections:
- receiver: generator.prompt
  sender: prompt_builder.prompt
max_loops_allowed: 2
metadata: {}
"""
    return template


class TestTemplateSource:
    #  If the provided template does not contain Jinja2 syntax.
    def test_from_str(self):
        with pytest.raises(ValueError):
            _templateSource.from_str("invalid_template")

    #  If the provided template contains Jinja2 syntax.
    def test_from_str_valid(self):
        ts = _templateSource.from_str("{{ valid_template }}")
        assert ts.template == "{{ valid_template }}"

    #  If the provided file path does not exist.
    def test_from_file_invalid_path(self):
        with pytest.raises(FileNotFoundError):
            _templateSource.from_file("invalid_path")

    #  If the provided file path exists.
    @pytest.mark.skipif(sys.platform == "win32", reason="Fails on Windows CI with permission denied")
    def test_from_file_valid_path(self, random_valid_template):
        temp_file = tempfile.NamedTemporaryFile(mode="w")
        temp_file.write(random_valid_template)
        temp_file.flush()
        ts = _templateSource.from_file(temp_file.name)
        assert ts.template == random_valid_template

    # Use predefined template
    def test_from_predefined_invalid_template(self):
        ts = _templateSource.from_predefined(PipelineType.INDEXING)
        assert len(ts.template) > 0


class TestPipelineTemplate:
    #  Raises PipelineValidationError when attempting to override a non-existent component
    def test_override_nonexistent_component(self):
        with pytest.raises(PipelineValidationError):
            PipelineTemplate(PipelineType.INDEXING).override(
                "nonexistent_component", SentenceTransformersDocumentEmbedder()
            )

    #  Building a pipeline directly using all default components specified in a predefined or custom template.
    def test_build_pipeline_with_default_components(self):
        pipeline = PipelineTemplate(PipelineType.INDEXING).build()
        assert isinstance(pipeline, Pipeline)

        # pipeline has components
        assert pipeline.get_component("cleaner")
        assert pipeline.get_component("writer")
        assert pipeline.get_component("embedder")

        # pipeline should have inputs and outputs
        assert len(pipeline.inputs()) > 0
        assert len(pipeline.outputs()) > 0

    # Customizing pipelines by overriding default components with custom component settings
    def test_customize_pipeline_with_overrides(self):
        pt = PipelineTemplate(PipelineType.INDEXING)

        pt.override("embedder", SentenceTransformersDocumentEmbedder(progress_bar=True, batch_size=64))
        pipe = pt.build()

        assert isinstance(pipe, Pipeline)
        assert pipe.get_component("embedder")
        embedder: SentenceTransformersDocumentEmbedder = pipe.get_component("embedder")
        embedder_dict = embedder.to_dict()
        assert embedder_dict["init_parameters"]["progress_bar"]
        assert embedder_dict["init_parameters"]["batch_size"] == 64

    #  Overrides a generator component specified in the pipeline template with a completely different generator
    @pytest.mark.integration
    def test_override_component(self):
        # integration because we'll fetch the tokenizer
        pipe = PipelineTemplate(PipelineType.QA).override("generator", HuggingFaceTGIGenerator()).build()
        assert isinstance(pipe, Pipeline)
        assert pipe.get_component("generator")
        assert isinstance(pipe.get_component("generator"), HuggingFaceTGIGenerator)

    #  Building a pipeline with a custom template that uses Jinja2 syntax to specify components and their connections
    # @pytest.mark.integration
    # def test_building_pipeline_with_direct_template(self, random_valid_template):
    #     pt = PipelineTemplate(TemplateSource.from_str(random_valid_template))
    #     pt.override("generator", HuggingFaceTGIGenerator())
    #     pt.override("prompt_builder", PromptBuilder("Some fake prompt"))
    #     pipe = pt.build()

    #     assert isinstance(pipe, Pipeline)
    #     assert pipe.get_component("generator")
    #     assert isinstance(pipe.get_component("generator"), HuggingFaceTGIGenerator)
    #     assert pipe.get_component("prompt_builder")
    #     assert isinstance(pipe.get_component("prompt_builder"), PromptBuilder)
