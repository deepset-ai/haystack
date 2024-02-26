import sys
import tempfile

import pytest

from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.generators import HuggingFaceTGIGenerator
from haystack.core.errors import PipelineValidationError
from haystack.templates.pipeline import PipelineTemplate, PredefinedPipeline


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


class TestPipelineTemplate:
    def test_from_str(self):
        with pytest.raises(ValueError):
            PipelineTemplate.from_str("{{ invalid template }")

        assert PipelineTemplate.from_str("{{ valid_template }}").template_content == "{{ valid_template }}"

    def test_from_file(self, random_valid_template):
        with pytest.raises(FileNotFoundError):
            PipelineTemplate.from_file("invalid/path")

        with tempfile.NamedTemporaryFile(mode="w") as fp:
            fp.write(random_valid_template)
            fp.seek(0)
            assert PipelineTemplate.from_file(fp.name).template_content == random_valid_template

    def test_from_predefined(self):
        tpl = PipelineTemplate.from_predefined(PredefinedPipeline.INDEXING)
        assert len(tpl.template_content)

    #  Raises PipelineValidationError when attempting to override a non-existent component
    def test_override_nonexistent_component(self):
        with pytest.raises(PipelineValidationError):
            PipelineTemplate.from_predefined(PredefinedPipeline.INDEXING).override(
                "nonexistent_component", SentenceTransformersDocumentEmbedder()
            )

    #  Building a pipeline directly using all default components specified in a predefined or custom template.
    def test_build_pipeline_with_default_components(self):
        pipeline = PipelineTemplate.from_predefined(PredefinedPipeline.INDEXING).build()
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
        pt = PipelineTemplate.from_predefined(PredefinedPipeline.INDEXING)

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
        pipe = (
            PipelineTemplate.from_predefined(PredefinedPipeline.QA)
            .override("generator", HuggingFaceTGIGenerator())
            .build()
        )
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
