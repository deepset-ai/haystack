from haystack.components.builders import PromptBuilder
from haystack.core.errors import PipelineValidationError

from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.generators import HuggingFaceTGIGenerator
from haystack.templates.pipelines import PipelineTemplate, PredefinedTemplate

import pytest


class TestPipelineTemplate:
    #  Raises ValueError if the specified `pipeline_template` is invalid or if no Jinja2 template syntax detected.
    def test_invalid_pipeline_template(self):
        with pytest.raises(ValueError):
            PipelineTemplate.from_string("invalid_template")

    #  Raises PipelineValidationError when attempting to override a non-existent component
    def test_override_nonexistent_component(self):
        pipeline = PipelineTemplate.from_predefined(PredefinedTemplate.INDEXING)

        with pytest.raises(PipelineValidationError):
            pipeline.override("nonexistent_component", SentenceTransformersDocumentEmbedder())

    #  If pipeline_template is not provided.
    def test_missing_pipeline_template(self):
        with pytest.raises(ValueError):
            PipelineTemplate.from_string("")

    #  Building a pipeline directly using all default components specified in a predefined or custom template.
    def test_build_pipeline_with_default_components(self):
        pipeline = PipelineTemplate.from_predefined(PredefinedTemplate.INDEXING).build()
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
        pt = PipelineTemplate.from_predefined(PredefinedTemplate.INDEXING)

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
            PipelineTemplate.from_predefined(PredefinedTemplate.QA)
            .override("generator", HuggingFaceTGIGenerator())
            .build()
        )
        assert isinstance(pipe, Pipeline)
        assert pipe.get_component("generator")
        assert isinstance(pipe.get_component("generator"), HuggingFaceTGIGenerator)

    #  Building a pipeline with a custom template that uses Jinja2 syntax to specify components and their connections
    @pytest.mark.integration
    def test_building_pipeline_with_direct_template(self):
        template = """components:
  generator: {{ generator | tojson }}
  prompt_builder: {{prompt_builder}}

connections:
- receiver: generator.prompt
  sender: prompt_builder.prompt
max_loops_allowed: 2
metadata: {}
"""
        pt = PipelineTemplate.from_string(template)
        pt.override("generator", HuggingFaceTGIGenerator())
        pt.override("prompt_builder", PromptBuilder("Some fake prompt"))
        pipe = pt.build()

        assert isinstance(pipe, Pipeline)
        assert pipe.get_component("generator")
        assert isinstance(pipe.get_component("generator"), HuggingFaceTGIGenerator)
        assert pipe.get_component("prompt_builder")
        assert isinstance(pipe.get_component("prompt_builder"), PromptBuilder)
