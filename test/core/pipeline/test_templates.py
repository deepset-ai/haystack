# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import tempfile
from unittest import mock

import pytest

from haystack import Pipeline
from haystack.core.pipeline.template import PipelineTemplate, PredefinedPipeline


@pytest.fixture
def random_valid_template():
    template = """
components:
  generator:
    {{ generator | indent }}

  prompt_builder:
    {{ prompt_builder | indent }}

connections:
  - receiver: generator.prompt
    sender: prompt_builder.prompt
"""
    return template


class TestPipelineTemplate:
    def test_from_str(self):
        with pytest.raises(ValueError):
            PipelineTemplate("{{ invalid template }")

        assert PipelineTemplate("{{ valid_template }}").template_content == "{{ valid_template }}"

    def test_from_file(self, random_valid_template):
        with pytest.raises(FileNotFoundError):
            PipelineTemplate.from_file("invalid/path")

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as fp:
            fp.write(random_valid_template)
            fp.seek(0)
            assert PipelineTemplate.from_file(fp.name).template_content == random_valid_template

    def test_from_predefined(self):
        tpl = PipelineTemplate.from_predefined(PredefinedPipeline.INDEXING)
        assert len(tpl.template_content)

    #  Building a pipeline directly using all default components specified in a predefined or custom template.
    def test_build_pipeline_with_default_components(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "fake_key")
        rendered = PipelineTemplate.from_predefined(PredefinedPipeline.INDEXING).render()
        pipeline = Pipeline.loads(rendered)

        # pipeline has components
        assert pipeline.get_component("cleaner")
        assert pipeline.get_component("writer")
        assert pipeline.get_component("embedder")

        # pipeline should have inputs and outputs
        assert len(pipeline.inputs()) > 0
        assert len(pipeline.outputs()) > 0
