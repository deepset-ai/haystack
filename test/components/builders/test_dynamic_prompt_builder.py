from typing import List

import pytest
from jinja2 import TemplateSyntaxError

from haystack import component, Document, Pipeline
from haystack.components.builders import DynamicPromptBuilder


class TestDynamicPromptBuilder:
    def test_initialization(self):
        runtime_variables = ["var1", "var2"]
        builder = DynamicPromptBuilder(runtime_variables)
        assert builder.runtime_variables == runtime_variables

        # regardless of the chat mode
        # we have inputs that contain: prompt_source, template_variables + runtime_variables
        expected_keys = set(runtime_variables + ["prompt_source", "template_variables"])
        assert set(builder.__canals_input__.keys()) == expected_keys

        # response is always prompt regardless of chat mode
        assert set(builder.__canals_output__.keys()) == {"prompt"}

        # prompt_source is a list of ChatMessage or a string
        assert builder.__canals_input__["prompt_source"].type == str

        # output is always prompt, but the type is different depending on the chat mode
        assert builder.__canals_output__["prompt"].type == str

    def test_processing_a_simple_template_with_provided_variables(self):
        runtime_variables = ["var1", "var2", "var3"]

        builder = DynamicPromptBuilder(runtime_variables)

        template = "Hello, {{ name }}!"
        template_variables = {"name": "John"}
        expected_result = {"prompt": "Hello, John!"}

        assert builder.run(template, template_variables) == expected_result

    def test_processing_a_simple_template_with_invalid_template(self):
        runtime_variables = ["var1", "var2", "var3"]
        builder = DynamicPromptBuilder(runtime_variables)

        template = "Hello, {{ name }!"
        template_variables = {"name": "John"}
        with pytest.raises(TemplateSyntaxError):
            builder.run(template, template_variables)

    def test_processing_a_simple_template_with_missing_variables(self):
        runtime_variables = ["var1", "var2", "var3"]
        builder = DynamicPromptBuilder(runtime_variables)

        with pytest.raises(ValueError):
            builder.run("Hello, {{ name }}!", {})

    def test_missing_template_variables(self):
        prompt_builder = DynamicPromptBuilder(runtime_variables=["documents"])

        # missing template variable city
        with pytest.raises(ValueError):
            prompt_builder._validate_template("Hello, I'm {{ name }}, and I live in {{ city }}.", {"name"})

        # missing template variable name
        with pytest.raises(ValueError):
            prompt_builder._validate_template("Hello, I'm {{ name }}, and I live in {{ city }}.", {"city"})

        # completely unknown template variable
        with pytest.raises(ValueError):
            prompt_builder._validate_template("Hello, I'm {{ name }}, and I live in {{ city }}.", {"age"})

    def test_provided_template_variables(self):
        prompt_builder = DynamicPromptBuilder(runtime_variables=["documents"])

        # both variables are provided
        prompt_builder._validate_template("Hello, I'm {{ name }}, and I live in {{ city }}.", {"name", "city"})

        # provided variables are a superset of the required variables
        prompt_builder._validate_template("Hello, I'm {{ name }}, and I live in {{ city }}.", {"name", "city", "age"})

    def test_example_in_pipeline(self):
        prompt_builder = DynamicPromptBuilder(runtime_variables=["documents"])

        @component
        class DocumentProducer:
            @component.output_types(documents=List[Document])
            def run(self, doc_input: str):
                return {"documents": [Document(content=doc_input)]}

        pipe = Pipeline()
        pipe.add_component("doc_producer", DocumentProducer())
        pipe.add_component("prompt_builder", prompt_builder)
        pipe.connect("doc_producer.documents", "prompt_builder.documents")

        template = "Here is the document: {{documents[0].content}} \\n Answer: {{query}}"
        result = pipe.run(
            data={
                "doc_producer": {"doc_input": "Hello world, I live in Berlin"},
                "prompt_builder": {
                    "prompt_source": template,
                    "template_variables": {"query": "Where does the speaker live?"},
                },
            }
        )

        assert result == {
            "prompt_builder": {
                "prompt": "Here is the document: Hello world, I live in Berlin \\n Answer: Where does the speaker live?"
            }
        }
