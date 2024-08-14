# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, List, Optional
from jinja2 import TemplateSyntaxError
import pytest
from unittest.mock import patch, MagicMock
import datetime

from haystack.components.builders.prompt_builder import PromptBuilder
from haystack import component
from haystack.core.pipeline.pipeline import Pipeline
from haystack.dataclasses.document import Document


class TestPromptBuilder:
    def test_init(self):
        builder = PromptBuilder(template="This is a {{ variable }}")
        assert builder.template is not None
        assert builder.required_variables == []
        assert builder._template_string == "This is a {{ variable }}"
        assert builder._variables is None
        assert builder._required_variables is None

        # we have inputs that contain: template, template_variables + inferred variables
        inputs = builder.__haystack_input__._sockets_dict
        assert set(inputs.keys()) == {"template", "template_variables", "variable"}
        assert inputs["template"].type == Optional[str]
        assert inputs["template_variables"].type == Optional[Dict[str, Any]]
        assert inputs["variable"].type == Any

        # response is always prompt
        outputs = builder.__haystack_output__._sockets_dict
        assert set(outputs.keys()) == {"prompt"}
        assert outputs["prompt"].type == str

    def test_init_with_required_variables(self):
        builder = PromptBuilder(template="This is a {{ variable }}", required_variables=["variable"])
        assert builder.template is not None
        assert builder.required_variables == ["variable"]
        assert builder._template_string == "This is a {{ variable }}"
        assert builder._variables is None
        assert builder._required_variables == ["variable"]

        # we have inputs that contain: template, template_variables + inferred variables
        inputs = builder.__haystack_input__._sockets_dict
        assert set(inputs.keys()) == {"template", "template_variables", "variable"}
        assert inputs["template"].type == Optional[str]
        assert inputs["template_variables"].type == Optional[Dict[str, Any]]
        assert inputs["variable"].type == Any

        # response is always prompt
        outputs = builder.__haystack_output__._sockets_dict
        assert set(outputs.keys()) == {"prompt"}
        assert outputs["prompt"].type == str

    def test_init_with_custom_variables(self):
        variables = ["var1", "var2", "var3"]
        template = "Hello, {{ var1 }}, {{ var2 }}!"
        builder = PromptBuilder(template=template, variables=variables)
        assert builder.template is not None
        assert builder.required_variables == []
        assert builder._variables == variables
        assert builder._template_string == "Hello, {{ var1 }}, {{ var2 }}!"
        assert builder._required_variables is None

        # we have inputs that contain: template, template_variables + variables
        inputs = builder.__haystack_input__._sockets_dict
        assert set(inputs.keys()) == {"template", "template_variables", "var1", "var2", "var3"}
        assert inputs["template"].type == Optional[str]
        assert inputs["template_variables"].type == Optional[Dict[str, Any]]
        assert inputs["var1"].type == Any
        assert inputs["var2"].type == Any
        assert inputs["var3"].type == Any

        # response is always prompt
        outputs = builder.__haystack_output__._sockets_dict
        assert set(outputs.keys()) == {"prompt"}
        assert outputs["prompt"].type == str

    def test_to_dict(self):
        builder = PromptBuilder(
            template="This is a {{ variable }}", variables=["var1", "var2"], required_variables=["var1", "var3"]
        )
        res = builder.to_dict()
        assert res == {
            "type": "haystack.components.builders.prompt_builder.PromptBuilder",
            "init_parameters": {
                "template": "This is a {{ variable }}",
                "variables": ["var1", "var2"],
                "required_variables": ["var1", "var3"],
            },
        }

    def test_to_dict_without_optional_params(self):
        builder = PromptBuilder(template="This is a {{ variable }}")
        res = builder.to_dict()
        assert res == {
            "type": "haystack.components.builders.prompt_builder.PromptBuilder",
            "init_parameters": {"template": "This is a {{ variable }}", "variables": None, "required_variables": None},
        }

    def test_run(self):
        builder = PromptBuilder(template="This is a {{ variable }}")
        res = builder.run(variable="test")
        assert res == {"prompt": "This is a test"}

    def test_run_template_variable(self):
        builder = PromptBuilder(template="This is a {{ variable }}")
        res = builder.run(template_variables={"variable": "test"})
        assert res == {"prompt": "This is a test"}

    def test_run_template_variable_overrides_variable(self):
        builder = PromptBuilder(template="This is a {{ variable }}")
        res = builder.run(template_variables={"variable": "test_from_template_var"}, variable="test")
        assert res == {"prompt": "This is a test_from_template_var"}

    def test_run_without_input(self):
        builder = PromptBuilder(template="This is a template without input")
        res = builder.run()
        assert res == {"prompt": "This is a template without input"}

    def test_run_with_missing_input(self):
        builder = PromptBuilder(template="This is a {{ variable }}")
        res = builder.run()
        assert res == {"prompt": "This is a "}

    def test_run_with_missing_required_input(self):
        builder = PromptBuilder(template="This is a {{ foo }}, not a {{ bar }}", required_variables=["foo", "bar"])
        with pytest.raises(ValueError, match="foo"):
            builder.run(bar="bar")
        with pytest.raises(ValueError, match="bar"):
            builder.run(foo="foo")
        with pytest.raises(ValueError, match="foo, bar"):
            builder.run()

    def test_run_with_variables(self):
        variables = ["var1", "var2", "var3"]
        template = "Hello, {{ name }}! {{ var1 }}"

        builder = PromptBuilder(template=template, variables=variables)

        template_variables = {"name": "John"}
        expected_result = {"prompt": "Hello, John! How are you?"}

        assert builder.run(template_variables=template_variables, var1="How are you?") == expected_result

    def test_run_overwriting_default_template(self):
        default_template = "Hello, {{ name }}!"

        builder = PromptBuilder(template=default_template)

        template = "Hello, {{ var1 }}{{ name }}!"
        expected_result = {"prompt": "Hello, John!"}

        assert builder.run(template, name="John") == expected_result

    def test_run_overwriting_default_template_with_template_variables(self):
        default_template = "Hello, {{ name }}!"

        builder = PromptBuilder(template=default_template)

        template = "Hello, {{ var1 }} {{ name }}!"
        template_variables = {"var1": "Big"}
        expected_result = {"prompt": "Hello, Big John!"}

        assert builder.run(template, template_variables, name="John") == expected_result

    def test_run_overwriting_default_template_with_variables(self):
        variables = ["var1", "var2", "name"]
        default_template = "Hello, {{ name }}!"

        builder = PromptBuilder(template=default_template, variables=variables)

        template = "Hello, {{ var1 }} {{ name }}!"
        expected_result = {"prompt": "Hello, Big John!"}

        assert builder.run(template, name="John", var1="Big") == expected_result

    def test_run_with_invalid_template(self):
        builder = PromptBuilder(template="Hello, {{ name }}!")

        template = "Hello, {{ name }!"
        template_variables = {"name": "John"}
        with pytest.raises(TemplateSyntaxError):
            builder.run(template, template_variables)

    def test_init_with_invalid_template(self):
        template = "Hello, {{ name }!"
        with pytest.raises(TemplateSyntaxError):
            PromptBuilder(template)

    def test_provided_template_variables(self):
        prompt_builder = PromptBuilder(template="", variables=["documents"], required_variables=["city"])

        # both variables are provided
        prompt_builder._validate_variables({"name", "city"})

        # provided variables are a superset of the required variables
        prompt_builder._validate_variables({"name", "city", "age"})

        with pytest.raises(ValueError):
            prompt_builder._validate_variables({"name"})

    def test_example_in_pipeline(self):
        default_template = "Here is the document: {{documents[0].content}} \\n Answer: {{query}}"
        prompt_builder = PromptBuilder(template=default_template, variables=["documents"])

        @component
        class DocumentProducer:
            @component.output_types(documents=List[Document])
            def run(self, doc_input: str):
                return {"documents": [Document(content=doc_input)]}

        pipe = Pipeline()
        pipe.add_component("doc_producer", DocumentProducer())
        pipe.add_component("prompt_builder", prompt_builder)
        pipe.connect("doc_producer.documents", "prompt_builder.documents")

        template = "Here is the document: {{documents[0].content}} \n Query: {{query}}"
        result = pipe.run(
            data={
                "doc_producer": {"doc_input": "Hello world, I live in Berlin"},
                "prompt_builder": {
                    "template": template,
                    "template_variables": {"query": "Where does the speaker live?"},
                },
            }
        )

        assert result == {
            "prompt_builder": {
                "prompt": "Here is the document: Hello world, I live in Berlin \n Query: Where does the speaker live?"
            }
        }

    def test_example_in_pipeline_simple(self):
        default_template = "This is the default prompt:\n Query: {{query}}"
        prompt_builder = PromptBuilder(template=default_template)

        pipe = Pipeline()
        pipe.add_component("prompt_builder", prompt_builder)

        # using the default prompt
        result = pipe.run(data={"query": "Where does the speaker live?"})
        expected_default = {
            "prompt_builder": {"prompt": "This is the default prompt:\n Query: Where does the speaker live?"}
        }
        assert result == expected_default

        # using the dynamic prompt
        result = pipe.run(
            data={"query": "Where does the speaker live?", "template": "This is the dynamic prompt:\n Query: {{query}}"}
        )
        expected_dynamic = {
            "prompt_builder": {"prompt": "This is the dynamic prompt:\n Query: Where does the speaker live?"}
        }
        assert result == expected_dynamic

    @patch("haystack.components.builders.prompt_builder.datetime")  # Mock datetime to control the current date and time
    def test_template_with_default_date_format(self, mock_datetime: MagicMock) -> None:
        mock_datetime.datetime.now.return_value = datetime.datetime(
            2024, 8, 14, 12, 30, 45, 123456, tzinfo=datetime.timezone.utc
        )

        template = "Today's date is {{ utc_now() }}"
        builder = PromptBuilder(template=template)

        result = builder.run()

        expected_prompt = "Today's date is 2024-08-14 12:30:45.123456"
        assert result["prompt"] == expected_prompt

    @patch("haystack.components.builders.prompt_builder.datetime")
    def test_template_with_custom_date_format(self, mock_datetime: MagicMock) -> None:
        mock_datetime.datetime.now.return_value = datetime.datetime(
            2024, 8, 14, 12, 30, 45, 123456, tzinfo=datetime.timezone.utc
        )

        template = "The date today is {{ utc_now('%Y/%m/%d') }}"
        builder = PromptBuilder(template=template)

        result = builder.run()

        expected_prompt = "The date today is 2024/08/14"
        assert result["prompt"] == expected_prompt

    @patch("haystack.components.builders.prompt_builder.datetime")
    def test_template_with_time_only_format(self, mock_datetime: MagicMock) -> None:
        mock_datetime.datetime.now.return_value = datetime.datetime(
            2024, 8, 14, 12, 30, 45, 123456, tzinfo=datetime.timezone.utc
        )

        template = "Current time is {{ utc_now('%H:%M:%S') }}"
        builder = PromptBuilder(template=template)

        result = builder.run()

        expected_prompt = "Current time is 12:30:45"
        assert result["prompt"] == expected_prompt

    @patch("haystack.components.builders.prompt_builder.datetime")
    def test_template_with_multiple_date_formats(self, mock_datetime: MagicMock) -> None:
        mock_datetime.datetime.now.return_value = datetime.datetime(
            2024, 8, 14, 12, 30, 45, 123456, tzinfo=datetime.timezone.utc
        )

        template = "Today is {{ utc_now('%Y-%m-%d') }}, and the current time is {{ utc_now('%H:%M') }}"
        builder = PromptBuilder(template=template)

        result = builder.run()

        expected_prompt = "Today is 2024-08-14, and the current time is 12:30"
        assert result["prompt"] == expected_prompt

    def test_template_without_utc_now(self):
        template = "Hello, this is a static template."
        builder = PromptBuilder(template=template)

        result = builder.run()

        expected_prompt = "Hello, this is a static template."
        assert result["prompt"] == expected_prompt

    def test_utc_now_empty_format(self):
        with pytest.raises(ValueError):
            template = "Hello, this is an empty date: {{ utc_now('') }}"
            builder = PromptBuilder(template=template)

            builder.run()

    def test_utc_now_invalid_format(self):
        with pytest.raises(ValueError):
            template = "Hello, this is an invalid date: {{ utc_now('%Q-%W-%R') }}"
            builder = PromptBuilder(template=template)

            builder.run()

    def test_utc_now_none(self):
        with pytest.raises(TypeError):
            template = "Hello, this is an invalid date: {{ utc_now(None) }}"
            builder = PromptBuilder(template=template)

            builder.run()

    def test_current_typeerror(self):
        with pytest.raises(TypeError):
            template = "Hello, this is an invalid date: {{ utc_now(10) }}"
            builder = PromptBuilder(template=template)

            builder.run()
