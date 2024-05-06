from typing import Any, Dict, List, Optional
from jinja2 import TemplateSyntaxError
import pytest

from haystack.components.builders.prompt_builder import PromptBuilder
from haystack import component
from haystack.core.pipeline.pipeline import Pipeline
from haystack.dataclasses.document import Document


class TestPromptBuilder:
    def test_init(self):
        builder = PromptBuilder(template="This is a {{ variable }}")
        assert builder.default_template is not None
        assert builder.required_default_template_variables == {"variable"}
        assert builder.optional_variables == []
        assert builder._default_template_string == "This is a {{ variable }}"
        assert builder._variables is None
        assert builder._optional_variables is None

        # we have inputs that contain: template, template_variables + inferred variables
        inputs = builder.__haystack_input__._sockets_dict
        assert set(inputs.keys()) == {"template", "template_variables", "variable"}
        assert inputs["template"].type == Optional[str]
        assert inputs["template_variables"].type == Optional[Dict[str, Any]]
        assert inputs["variable"].type == Optional[Any]

        # response is always prompt
        outputs = builder.__haystack_output__._sockets_dict
        assert set(outputs.keys()) == {"prompt"}
        assert outputs["prompt"].type == str

    def test_init_without_template(self):
        variables = ["var1", "var2"]
        builder = PromptBuilder(variables=variables)
        assert builder.default_template is None
        assert builder.required_default_template_variables == set()
        assert builder.optional_variables == []
        assert builder._variables == variables
        assert builder._optional_variables is None
        assert builder._default_template_string is None

        # we have inputs that contain: template, template_variables + variables
        inputs = builder.__haystack_input__._sockets_dict
        assert set(inputs.keys()) == {"template", "template_variables", "var1", "var2"}
        assert inputs["template"].type == Optional[str]
        assert inputs["template_variables"].type == Optional[Dict[str, Any]]
        assert inputs["var1"].type == Optional[Any]
        assert inputs["var2"].type == Optional[Any]

        # response is always prompt
        outputs = builder.__haystack_output__._sockets_dict
        assert set(outputs.keys()) == {"prompt"}
        assert outputs["prompt"].type == str

    def test_init_with_optional_variables(self):
        builder = PromptBuilder(template="This is a {{ variable }}", optional_variables=["variable"])
        assert builder.default_template is not None
        assert builder.required_default_template_variables == {"variable"}
        assert builder.optional_variables == ["variable"]
        assert builder._default_template_string == "This is a {{ variable }}"
        assert builder._variables is None
        assert builder._optional_variables == ["variable"]

        # we have inputs that contain: template, template_variables + inferred variables
        inputs = builder.__haystack_input__._sockets_dict
        assert set(inputs.keys()) == {"template", "template_variables", "variable"}
        assert inputs["template"].type == Optional[str]
        assert inputs["template_variables"].type == Optional[Dict[str, Any]]
        assert inputs["variable"].type == Optional[Any]

        # response is always prompt
        outputs = builder.__haystack_output__._sockets_dict
        assert set(outputs.keys()) == {"prompt"}
        assert outputs["prompt"].type == str

    def test_init_with_custom_variables(self):
        variables = ["var1", "var2", "var3"]
        template = "Hello, {{ var1 }}, {{ var2 }}!"
        builder = PromptBuilder(template=template, variables=variables)
        assert builder.default_template is not None
        assert builder.required_default_template_variables == {"var1", "var2"}
        assert builder.optional_variables == []
        assert builder._variables == variables
        assert builder._default_template_string == "Hello, {{ var1 }}, {{ var2 }}!"
        assert builder._optional_variables is None

        # we have inputs that contain: template, template_variables + variables
        inputs = builder.__haystack_input__._sockets_dict
        assert set(inputs.keys()) == {"template", "template_variables", "var1", "var2", "var3"}
        assert inputs["template"].type == Optional[str]
        assert inputs["template_variables"].type == Optional[Dict[str, Any]]
        assert inputs["var1"].type == Optional[Any]
        assert inputs["var2"].type == Optional[Any]
        assert inputs["var3"].type == Optional[Any]

        # response is always prompt
        outputs = builder.__haystack_output__._sockets_dict
        assert set(outputs.keys()) == {"prompt"}
        assert outputs["prompt"].type == str

    def test_to_dict(self):
        builder = PromptBuilder(template="This is a {{ variable }}")
        res = builder.to_dict()
        assert res == {
            "type": "haystack.components.builders.prompt_builder.PromptBuilder",
            "init_parameters": {"template": "This is a {{ variable }}", "variables": None, "optional_variables": None},
        }

    def test_run(self):
        builder = PromptBuilder(template="This is a {{ variable }}")
        res = builder.run(variable="test")
        assert res == {"prompt": "This is a test"}

    def test_run_without_input(self):
        builder = PromptBuilder(template="This is a template without input")
        res = builder.run()
        assert res == {"prompt": "This is a template without input"}

    def test_run_with_missing_input(self):
        builder = PromptBuilder(template="This is a {{ variable }}")
        with pytest.raises(ValueError, match="variable"):
            builder.run()

    def test_run_with_missing_input_for_optional_variable(self):
        builder = PromptBuilder(template="This is a {{ variable }}", optional_variables=["variable"])
        res = builder.run()
        assert res == {"prompt": "This is a "}

    def test_run_with_missing_required_input(self):
        builder = PromptBuilder(template="This is a {{ foo }}, not a {{ bar }}")
        with pytest.raises(ValueError, match="'foo'"):
            builder.run(bar="bar")
        with pytest.raises(ValueError, match="'bar'"):
            builder.run(foo="foo")
        with pytest.raises(ValueError, match="'foo', 'bar'"):
            builder.run()

    def test_processing_a_simple_template_with_provided_variables(self):
        variables = ["var1", "var2", "var3"]

        builder = PromptBuilder(variables=variables)

        template = "Hello, {{ name }}! {{ var1 }}"
        template_variables = {"name": "John"}
        expected_result = {"prompt": "Hello, John! How are you?"}

        assert (
            builder.run(template=template, template_variables=template_variables, var1="How are you?")
            == expected_result
        )

    def test_processing_a_simple_default_template_with_provided_variables(self):
        variables = ["var1", "var2", "var3"]
        template = "Hello, {{ name }}! {{ var1 }}"

        builder = PromptBuilder(template=template, variables=variables)

        template_variables = {"name": "John"}
        expected_result = {"prompt": "Hello, John! How are you?"}

        assert builder.run(template_variables=template_variables, var1="How are you?") == expected_result

    def test_overwriting_default_template(self):
        variables = ["var1", "var2", "var3"]
        default_template = "Hello, {{ name }}!"

        builder = PromptBuilder(template=default_template, variables=variables)

        template = "Hello, {{ var1 }} {{ name }}!"
        template_variables = {"name": "John"}
        expected_result = {"prompt": "Hello, Big John!"}

        assert builder.run(template, template_variables, var1="Big") == expected_result

    def test_processing_a_simple_template_with_invalid_template(self):
        builder = PromptBuilder()

        template = "Hello, {{ name }!"
        template_variables = {"name": "John"}
        with pytest.raises(TemplateSyntaxError):
            builder.run(template, template_variables)

    def test_processing_a_simple_default_template_with_invalid_template(self):
        template = "Hello, {{ name }!"
        with pytest.raises(TemplateSyntaxError):
            PromptBuilder(template)

    def test_processing_a_simple_template_with_missing_variables(self):
        builder = PromptBuilder()

        with pytest.raises(ValueError, match="name"):
            builder.run("Hello, {{ name }}!")

    def test_missing_template_variables(self):
        prompt_builder = PromptBuilder(variables=["documents"])

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
        prompt_builder = PromptBuilder(variables=["documents"])

        # both variables are provided
        prompt_builder._validate_template("Hello, I'm {{ name }}, and I live in {{ city }}.", {"name", "city"})

        # provided variables are a superset of the required variables
        prompt_builder._validate_template("Hello, I'm {{ name }}, and I live in {{ city }}.", {"name", "city", "age"})

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

        template = "Here is the document: {{documents[0].content}} \\n Query: {{query}}"
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
                "prompt": "Here is the document: Hello world, I live in Berlin \\n Query: Where does the speaker live?"
            }
        }
