import pytest

from haystack import Pipeline, component
from haystack.components.converters import OutputAdapter
from haystack.components.converters.output_adapter import OutputAdaptationException


class TestOutputAdapter:
    #  OutputAdapter can be initialized with a valid Jinja2 template string and output type.
    def test_initialized_with_valid_template_and_output_type(self):
        template = "{{ documents[0].content }}"
        output_type = str
        adapter = OutputAdapter(template="{{ documents[0].content }}", output_type=str)

        assert adapter.template == template
        assert adapter.__haystack_output__.output.name == "output"
        assert adapter.__haystack_output__.output.type == output_type

    #  OutputAdapter can adapt the output of one component to be compatible with the input of another
    #  component using Jinja2 template expressions.
    def test_output_adaptation(self):
        adapter = OutputAdapter(template="{{ documents[0].content }}", output_type=str)

        input_data = {"documents": [{"content": "Test content"}]}
        expected_output = {"output": "Test content"}

        assert adapter.run(**input_data) == expected_output

    #  OutputAdapter can handle predefined filters like 'json_loads' and 'json_dumps'.
    def test_predefined_filters(self):
        adapter = OutputAdapter(template="{{ documents[0].content|json_loads }}", output_type=dict)

        input_data = {"documents": [{"content": '{"key": "value"}'}]}
        expected_output = {"output": {"key": "value"}}

        assert adapter.run(**input_data) == expected_output

    #  OutputAdapter can handle custom filters provided in the component configuration.
    def test_custom_filters(self):
        def custom_filter(value):
            return value.upper()

        custom_filters = {"custom_filter": custom_filter}
        adapter = OutputAdapter(
            template="{{ documents[0].content|custom_filter }}", output_type=str, custom_filters=custom_filters
        )

        input_data = {"documents": [{"content": "test content"}]}
        expected_output = {"output": "TEST CONTENT"}

        assert adapter.run(**input_data) == expected_output

    #  OutputAdapter raises an exception on init if the Jinja2 template string is invalid.
    def test_invalid_template_string(self):
        with pytest.raises(ValueError):
            OutputAdapter(template="{{ documents[0].content }", output_type=str)

    #  OutputAdapter raises an exception if no input data is provided for output adaptation.
    def test_no_input_data_provided(self):
        adapter = OutputAdapter(template="{{ documents[0].content }}", output_type=str)
        with pytest.raises(ValueError):
            adapter.run()

    #  OutputAdapter raises an exception if there's an error during the adaptation process.
    def test_error_during_adaptation(self):
        adapter = OutputAdapter(template="{{ documents[0].content }}", output_type=str)
        input_data = {"documents": [{"title": "Test title"}]}

        with pytest.raises(OutputAdaptationException):
            adapter.run(**input_data)

    # OutputAdapter can be serialized to a dictionary and deserialized back to an OutputAdapter instance.
    def test_sede(self):
        adapter = OutputAdapter(template="{{ documents[0].content }}", output_type=str)
        adapter_dict = adapter.to_dict()
        deserialized_adapter = OutputAdapter.from_dict(adapter_dict)

        assert adapter.template == deserialized_adapter.template
        assert adapter.output_type == deserialized_adapter.output_type

    def test_output_adapter_in_pipeline(self):
        @component
        class DocumentProducer:
            @component.output_types(documents=dict)
            def run(self):
                return {"documents": [{"content": '{"framework": "Haystack"}'}]}

        pipe = Pipeline()
        pipe.add_component(
            name="output_adapter",
            instance=OutputAdapter(template="{{ documents[0].content | json_loads}}", output_type=str),
        )
        pipe.add_component(name="document_producer", instance=DocumentProducer())
        pipe.connect("document_producer", "output_adapter")
        result = pipe.run(data={})
        assert result
        assert result["output_adapter"]["output"] == {"framework": "Haystack"}
