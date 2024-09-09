# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import List
import json

import pytest

from haystack import Pipeline, component
from haystack.dataclasses import Document
from haystack.components.converters import OutputAdapter
from haystack.components.converters.output_adapter import OutputAdaptationException


def custom_filter_to_sede(value):
    return value.upper()


def another_custom_filter(value):
    return value.upper()


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

    #  OutputAdapter can add filter 'json_loads' and use it
    def test_predefined_filters(self):
        adapter = OutputAdapter(
            template="{{ documents[0].content|json_loads }}",
            output_type=dict,
            custom_filters={"json_loads": lambda s: json.loads(str(s))},
        )

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

    # OutputAdapter can be serialized to a dictionary and deserialized along with custom filters
    def test_sede_with_custom_filters(self):
        # NOTE: filters need to be declared in a namespace visible to the deserialization function
        custom_filters = {"custom_filter": custom_filter_to_sede}
        adapter = OutputAdapter(
            template="{{ documents[0].content|custom_filter }}", output_type=str, custom_filters=custom_filters
        )
        adapter_dict = adapter.to_dict()
        deserialized_adapter = OutputAdapter.from_dict(adapter_dict)

        assert adapter.template == deserialized_adapter.template
        assert adapter.output_type == deserialized_adapter.output_type
        assert adapter.custom_filters == deserialized_adapter.custom_filters == custom_filters

        # invoke the custom filter to check if it is deserialized correctly
        assert deserialized_adapter.custom_filters["custom_filter"]("test") == "TEST"

    # OutputAdapter can be serialized to a dictionary and deserialized along with multiple custom filters
    def test_sede_with_multiple_custom_filters(self):
        # NOTE: filters need to be declared in a namespace visible to the deserialization function
        custom_filters = {"custom_filter": custom_filter_to_sede, "another_custom_filter": another_custom_filter}
        adapter = OutputAdapter(
            template="{{ documents[0].content|custom_filter }}", output_type=str, custom_filters=custom_filters
        )
        adapter_dict = adapter.to_dict()
        deserialized_adapter = OutputAdapter.from_dict(adapter_dict)

        assert adapter.template == deserialized_adapter.template
        assert adapter.output_type == deserialized_adapter.output_type
        assert adapter.custom_filters == deserialized_adapter.custom_filters == custom_filters

        # invoke the custom filter to check if it is deserialized correctly
        assert deserialized_adapter.custom_filters["custom_filter"]("test") == "TEST"

    def test_sede_with_list_output_type_in_pipeline(self):
        pipe = Pipeline()
        pipe.add_component("adapter", OutputAdapter(template="{{ test }}", output_type=List[str]))
        serialized_pipe = pipe.dumps()

        # we serialize the pipeline and check if the output type is serialized correctly (as typing.List[str])
        assert "typing.List[str]" in serialized_pipe

        deserialized_pipe = Pipeline.loads(serialized_pipe)
        assert deserialized_pipe.get_component("adapter").output_type == List[str]

    def test_output_adapter_from_dict_custom_filters_none(self):
        component = OutputAdapter.from_dict(
            data={
                "type": "haystack.components.converters.output_adapter.OutputAdapter",
                "init_parameters": {
                    "template": "{{ documents[0].content}}",
                    "output_type": "str",
                    "custom_filters": None,
                    "unsafe": False,
                },
            }
        )

        assert component.template == "{{ documents[0].content}}"
        assert component.output_type == str
        assert component.custom_filters == {}
        assert not component._unsafe

    def test_output_adapter_in_pipeline(self):
        @component
        class DocumentProducer:
            @component.output_types(documents=dict)
            def run(self):
                return {"documents": [{"content": '{"framework": "Haystack"}'}]}

        pipe = Pipeline()
        pipe.add_component(
            name="output_adapter",
            instance=OutputAdapter(
                template="{{ documents[0].content | json_loads}}",
                output_type=str,
                custom_filters={"json_loads": lambda s: json.loads(str(s))},
            ),
        )
        pipe.add_component(name="document_producer", instance=DocumentProducer())
        pipe.connect("document_producer", "output_adapter")
        result = pipe.run(data={})
        assert result
        assert result["output_adapter"]["output"] == {"framework": "Haystack"}

    def test_unsafe(self):
        adapter = OutputAdapter(template="{{ documents[0] }}", output_type=Document, unsafe=True)
        documents = [
            Document(content="Test document"),
            Document(content="Another test document"),
            Document(content="Yet another test document"),
        ]
        res = adapter.run(documents=documents)
        assert res["output"] == documents[0]
