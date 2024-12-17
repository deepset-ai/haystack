# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0
import copy
from typing import List
from unittest import mock

import pytest

from haystack import Pipeline
from haystack.components.routers import ConditionalRouter
from haystack.components.routers.conditional_router import NoRouteSelectedException
from haystack.dataclasses import ChatMessage


def custom_filter_to_sede(value):
    """splits by hyphen and returns the first part"""
    return int(value.split("-")[0])


class TestRouter:
    def test_missing_mandatory_fields(self):
        """
        Router raises a ValueError if each route does not contain 'condition', 'output', and 'output_type' keys
        """
        routes = [
            {"condition": "{{streams|length < 2}}", "output": "{{query}}"},
            {"condition": "{{streams|length < 2}}", "output_type": str},
        ]
        with pytest.raises(ValueError):
            ConditionalRouter(routes)

    def test_invalid_condition_field(self):
        """
        ConditionalRouter init raises a ValueError if one of the routes contains invalid condition
        """
        # invalid condition field
        routes = [{"condition": "{{streams|length < 2", "output": "query", "output_type": str, "output_name": "test"}]
        with pytest.raises(ValueError, match="Invalid template"):
            ConditionalRouter(routes)

    def test_no_vars_in_output_route_but_with_output_name(self):
        """
        Router can't accept a route with no variables used in the output field
        """
        routes = [
            {
                "condition": "{{streams|length > 2}}",
                "output": "This is a constant",
                "output_name": "enough_streams",
                "output_type": str,
            }
        ]
        router = ConditionalRouter(routes)
        kwargs = {"streams": [1, 2, 3], "query": "Haystack"}
        result = router.run(**kwargs)
        assert result == {"enough_streams": "This is a constant"}

    def test_mandatory_and_optional_fields_with_extra_fields(self):
        """
        Router accepts a list of routes with mandatory and optional fields but not if some new field is added
        """

        routes = [
            {
                "condition": "{{streams|length < 2}}",
                "output": "{{query}}",
                "output_type": str,
                "output_name": "test",
                "bla": "bla",
            },
            {"condition": "{{streams|length < 2}}", "output": "{{query}}", "output_type": str},
        ]

        with pytest.raises(ValueError):
            ConditionalRouter(routes)

    def test_router_initialized(self):
        routes = [
            {"condition": "{{streams|length < 2}}", "output": "{{query}}", "output_type": str, "output_name": "query"},
            {
                "condition": "{{streams|length >= 2}}",
                "output": "{{streams}}",
                "output_type": List[int],
                "output_name": "streams",
            },
        ]
        router = ConditionalRouter(routes)

        assert router.routes == routes
        assert set(router.__haystack_input__._sockets_dict.keys()) == {"query", "streams"}
        assert set(router.__haystack_output__._sockets_dict.keys()) == {"query", "streams"}

    def test_router_evaluate_condition_expressions(self):
        router = ConditionalRouter(
            [
                {
                    "condition": "{{streams|length < 2}}",
                    "output": "{{query}}",
                    "output_type": str,
                    "output_name": "query",
                },
                {
                    "condition": "{{streams|length >= 2}}",
                    "output": "{{streams}}",
                    "output_type": List[int],
                    "output_name": "streams",
                },
            ]
        )
        # first route should be selected
        kwargs = {"streams": [1, 2, 3], "query": "test"}
        result = router.run(**kwargs)
        assert result == {"streams": [1, 2, 3]}

        # second route should be selected
        kwargs = {"streams": [1], "query": "test"}
        result = router.run(**kwargs)
        assert result == {"query": "test"}

    def test_router_evaluate_condition_expressions_using_output_slot(self):
        routes = [
            {
                "condition": "{{streams|length > 2}}",
                "output": "{{streams}}",
                "output_name": "enough_streams",
                "output_type": List[int],
            },
            {
                "condition": "{{streams|length <= 2}}",
                "output": "{{streams}}",
                "output_name": "insufficient_streams",
                "output_type": List[int],
            },
        ]
        router = ConditionalRouter(routes)
        # enough_streams output slot will be selected with [1, 2, 3] list being outputted
        kwargs = {"streams": [1, 2, 3], "query": "Haystack"}
        result = router.run(**kwargs)
        assert result == {"enough_streams": [1, 2, 3]}

    def test_complex_condition(self):
        routes = [
            {
                "condition": "{{messages[-1].meta.finish_reason == 'function_call'}}",
                "output": "{{streams}}",
                "output_type": List[int],
                "output_name": "streams",
            },
            {
                "condition": "{{True}}",
                "output": "{{query}}",
                "output_type": str,
                "output_name": "query",
            },  # catch-all condition
        ]
        router = ConditionalRouter(routes)
        message = mock.MagicMock()
        message.meta.finish_reason = "function_call"
        result = router.run(messages=[message], streams=[1, 2, 3], query="my query")
        assert result == {"streams": [1, 2, 3]}

    def test_router_no_route(self):
        # should raise an exception
        router = ConditionalRouter(
            [
                {
                    "condition": "{{streams|length < 2}}",
                    "output": "{{query}}",
                    "output_type": str,
                    "output_name": "query",
                },
                {
                    "condition": "{{streams|length >= 5}}",
                    "output": "{{streams}}",
                    "output_type": List[int],
                    "output_name": "streams",
                },
            ]
        )

        kwargs = {"streams": [1, 2, 3], "query": "test"}
        with pytest.raises(NoRouteSelectedException):
            router.run(**kwargs)

    def test_router_raises_value_error_if_route_not_dictionary(self):
        """
        Router raises a ValueError if each route is not a dictionary
        """
        routes = [
            {"condition": "{{streams|length < 2}}", "output": "{{query}}", "output_type": str, "output_name": "query"},
            ["{{streams|length >= 2}}", "streams", List[int]],
        ]

        with pytest.raises(ValueError):
            ConditionalRouter(routes)

    def test_router_raises_value_error_if_route_missing_keys(self):
        """
        Router raises a ValueError if each route does not contain 'condition', 'output', and 'output_type' keys
        """
        routes = [
            {"condition": "{{streams|length < 2}}", "output": "{{query}}"},
            {"condition": "{{streams|length < 2}}", "output_type": str},
        ]

        with pytest.raises(ValueError):
            ConditionalRouter(routes)

    def test_router_de_serialization(self):
        routes = [
            {"condition": "{{streams|length < 2}}", "output": "{{query}}", "output_type": str, "output_name": "query"},
            {
                "condition": "{{streams|length >= 2}}",
                "output": "{{streams}}",
                "output_type": List[int],
                "output_name": "streams",
            },
        ]
        router = ConditionalRouter(routes)
        router_dict = router.to_dict()

        # assert that the router dict is correct, with all keys and values being strings
        for route in router_dict["init_parameters"]["routes"]:
            for key in route.keys():
                assert isinstance(key, str)
                assert isinstance(route[key], str)

        new_router = ConditionalRouter.from_dict(router_dict)
        assert router.routes == new_router.routes

        # now use both routers with the same input
        kwargs = {"streams": [1, 2, 3], "query": "Haystack"}
        result1 = router.run(**kwargs)
        result2 = new_router.run(**kwargs)

        # check that the result is the same and correct
        assert result1 == result2 and result1 == {"streams": [1, 2, 3]}

    def test_router_de_serialization_with_none_argument(self):
        new_router = ConditionalRouter.from_dict(
            {
                "type": "haystack.components.routers.conditional_router.ConditionalRouter",
                "init_parameters": {
                    "routes": [
                        {
                            "condition": "{{streams|length < 2}}",
                            "output": "{{query}}",
                            "output_type": "str",
                            "output_name": "query",
                        },
                        {
                            "condition": "{{streams|length >= 2}}",
                            "output": "{{streams}}",
                            "output_type": "typing.List[int]",
                            "output_name": "streams",
                        },
                    ],
                    "custom_filters": None,
                    "unsafe": False,
                },
            }
        )

        # now use both routers with the same input
        kwargs = {"streams": [1, 2, 3], "query": "Haystack"}
        result2 = new_router.run(**kwargs)
        assert result2 == {"streams": [1, 2, 3]}

    def test_router_serialization_idempotence(self):
        routes = [
            {
                "condition": "{{streams|length < 2}}",
                "output": "{{message}}",
                "output_type": ChatMessage,
                "output_name": "message",
            },
            {
                "condition": "{{streams|length >= 2}}",
                "output": "{{streams}}",
                "output_type": List[int],
                "output_name": "streams",
            },
        ]
        router = ConditionalRouter(routes)
        # invoke to_dict twice and check that the result is the same
        router_dict_first_invocation = copy.deepcopy(router.to_dict())
        router_dict_second_invocation = router.to_dict()
        assert router_dict_first_invocation == router_dict_second_invocation

    def test_custom_filter(self):
        routes = [
            {
                "condition": "{{phone_num|get_area_code == 123}}",
                "output": "Phone number has a 123 area code",
                "output_name": "good_phone_num",
                "output_type": str,
            },
            {
                "condition": "{{phone_num|get_area_code != 123}}",
                "output": "Phone number does not have 123 area code",
                "output_name": "bad_phone_num",
                "output_type": str,
            },
        ]

        router = ConditionalRouter(routes, custom_filters={"get_area_code": custom_filter_to_sede})
        kwargs = {"phone_num": "123-456-7890"}
        result = router.run(**kwargs)
        assert result == {"good_phone_num": "Phone number has a 123 area code"}
        kwargs = {"phone_num": "321-456-7890"}
        result = router.run(**kwargs)
        assert result == {"bad_phone_num": "Phone number does not have 123 area code"}

    def test_sede_with_custom_filter(self):
        routes = [
            {
                "condition": "{{ test|custom_filter_to_sede == 123 }}",
                "output": "123",
                "output_name": "test",
                "output_type": int,
            }
        ]
        custom_filters = {"custom_filter_to_sede": custom_filter_to_sede}
        router = ConditionalRouter(routes, custom_filters=custom_filters)
        kwargs = {"test": "123-456-789"}
        result = router.run(**kwargs)
        assert result == {"test": 123}
        serialized_router = router.to_dict()
        deserialized_router = ConditionalRouter.from_dict(serialized_router)
        assert deserialized_router.custom_filters == router.custom_filters
        assert deserialized_router.custom_filters["custom_filter_to_sede"]("123-456-789") == 123
        assert result == deserialized_router.run(**kwargs)

    def test_unsafe(self):
        routes = [
            {
                "condition": "{{streams|length < 2}}",
                "output": "{{message}}",
                "output_type": ChatMessage,
                "output_name": "message",
            },
            {
                "condition": "{{streams|length >= 2}}",
                "output": "{{streams}}",
                "output_type": List[int],
                "output_name": "streams",
            },
        ]
        router = ConditionalRouter(routes, unsafe=True)
        streams = [1]
        message = ChatMessage.from_user("This is a message")
        res = router.run(streams=streams, message=message)
        assert res == {"message": message}

    def test_validate_output_type_without_unsafe(self):
        routes = [
            {
                "condition": "{{streams|length < 2}}",
                "output": "{{message}}",
                "output_type": ChatMessage,
                "output_name": "message",
            },
            {
                "condition": "{{streams|length >= 2}}",
                "output": "{{streams}}",
                "output_type": List[int],
                "output_name": "streams",
            },
        ]
        router = ConditionalRouter(routes, validate_output_type=True)
        streams = [1]
        message = ChatMessage.from_user("This is a message")
        with pytest.raises(ValueError, match="Route 'message' type doesn't match expected type"):
            router.run(streams=streams, message=message)

    def test_validate_output_type_with_unsafe(self):
        routes = [
            {
                "condition": "{{streams|length < 2}}",
                "output": "{{message}}",
                "output_type": ChatMessage,
                "output_name": "message",
            },
            {
                "condition": "{{streams|length >= 2}}",
                "output": "{{streams}}",
                "output_type": List[int],
                "output_name": "streams",
            },
        ]
        router = ConditionalRouter(routes, unsafe=True, validate_output_type=True)
        streams = [1]
        message = ChatMessage.from_user("This is a message")
        res = router.run(streams=streams, message=message)
        assert isinstance(res["message"], ChatMessage)

        streams = ["1", "2", "3", "4"]
        with pytest.raises(ValueError, match="Route 'streams' type doesn't match expected type"):
            router.run(streams=streams, message=message)

    def test_router_with_optional_parameters(self):
        """
        Test that the router works with optional parameters, particularly testing the default/fallback route
        when an expected parameter is not provided.
        """
        routes = [
            {"condition": '{{path == "rag"}}', "output": "{{question}}", "output_name": "normal", "output_type": str},
            {
                "condition": '{{path == "followup_short"}}',
                "output": "{{question}}",
                "output_name": "followup_short",
                "output_type": str,
            },
            {
                "condition": '{{path == "followup_elaborate"}}',
                "output": "{{question}}",
                "output_name": "followup_elaborate",
                "output_type": str,
            },
            {"condition": "{{ True }}", "output": "{{ question }}", "output_name": "fallback", "output_type": str},
        ]

        router = ConditionalRouter(routes, optional_variables=["path"])

        # Test direct component usage
        result = router.run(question="What?")
        assert result == {"fallback": "What?"}, "Default route should be taken when 'path' is not provided"

        # Test with path parameter
        result = router.run(question="What?", path="rag")
        assert result == {"normal": "What?"}, "Specific route should be taken when 'path' is provided"

        pipe = Pipeline()
        pipe.add_component("router", router)

        # Test pipeline without path parameter
        result = pipe.run(data={"router": {"question": "What?"}})
        assert result["router"] == {
            "fallback": "What?"
        }, "Default route should work in pipeline when 'path' is not provided"

        # Test pipeline with path parameter
        result = pipe.run(data={"router": {"question": "What?", "path": "followup_short"}})
        assert result["router"] == {"followup_short": "What?"}, "Specific route should work in pipeline"

    def test_router_with_multiple_optional_parameters(self):
        """
        Test ConditionalRouter with a mix of mandatory and optional parameters,
        exploring various combinations of provided/missing optional variables.
        """
        routes = [
            {
                "condition": '{{mode == "chat" and language == "en" and source == "doc"}}',
                "output": "{{question}}",
                "output_name": "en_doc_chat",
                "output_type": str,
            },
            {
                "condition": '{{mode == "qa" and source == "web"}}',
                "output": "{{question}}",
                "output_name": "web_qa",
                "output_type": str,
            },
            {
                "condition": '{{mode == "qa" and source == "doc"}}',
                "output": "{{question}}",
                "output_name": "doc_qa",
                "output_type": str,
            },
            {
                "condition": '{{mode == "chat" and language == "en"}}',
                "output": "{{question}}",
                "output_name": "en_chat",
                "output_type": str,
            },
            {
                "condition": '{{mode == "chat"}}',  # fallback for chat without language
                "output": "{{question}}",
                "output_name": "default_chat",
                "output_type": str,
            },
            {
                "condition": "{{ True }}",  # global fallback
                "output": "{{question}}",
                "output_name": "fallback",
                "output_type": str,
            },
        ]

        # There are four variables in the routes:
        # - mandatory: mode, question (always must be provided) or we'll route to fallback
        # - optional: source, language
        router = ConditionalRouter(routes, optional_variables=["source", "language"])

        # Test with mandatory parameter only
        result = router.run(question="What?", mode="chat")
        assert result == {"default_chat": "What?"}, "Should use chat fallback when language not provided"

        # Test with all parameters provided
        result = router.run(question="What?", mode="chat", language="en", source="doc")
        assert result == {"en_doc_chat": "What?"}, "Should use specific route when all params provided"

        # Test with different mandatory value and one optional
        result = router.run(question="What?", mode="qa", source="web")
        assert result == {"web_qa": "What?"}, "Should route qa with source correctly"

        # Test with mandatory the routes to fallback
        result = router.run(question="What?", mode="qa")
        assert result == {"fallback": "What?"}, "Should use global fallback for qa without source"

        # Test in pipeline
        pipe = Pipeline()
        pipe.add_component("router", router)

        # Test pipeline with mandatory only
        result = pipe.run(data={"router": {"question": "What?", "mode": "chat"}})
        assert result["router"] == {"default_chat": "What?"}, "Pipeline should handle missing optionals"

        # Test pipeline with mandatory and one optional
        result = pipe.run(data={"router": {"question": "What?", "mode": "qa", "source": "doc"}})
        assert result["router"] == {"doc_qa": "What?"}, "Pipeline should handle all parameters"

        # Test pipeline with mandatory and both optionals
        result = pipe.run(data={"router": {"question": "What?", "mode": "chat", "language": "en", "source": "doc"}})
        assert result["router"] == {"en_doc_chat": "What?"}, "Pipeline should handle all parameters"

    def test_warns_on_unused_optional_variables(self):
        """
        Test that a warning is raised when optional_variables contains variables
        that are not used in any route conditions or outputs.
        """
        routes = [
            {"condition": '{{mode == "chat"}}', "output": "{{question}}", "output_name": "chat", "output_type": str},
            {"condition": "{{ True }}", "output": "{{question}}", "output_name": "fallback", "output_type": str},
        ]

        # Initialize with unused optional variables and capture warning
        with pytest.warns(UserWarning, match="optional variables"):
            router = ConditionalRouter(routes=routes, optional_variables=["unused_var1", "unused_var2"])

        # Verify router still works normally
        result = router.run(question="What?", mode="chat")
        assert result == {"chat": "What?"}
