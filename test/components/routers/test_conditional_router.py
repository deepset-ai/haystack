import copy
from typing import List
from unittest import mock

import pytest

from haystack.components.routers import ConditionalRouter
from haystack.components.routers.conditional_router import NoRouteSelectedException
from haystack.dataclasses import ChatMessage


class TestRouter:
    @pytest.fixture
    def routes(self):
        return [
            {"condition": "{{streams|length < 2}}", "output": "{{query}}", "output_type": str, "output_name": "query"},
            {
                "condition": "{{streams|length >= 2}}",
                "output": "{{streams}}",
                "output_type": List[int],
                "output_name": "streams",
            },
        ]

    @pytest.fixture
    def router(self, routes):
        return ConditionalRouter(routes)

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

    def test_router_initialized(self, routes):
        router = ConditionalRouter(routes)

        assert router.routes == routes
        assert set(router.__canals_input__.keys()) == {"query", "streams"}
        assert set(router.__canals_output__.keys()) == {"query", "streams"}

    def test_router_evaluate_condition_expressions(self, router):
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

    def test_router_no_route(self, router):
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

    def test_router_de_serialization_user_type(self):
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
        router_dict = router.to_dict()

        # assert that the router dict is correct, with all keys and values being strings
        for route in router_dict["init_parameters"]["routes"]:
            for key in route.keys():
                assert isinstance(key, str)
                assert isinstance(route[key], str)

        # check that the output_type is a string and a proper class name
        assert (
            router_dict["init_parameters"]["routes"][0]["output_type"]
            == "haystack.dataclasses.chat_message.ChatMessage"
        )

        # deserialize the router
        new_router = ConditionalRouter.from_dict(router_dict)

        # check that the output_type is the right class
        assert new_router.routes[0]["output_type"] == ChatMessage
        assert router.routes == new_router.routes

        # now use both routers to run the same message
        message = ChatMessage.from_user("ciao")
        kwargs = {"streams": [1], "message": message}
        result1 = router.run(**kwargs)
        result2 = new_router.run(**kwargs)

        # check that the result is the same and correct
        assert result1 == result2 and result1["message"].content == message.content

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
