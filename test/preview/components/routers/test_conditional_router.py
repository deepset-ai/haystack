from unittest import mock
from typing import List

import pytest

from haystack.preview.components.routers import ConditionalRouter
from haystack.preview.components.routers.conditional_router import NoRouteSelectedException


class TestRouter:
    @pytest.fixture
    def routes(self):
        return [
            {"condition": "{{streams|length < 2}}", "output": "query", "output_type": str},
            {"condition": "{{streams|length >= 2}}", "output": "streams", "output_type": List[int]},
        ]

    @pytest.fixture
    def router(self, routes):
        return ConditionalRouter(routes)

    @pytest.mark.unit
    def test_missing_mandatory_fields(self):
        """
        Router raises a ValueError if each route does not contain 'condition', 'output', and 'output_type' keys
        """
        routes = [
            {"condition": "{{streams|length < 2}}", "output": "query"},
            {"condition": "{{streams|length < 2}}", "output_type": str},
        ]
        with pytest.raises(ValueError):
            ConditionalRouter(routes)

    @pytest.mark.unit
    def test_mandatory_and_optional_fields(self):
        """
        Router accepts a list of routes with mandatory and optional fields
        """
        routes = [
            {"condition": "{{streams|length < 2}}", "output": "query", "output_type": str, "output_slot": "test"},
            {"condition": "{{streams|length < 2}}", "output": "query", "output_type": str},
        ]

        ConditionalRouter(routes)

    @pytest.mark.unit
    def test_mandatory_and_optional_fields_with_extra_fields(self):
        """
        Router accepts a list of routes with mandatory and optional fields but not if some new field is added
        """

        routes = [
            {
                "condition": "{{streams|length < 2}}",
                "output": "query",
                "output_type": str,
                "output_slot": "test",
                "bla": "bla",
            },
            {"condition": "{{streams|length < 2}}", "output": "query", "output_type": str},
        ]

        with pytest.raises(ValueError):
            ConditionalRouter(routes)

    @pytest.mark.unit
    def test_router_initialized(self, routes):
        router = ConditionalRouter(routes)

        assert router.routes == routes
        assert set(router.__canals_input__.keys()) == {"query", "streams"}
        assert set(router.__canals_output__.keys()) == {"query", "streams"}

    @pytest.mark.unit
    def test_router_evaluate_condition_expressions(self, router):
        # first route should be selected
        kwargs = {"streams": [1, 2, 3], "query": "test"}
        result = router.run(**kwargs)
        assert result == {"streams": [1, 2, 3]}

        # second route should be selected
        kwargs = {"streams": [1], "query": "test"}
        result = router.run(**kwargs)
        assert result == {"query": "test"}

    @pytest.mark.unit
    def test_router_evaluate_condition_expressions_using_output_slot(self):
        routes = [
            {
                "condition": "{{streams|length > 2}}",
                "output": "streams",
                "output_slot": "enough_streams",
                "output_type": str,
            },
            {
                "condition": "{{streams|length <= 2}}",
                "output": "streams",
                "output_slot": "insufficient_streams",
                "output_type": List[int],
            },
        ]
        router = ConditionalRouter(routes)
        # enough_streams output slot will be selected with [1, 2, 3] list being outputted
        kwargs = {"streams": [1, 2, 3], "query": "Haystack"}
        result = router.run(**kwargs)
        assert result == {"enough_streams": [1, 2, 3]}

    @pytest.mark.unit
    def test_complex_condition(self):
        routes = [
            {
                "condition": "{{messages[-1].metadata.finish_reason == 'function_call'}}",
                "output": "streams",
                "output_type": List[int],
            },
            {"condition": "{{True}}", "output": "query", "output_type": str},  # catch-all condition
        ]
        router = ConditionalRouter(routes)
        message = mock.MagicMock()
        message.metadata.finish_reason = "function_call"
        result = router.run(messages=[message], streams=[1, 2, 3], query="my query")
        assert result == {"streams": [1, 2, 3]}

    @pytest.mark.unit
    def test_router_no_route(self, router):
        # should raise an exception
        router = ConditionalRouter(
            [
                {"condition": "{{streams|length < 2}}", "output": "query", "output_type": str},
                {"condition": "{{streams|length >= 5}}", "output": "streams", "output_type": List[int]},
            ]
        )

        kwargs = {"streams": [1, 2, 3], "query": "test"}
        with pytest.raises(NoRouteSelectedException):
            router.run(**kwargs)

    @pytest.mark.unit
    def test_router_raises_value_error_if_route_not_dictionary(self):
        """
        Router raises a ValueError if each route is not a dictionary
        """
        routes = [
            {"condition": "{{streams|length < 2}}", "output": "query", "output_type": str},
            ["{{streams|length >= 2}}", "streams", List[int]],
        ]

        with pytest.raises(ValueError):
            ConditionalRouter(routes)

    @pytest.mark.unit
    def test_router_raises_value_error_if_route_missing_keys(self):
        """
        Router raises a ValueError if each route does not contain 'condition', 'output', and 'output_type' keys
        """
        routes = [
            {"condition": "{{streams|length < 2}}", "output": "query"},
            {"condition": "{{streams|length < 2}}", "output_type": str},
        ]

        with pytest.raises(ValueError):
            ConditionalRouter(routes)
