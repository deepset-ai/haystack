from typing import List

import pytest

from haystack.preview.components.routers import Router
from haystack.preview.components.routers.router import NoRouteSelectedException
from haystack.preview.dataclasses import ByteStream


class TestRouter:
    @pytest.mark.unit
    def test_router_initialized_with_routes_and_routing_variables(self):
        """
        Router can be initialized with a list of routes and routing variables
        """
        routes = [
            {"condition": "len(streams) < 2", "output": "query", "output_type": str},
            {"condition": "len(streams) >= 2", "output": "streams", "output_type": List[ByteStream]},
        ]
        routing_variables = ["query", "streams"]

        router = Router(routes=routes, routing_variables=routing_variables)

        assert router.routes == routes
        assert router.routing_variables == routing_variables

    @pytest.mark.unit
    def test_router_evaluate_condition_expressions(self):
        """
        Router can evaluate the specified boolean condition expressions for each route in the order they are listed
        """
        routes = [
            {"condition": "len(streams) < 2", "output": "query", "output_type": str},
            {"condition": "len(streams) >= 2", "output": "streams", "output_type": List[ByteStream]},
        ]
        routing_variables = ["query", "streams"]

        router = Router(routes=routes, routing_variables=routing_variables)

        # first route should be selected
        kwargs = {"streams": [1, 2, 3], "query": "test"}
        result = router.run(**kwargs)
        assert result == {"streams": [1, 2, 3]}

        # second route should be selected
        kwargs = {"streams": [1], "query": "test"}
        result = router.run(**kwargs)
        assert result == {"query": "test"}

    @pytest.mark.unit
    def test_router_no_route(self):
        """
        Router raises an exception if no route's expression evaluates to True
        """
        routes = [
            {"condition": "len(streams) < 2", "output": "query", "output_type": str},
            {"condition": "len(streams) >= 5", "output": "streams", "output_type": List[ByteStream]},
        ]
        routing_variables = ["query", "streams"]

        router = Router(routes=routes, routing_variables=routing_variables)

        # should raise an exception
        kwargs = {"streams": [1, 2, 3], "query": "test"}
        with pytest.raises(NoRouteSelectedException):
            router.run(**kwargs)

    @pytest.mark.unit
    def test_router_direct_flow_to_output_slot(self):
        """
        Router can direct the flow of data to the output slot specified in the first route
        whose expression evaluates to True
        """
        routes = [
            {"condition": "len(streams) < 2", "output": "query", "output_type": str},
            {"condition": "len(streams) >= 2", "output": "streams", "output_type": List[ByteStream]},
        ]
        routing_variables = ["query", "streams"]

        router = Router(routes=routes, routing_variables=routing_variables)

        kwargs = {"streams": [1], "query": "test"}
        result = router.run(**kwargs)

        assert "query" in result

    @pytest.mark.unit
    def test_router_raises_value_error_if_route_not_dictionary(self):
        """
        Router raises a ValueError if each route is not a dictionary
        """
        routes = [
            {"condition": "len(streams) < 2", "output": "query", "output_type": str},
            ["len(streams) >= 2", "streams", "List[ByteStream]"],
        ]
        routing_variables = ["query", "streams"]

        with pytest.raises(ValueError):
            Router(routes=routes, routing_variables=routing_variables)

    @pytest.mark.unit
    def test_router_raises_value_error_if_route_missing_keys(self):
        """
        Router raises a ValueError if each route does not contain 'condition', 'output', and 'output_type' keys
        """
        routes = [
            {"condition": "len(streams) < 2", "output": "query"},
            {"condition": "len(streams) >= 2", "output": "streams", "output_type": List[ByteStream]},
        ]
        routing_variables = ["query", "streams"]

        with pytest.raises(ValueError):
            Router(routes=routes, routing_variables=routing_variables)

    @pytest.mark.unit
    def test_router_raises_value_error_if_routing_variable_not_string(self):
        """
        Router raises a ValueError if each routing variable is not a string
        """
        routes = [
            {"condition": "len(streams) < 2", "output": "query", "output_type": str},
            {"condition": "len(streams) >= 2", "output": "streams", "output_type": List[ByteStream]},
        ]
        routing_variables = ["query", 123]

        with pytest.raises(ValueError):
            Router(routes=routes, routing_variables=routing_variables)
