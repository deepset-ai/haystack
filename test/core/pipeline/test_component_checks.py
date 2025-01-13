import pytest
from typing import Dict, Any

from haystack.core.pipeline.component_checks import *
from haystack.core.pipeline.component_checks import _NO_OUTPUT_PRODUCED
from haystack.core.component.types import InputSocket, OutputSocket, Variadic, GreedyVariadic

@pytest.fixture
def basic_component():
    """Basic component with one mandatory and one optional input"""
    return {
        "instance": "mock_instance",
        "visits": 0,
        "input_sockets": {
            "mandatory_input": InputSocket("mandatory_input", int),
            "optional_input": InputSocket("optional_input", str, default_value="default")
        },
        "output_sockets": {
            "output": OutputSocket("output", int)
        }
    }

@pytest.fixture
def variadic_component():
    """Component with variadic input"""
    return {
        "instance": "mock_instance",
        "visits": 0,
        "input_sockets": {
            "variadic_input": InputSocket("variadic_input", Variadic[int]),
            "normal_input": InputSocket("normal_input", str)
        },
        "output_sockets": {
            "output": OutputSocket("output", int)
        }
    }

@pytest.fixture
def greedy_variadic_component():
    """Component with greedy variadic input"""
    return {
        "instance": "mock_instance",
        "visits": 0,
        "input_sockets": {
            "greedy_input": InputSocket("greedy_input", GreedyVariadic[int]),
            "normal_input": InputSocket("normal_input", str)
        },
        "output_sockets": {
            "output": OutputSocket("output", int)
        }
    }

@pytest.fixture
def input_socket_with_sender():
    """Regular input socket with a single sender"""
    socket = InputSocket("test_input", int)
    socket.senders = ["component1"]
    return socket

@pytest.fixture
def variadic_socket_with_senders():
    """Variadic input socket with multiple senders"""
    socket = InputSocket("test_variadic", Variadic[int])
    socket.senders = ["component1", "component2"]
    return socket

@pytest.fixture
def component_with_multiple_sockets(input_socket_with_sender, variadic_socket_with_senders):
    """Component with multiple input sockets including both regular and variadic"""
    return {
        "instance": "mock_instance",
        "input_sockets": {
            "socket1": input_socket_with_sender,
            "socket2": variadic_socket_with_senders,
            "socket3": InputSocket("socket3", str)  # No senders
        }
    }

@pytest.fixture
def regular_socket():
    """Regular input socket with one sender"""
    socket = InputSocket("regular", int)
    socket.senders = ["component1"]
    return socket

@pytest.fixture
def lazy_variadic_socket():
    """Lazy variadic input socket with multiple senders"""
    socket = InputSocket("lazy_variadic", Variadic[int])
    socket.senders = ["component1", "component2"]
    return socket

@pytest.fixture
def greedy_variadic_socket():
    """Greedy variadic input socket with multiple senders"""
    socket = InputSocket("greedy_variadic", GreedyVariadic[int])
    socket.senders = ["component1", "component2", "component3"]
    return socket

@pytest.fixture
def complex_component(regular_socket, lazy_variadic_socket, greedy_variadic_socket):
    """Component with all types of sockets"""
    return {
        "instance": "mock_instance",
        "input_sockets": {
            "regular": regular_socket,
            "lazy_var": lazy_variadic_socket,
            "greedy_var": greedy_variadic_socket
        }
    }

class TestCanComponentRun:
    def test_component_with_all_mandatory_inputs_and_trigger(self, basic_component):
        inputs = {
            "mandatory_input": [{"sender": "previous_component", "value": 42}]
        }
        assert can_component_run(basic_component, inputs) is True

    def test_component_missing_mandatory_input(self, basic_component):
        inputs = {
            "optional_input": [{"sender": "previous_component", "value": "test"}]
        }
        assert can_component_run(basic_component, inputs) is False

    def test_component_with_no_trigger_but_all_inputs(self, basic_component):
        """
        Test case where all mandatory inputs are present with valid values,
        but there is no trigger (no new input from predecessor, not first visit)
        """
        # Set visits > 0 so it's not triggered by first visit
        basic_component["visits"] = 1
        inputs = {
            "mandatory_input": [{"sender": None, "value": 42}]  # Valid input value
        }
        assert can_component_run(basic_component, inputs) is False

    def test_component_with_multiple_visits(self, basic_component):
        basic_component["visits"] = 2
        inputs = {
            "mandatory_input": [{"sender": "previous_component", "value": 42}]
        }
        assert can_component_run(basic_component, inputs) is True

    def test_component_with_no_inputs_first_visit(self, basic_component):
        basic_component["input_sockets"] = {}  # No inputs required
        inputs = {}
        assert can_component_run(basic_component, inputs) is True

class TestHasAnyTrigger:
    def test_trigger_from_predecessor(self, basic_component):
        inputs = {
            "mandatory_input": [{"sender": "previous_component", "value": 42}]
        }
        assert has_any_trigger(basic_component, inputs) is True

    def test_trigger_from_user_first_visit(self, basic_component):
        inputs = {
            "mandatory_input": [{"sender": None, "value": 42}]
        }
        assert has_any_trigger(basic_component, inputs) is True

    def test_no_trigger_from_user_after_first_visit(self, basic_component):
        basic_component["visits"] = 1
        inputs = {
            "mandatory_input": [{"sender": None, "value": 42}]
        }
        assert has_any_trigger(basic_component, inputs) is False

    def test_trigger_without_inputs_first_visit(self, basic_component):
        basic_component["input_sockets"] = {}  # No inputs
        inputs = {}
        assert has_any_trigger(basic_component, inputs) is True

    def test_no_trigger_without_inputs_after_first_visit(self, basic_component):
        basic_component["input_sockets"] = {}
        basic_component["visits"] = 1
        inputs = {}
        assert has_any_trigger(basic_component, inputs) is False

class TestAllMandatorySocketsReady:
    def test_all_mandatory_sockets_filled(self, basic_component):
        inputs = {
            "mandatory_input": [{"sender": "previous_component", "value": 42}]
        }
        assert are_all_mandatory_sockets_ready(basic_component, inputs) is True

    def test_missing_mandatory_socket(self, basic_component):
        inputs = {
            "optional_input": [{"sender": "previous_component", "value": "test"}]
        }
        assert are_all_mandatory_sockets_ready(basic_component, inputs) is False

    def test_variadic_socket_with_input(self, variadic_component):
        inputs = {
            "variadic_input": [{"sender": "previous_component", "value": 42}],
            "normal_input": [{"sender": "previous_component", "value": "test"}]
        }
        assert are_all_mandatory_sockets_ready(variadic_component, inputs) is True

    def test_greedy_variadic_socket_with_partial_input(self, greedy_variadic_component):
        inputs = {
            "greedy_input": [{"sender": "previous_component", "value": 42}],
            "normal_input": [{"sender": "previous_component", "value": "test"}]
        }
        assert are_all_mandatory_sockets_ready(greedy_variadic_component, inputs) is True

    def test_variadic_socket_no_input(self, variadic_component):
        inputs = {
            "normal_input": [{"sender": "previous_component", "value": "test"}]
        }
        assert are_all_mandatory_sockets_ready(variadic_component, inputs) is False

    def test_empty_inputs(self, basic_component):
        inputs = {}
        assert are_all_mandatory_sockets_ready(basic_component, inputs) is False

    def test_no_mandatory_sockets(self, basic_component):
        basic_component["input_sockets"] = {
            "optional_1": InputSocket("optional_1", str, default_value="default1"),
            "optional_2": InputSocket("optional_2", str, default_value="default2")
        }
        inputs = {}
        assert are_all_mandatory_sockets_ready(basic_component, inputs) is True

    def test_multiple_mandatory_sockets(self, basic_component):
        basic_component["input_sockets"] = {
            "mandatory_1": InputSocket("mandatory_1", int),
            "mandatory_2": InputSocket("mandatory_2", str),
            "optional": InputSocket("optional", bool, default_value=False)
        }
        inputs = {
            "mandatory_1": [{"sender": "comp1", "value": 42}],
            "mandatory_2": [{"sender": "comp2", "value": "test"}]
        }
        assert are_all_mandatory_sockets_ready(basic_component, inputs) is True

        # Missing one mandatory input
        inputs = {
            "mandatory_1": [{"sender": "comp1", "value": 42}],
            "optional": [{"sender": "comp3", "value": True}]
        }
        assert are_all_mandatory_sockets_ready(basic_component, inputs) is False

class TestPredecessorInputDetection:
    def test_any_predecessors_provided_input_with_predecessor(self, component_with_multiple_sockets):
        inputs = {
            "socket1": [{"sender": "component1", "value": 42}],
            "socket2": [{"sender": None, "value": "test"}]
        }
        assert any_predecessors_provided_input(component_with_multiple_sockets, inputs) is True

    def test_any_predecessors_provided_input_no_predecessor(self, component_with_multiple_sockets):
        inputs = {
            "socket1": [{"sender": None, "value": 42}],
            "socket2": [{"sender": None, "value": "test"}]
        }
        assert any_predecessors_provided_input(component_with_multiple_sockets, inputs) is False

    def test_any_predecessors_provided_input_with_no_output(self, component_with_multiple_sockets):
        inputs = {
            "socket1": [{"sender": "component1", "value": _NO_OUTPUT_PRODUCED}],
            "socket2": [{"sender": None, "value": "test"}]
        }
        assert any_predecessors_provided_input(component_with_multiple_sockets, inputs) is False

    def test_any_predecessors_provided_input_empty_inputs(self, component_with_multiple_sockets):
        inputs = {}
        assert any_predecessors_provided_input(component_with_multiple_sockets, inputs) is False

class TestSocketValueFromPredecessor:
    def test_socket_value_from_predecessor_with_valid_input(self):
        socket_inputs = [{"sender": "component1", "value": 42}]
        assert any_socket_value_from_predecessor_received(socket_inputs) is True

    def test_socket_value_from_predecessor_with_no_output(self):
        socket_inputs = [{"sender": "component1", "value": _NO_OUTPUT_PRODUCED}]
        assert any_socket_value_from_predecessor_received(socket_inputs) is False

    def test_socket_value_from_predecessor_with_user_input(self):
        socket_inputs = [{"sender": None, "value": 42}]
        assert any_socket_value_from_predecessor_received(socket_inputs) is False

    def test_socket_value_from_predecessor_with_mixed_inputs(self):
        socket_inputs = [
            {"sender": None, "value": 42},
            {"sender": "component1", "value": _NO_OUTPUT_PRODUCED},
            {"sender": "component2", "value": 100}
        ]
        assert any_socket_value_from_predecessor_received(socket_inputs) is True

    def test_socket_value_from_predecessor_empty_list(self):
        assert any_socket_value_from_predecessor_received([]) is False

class TestUserInputDetection:
    def test_has_user_input_with_user_input(self):
        inputs = {
            "socket1": [{"sender": None, "value": 42}],
            "socket2": [{"sender": "component1", "value": "test"}]
        }
        assert has_user_input(inputs) is True

    def test_has_user_input_without_user_input(self):
        inputs = {
            "socket1": [{"sender": "component1", "value": 42}],
            "socket2": [{"sender": "component2", "value": "test"}]
        }
        assert has_user_input(inputs) is False

    def test_has_user_input_empty_inputs(self):
        inputs = {}
        assert has_user_input(inputs) is False

    def test_has_user_input_with_no_output(self):
        inputs = {
            "socket1": [{"sender": None, "value": _NO_OUTPUT_PRODUCED}]
        }
        assert has_user_input(inputs) is True

class TestPipelineInputCapability:
    def test_cannot_receive_inputs_no_senders(self):
        component = {
            "input_sockets": {
                "socket1": InputSocket("socket1", int),
                "socket2": InputSocket("socket2", str)
            }
        }
        assert can_not_receive_inputs_from_pipeline(component) is True

    def test_cannot_receive_inputs_with_senders(self, component_with_multiple_sockets):
        assert can_not_receive_inputs_from_pipeline(component_with_multiple_sockets) is False

    def test_cannot_receive_inputs_mixed_senders(self, input_socket_with_sender):
        component = {
            "input_sockets": {
                "socket1": input_socket_with_sender,
                "socket2": InputSocket("socket2", str)  # No senders
            }
        }
        assert can_not_receive_inputs_from_pipeline(component) is False

class TestSocketExecutionStatus:
    def test_regular_socket_predecessor_executed(self, input_socket_with_sender):
        socket_inputs = [{"sender": "component1", "value": 42}]
        assert all_socket_predecessors_executed(input_socket_with_sender, socket_inputs) is True

    def test_regular_socket_predecessor_not_executed(self, input_socket_with_sender):
        socket_inputs = []
        assert all_socket_predecessors_executed(input_socket_with_sender, socket_inputs) is False

    def test_regular_socket_with_wrong_predecessor(self, input_socket_with_sender):
        socket_inputs = [{"sender": "component2", "value": 42}]
        assert all_socket_predecessors_executed(input_socket_with_sender, socket_inputs) is False

    def test_variadic_socket_all_predecessors_executed(self, variadic_socket_with_senders):
        socket_inputs = [
            {"sender": "component1", "value": 42},
            {"sender": "component2", "value": 43}
        ]
        assert all_socket_predecessors_executed(variadic_socket_with_senders, socket_inputs) is True

    def test_variadic_socket_partial_execution(self, variadic_socket_with_senders):
        socket_inputs = [{"sender": "component1", "value": 42}]
        assert all_socket_predecessors_executed(variadic_socket_with_senders, socket_inputs) is False

    def test_variadic_socket_with_user_input(self, variadic_socket_with_senders):
        socket_inputs = [
            {"sender": "component1", "value": 42},
            {"sender": None, "value": 43},
            {"sender": "component2", "value": 44}
        ]
        assert all_socket_predecessors_executed(variadic_socket_with_senders, socket_inputs) is True

    def test_variadic_socket_no_execution(self, variadic_socket_with_senders):
        socket_inputs = []
        assert all_socket_predecessors_executed(variadic_socket_with_senders, socket_inputs) is False

class TestSocketInputReceived:
    def test_any_socket_input_received_with_value(self):
        socket_inputs = [{"sender": "component1", "value": 42}]
        assert any_socket_input_received(socket_inputs) is True

    def test_any_socket_input_received_with_no_output(self):
        socket_inputs = [{"sender": "component1", "value": _NO_OUTPUT_PRODUCED}]
        assert any_socket_input_received(socket_inputs) is False

    def test_any_socket_input_received_mixed_inputs(self):
        socket_inputs = [
            {"sender": "component1", "value": _NO_OUTPUT_PRODUCED},
            {"sender": "component2", "value": 42}
        ]
        assert any_socket_input_received(socket_inputs) is True

    def test_any_socket_input_received_empty_list(self):
        assert any_socket_input_received([]) is False

class TestLazyVariadicSocket:
    def test_lazy_variadic_all_inputs_received(self, variadic_socket_with_senders):
        socket_inputs = [
            {"sender": "component1", "value": 42},
            {"sender": "component2", "value": 43}
        ]
        assert has_lazy_variadic_socket_received_all_inputs(variadic_socket_with_senders, socket_inputs) is True

    def test_lazy_variadic_partial_inputs(self, variadic_socket_with_senders):
        socket_inputs = [{"sender": "component1", "value": 42}]
        assert has_lazy_variadic_socket_received_all_inputs(variadic_socket_with_senders, socket_inputs) is False

    def test_lazy_variadic_with_no_output(self, variadic_socket_with_senders):
        socket_inputs = [
            {"sender": "component1", "value": _NO_OUTPUT_PRODUCED},
            {"sender": "component2", "value": 42}
        ]
        assert has_lazy_variadic_socket_received_all_inputs(variadic_socket_with_senders, socket_inputs) is False

    def test_lazy_variadic_with_user_input(self, variadic_socket_with_senders):
        socket_inputs = [
            {"sender": "component1", "value": 42},
            {"sender": None, "value": 43},
            {"sender": "component2", "value": 44}
        ]
        assert has_lazy_variadic_socket_received_all_inputs(variadic_socket_with_senders, socket_inputs) is True

    def test_lazy_variadic_empty_inputs(self, variadic_socket_with_senders):
        assert has_lazy_variadic_socket_received_all_inputs(variadic_socket_with_senders, []) is False

class TestSocketTypeDetection:
    def test_is_socket_lazy_variadic_with_lazy_socket(self, lazy_variadic_socket):
        assert is_socket_lazy_variadic(lazy_variadic_socket) is True

    def test_is_socket_lazy_variadic_with_greedy_socket(self, greedy_variadic_socket):
        assert is_socket_lazy_variadic(greedy_variadic_socket) is False

    def test_is_socket_lazy_variadic_with_regular_socket(self, regular_socket):
        assert is_socket_lazy_variadic(regular_socket) is False

class TestSocketInputCompletion:
    def test_regular_socket_complete(self, regular_socket):
        inputs = [{"sender": "component1", "value": 42}]
        assert has_socket_received_all_inputs(regular_socket, inputs) is True

    def test_regular_socket_incomplete(self, regular_socket):
        inputs = [{"sender": "component1", "value": _NO_OUTPUT_PRODUCED}]
        assert has_socket_received_all_inputs(regular_socket, inputs) is False

    def test_regular_socket_no_inputs(self, regular_socket):
        inputs = []
        assert has_socket_received_all_inputs(regular_socket, inputs) is False

    def test_lazy_variadic_socket_all_inputs(self, lazy_variadic_socket):
        inputs = [
            {"sender": "component1", "value": 42},
            {"sender": "component2", "value": 43}
        ]
        assert has_socket_received_all_inputs(lazy_variadic_socket, inputs) is True

    def test_lazy_variadic_socket_partial_inputs(self, lazy_variadic_socket):
        inputs = [{"sender": "component1", "value": 42}]
        assert has_socket_received_all_inputs(lazy_variadic_socket, inputs) is False

    def test_lazy_variadic_socket_with_no_output(self, lazy_variadic_socket):
        inputs = [
            {"sender": "component1", "value": 42},
            {"sender": "component2", "value": _NO_OUTPUT_PRODUCED}
        ]
        assert has_socket_received_all_inputs(lazy_variadic_socket, inputs) is False

    def test_greedy_variadic_socket_one_input(self, greedy_variadic_socket):
        inputs = [{"sender": "component1", "value": 42}]
        assert has_socket_received_all_inputs(greedy_variadic_socket, inputs) is True

    def test_greedy_variadic_socket_multiple_inputs(self, greedy_variadic_socket):
        inputs = [
            {"sender": "component1", "value": 42},
            {"sender": "component2", "value": 43}
        ]
        assert has_socket_received_all_inputs(greedy_variadic_socket, inputs) is True

    def test_greedy_variadic_socket_no_valid_inputs(self, greedy_variadic_socket):
        inputs = [{"sender": "component1", "value": _NO_OUTPUT_PRODUCED}]
        assert has_socket_received_all_inputs(greedy_variadic_socket, inputs) is False

class TestPredecessorExecution:
    def test_all_predecessors_executed_complete(self, complex_component):
        inputs = {
            "regular": [{"sender": "component1", "value": 42}],
            "lazy_var": [
                {"sender": "component1", "value": 42},
                {"sender": "component2", "value": 43}
            ],
            "greedy_var": [
                {"sender": "component1", "value": 42},
                {"sender": "component2", "value": 43},
                {"sender": "component3", "value": 44}
            ]
        }
        assert all_predecessors_executed(complex_component, inputs) is True

    def test_all_predecessors_executed_partial(self, complex_component):
        inputs = {
            "regular": [{"sender": "component1", "value": 42}],
            "lazy_var": [{"sender": "component1", "value": 42}],  # Missing component2
            "greedy_var": [
                {"sender": "component1", "value": 42},
                {"sender": "component2", "value": 43}
            ]
        }
        assert all_predecessors_executed(complex_component, inputs) is False

    def test_all_predecessors_executed_with_user_input(self, complex_component):
        inputs = {
            "regular": [{"sender": "component1", "value": 42}],
            "lazy_var": [
                {"sender": "component1", "value": 42},
                {"sender": None, "value": 43}  # User input shouldn't affect predecessor execution
            ],
            "greedy_var": [
                {"sender": "component1", "value": 42},
                {"sender": "component2", "value": 43},
                {"sender": "component3", "value": 44}
            ]
        }
        assert all_predecessors_executed(complex_component, inputs) is False

class TestLazyVariadicResolution:
    def test_lazy_variadic_sockets_all_resolved(self, complex_component):
        inputs = {
            "lazy_var": [
                {"sender": "component1", "value": 42},
                {"sender": "component2", "value": 43}
            ]
        }
        assert are_all_lazy_variadic_sockets_resolved(complex_component, inputs) is True

    def test_lazy_variadic_sockets_partially_resolved(self, complex_component):
        inputs = {
            "lazy_var": [{"sender": "component1", "value": 42}]  # Missing component2
        }
        assert are_all_lazy_variadic_sockets_resolved(complex_component, inputs) is False

    def test_lazy_variadic_sockets_with_no_inputs(self, complex_component):
        inputs = {}
        assert are_all_lazy_variadic_sockets_resolved(complex_component, inputs) is False

    def test_lazy_variadic_sockets_with_predecessors_executed(self, complex_component):
        inputs = {
            "lazy_var": [
                {"sender": "component1", "value": _NO_OUTPUT_PRODUCED},
                {"sender": "component2", "value": _NO_OUTPUT_PRODUCED}
            ]
        }
        # All predecessors executed but produced no output
        assert are_all_lazy_variadic_sockets_resolved(complex_component, inputs) is True

class TestGreedySocketReadiness:
    def test_greedy_socket_ready(self, complex_component):
        inputs = {
            "greedy_var": [{"sender": "component1", "value": 42}]
        }
        assert is_any_greedy_socket_ready(complex_component, inputs) is True

    def test_greedy_socket_multiple_inputs_ready(self, complex_component):
        inputs = {
            "greedy_var": [
                {"sender": "component1", "value": 42},
                {"sender": "component2", "value": 43}
            ]
        }
        assert is_any_greedy_socket_ready(complex_component, inputs) is True

    def test_greedy_socket_not_ready(self, complex_component):
        inputs = {
            "greedy_var": [{"sender": "component1", "value": _NO_OUTPUT_PRODUCED}]
        }
        assert is_any_greedy_socket_ready(complex_component, inputs) is False

    def test_greedy_socket_no_inputs(self, complex_component):
        inputs = {}
        assert is_any_greedy_socket_ready(complex_component, inputs) is False

    def test_greedy_socket_with_user_input(self, complex_component):
        inputs = {
            "greedy_var": [{"sender": None, "value": 42}]
        }
        assert is_any_greedy_socket_ready(complex_component, inputs) is True