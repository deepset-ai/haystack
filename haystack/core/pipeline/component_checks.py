# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from haystack.core.component.types import InputSocket, _empty

_NO_OUTPUT_PRODUCED = _empty


def can_component_run(component: dict, inputs: dict) -> bool:
    """
    Checks if the component can run, given the current state of its inputs.

    A component needs to pass two gates so that it is ready to run:
    1. It has received all mandatory inputs.
    2. It has received a trigger.
    :param component: Component metadata and the component instance.
    :param inputs: Inputs for the component.
    """
    received_all_mandatory_inputs = are_all_sockets_ready(component, inputs, only_check_mandatory=True)
    received_trigger = has_any_trigger(component, inputs)

    return received_all_mandatory_inputs and received_trigger


def has_any_trigger(component: dict, inputs: dict) -> bool:
    """
    Checks if a component was triggered to execute.

    There are 3 triggers:
    1. A predecessor provided input to the component.
    2. Input to the component was provided from outside the pipeline (e.g. user input).
    3. The component does not receive input from any other components in the pipeline and `Pipeline.run` was called.

    A trigger can only cause a component to execute ONCE because:
    1. Components consume inputs from predecessors before execution (they are deleted).
    2. Inputs from outside the pipeline can only trigger a component when it is executed for the first time.
    3. `Pipeline.run` can only trigger a component when it is executed for the first time.

    :param component: Component metadata and the component instance.
    :param inputs: Inputs for the component.
    """
    trigger_from_predecessor = any_predecessors_provided_input(component, inputs)
    trigger_from_user = has_user_input(inputs) and component["visits"] == 0
    trigger_without_inputs = can_not_receive_inputs_from_pipeline(component) and component["visits"] == 0

    return trigger_from_predecessor or trigger_from_user or trigger_without_inputs


def are_all_sockets_ready(component: dict, inputs: dict, only_check_mandatory: bool = False) -> bool:
    """
    Checks if all sockets of a component have enough inputs for the component to execute.

    :param component: Component metadata and the component instance.
    :param inputs: Inputs for the component.
    :param only_check_mandatory: If only mandatory sockets should be checked.
    """
    filled_sockets = set()
    expected_sockets = set()
    if only_check_mandatory:
        sockets_to_check = {
            socket_name: socket for socket_name, socket in component["input_sockets"].items() if socket.is_mandatory
        }
    else:
        sockets_to_check = {
            socket_name: socket
            for socket_name, socket in component["input_sockets"].items()
            if socket.is_mandatory or len(socket.senders)
        }

    for socket_name, socket in sockets_to_check.items():
        socket_inputs = inputs.get(socket_name, [])
        expected_sockets.add(socket_name)

        # Check if socket has all required inputs or is a lazy variadic socket with any input
        if has_socket_received_all_inputs(socket, socket_inputs) or (
            socket.is_lazy_variadic and any_socket_input_received(socket_inputs)
        ):
            filled_sockets.add(socket_name)

    return filled_sockets == expected_sockets


def any_predecessors_provided_input(component: dict, inputs: dict) -> bool:
    """
    Checks if a component received inputs from any predecessors.

    :param component: Component metadata and the component instance.
    :param inputs: Inputs for the component.
    """
    return any(
        any_socket_value_from_predecessor_received(inputs.get(socket_name, []))
        for socket_name in component["input_sockets"].keys()
    )


def any_socket_value_from_predecessor_received(socket_inputs: list[dict[str, Any]]) -> bool:
    """
    Checks if a component socket received input from any predecessors.

    :param socket_inputs: Inputs for the component's socket.
    """
    # When sender is None, the input was provided from outside the pipeline.
    return any(inp["value"] is not _NO_OUTPUT_PRODUCED and inp["sender"] is not None for inp in socket_inputs)


def has_user_input(inputs: dict) -> bool:
    """
    Checks if a component has received input from outside the pipeline (e.g. user input).

    :param inputs: Inputs for the component.
    """
    return any(inp for socket in inputs.values() for inp in socket if inp["sender"] is None)


def can_not_receive_inputs_from_pipeline(component: dict) -> bool:
    """
    Checks if a component can not receive inputs from any other components in the pipeline.

    :param: Component metadata and the component instance.
    """
    return all(len(sock.senders) == 0 for sock in component["input_sockets"].values())


def all_socket_predecessors_executed(socket: InputSocket, socket_inputs: list[dict]) -> bool:
    """
    Checks if all components connecting to an InputSocket have executed.

    :param: The InputSocket of a component.
    :param: socket_inputs: Inputs for the socket.
    """
    expected_senders = set(socket.senders)
    executed_senders = {inp["sender"] for inp in socket_inputs if inp["sender"] is not None}

    return expected_senders == executed_senders


def any_socket_input_received(socket_inputs: list[dict]) -> bool:
    """
    Checks if a socket has received any input from any other components in the pipeline or from outside the pipeline.

    :param socket_inputs: Inputs for the socket.
    """
    return any(inp["value"] is not _NO_OUTPUT_PRODUCED for inp in socket_inputs)


def has_lazy_variadic_socket_received_all_inputs(socket: InputSocket, socket_inputs: list[dict]) -> bool:
    """
    Checks if a lazy variadic socket has received all expected inputs from other components in the pipeline.

    :param socket: The InputSocket of a component.
    :param socket_inputs: Inputs for the socket.
    """
    expected_senders = set(socket.senders)
    actual_senders = {
        sock["sender"]
        for sock in socket_inputs
        if sock["value"] is not _NO_OUTPUT_PRODUCED and sock["sender"] is not None
    }
    return expected_senders == actual_senders


def has_socket_received_all_inputs(socket: InputSocket, socket_inputs: list[dict]) -> bool:
    """
    Checks if a socket has received all expected inputs.

    :param socket: The InputSocket of a component.
    :param socket_inputs: Inputs for the socket.
    """
    # No inputs received for the socket, it is not filled.
    if len(socket_inputs) == 0:
        return False

    # The socket is greedy variadic and at least one input was produced, it is complete.
    if (
        socket.is_variadic
        and socket.is_greedy
        and any(sock["value"] is not _NO_OUTPUT_PRODUCED for sock in socket_inputs)
    ):
        return True

    # The socket is lazy variadic and all expected inputs were produced.
    if socket.is_lazy_variadic and has_lazy_variadic_socket_received_all_inputs(socket, socket_inputs):
        return True

    # The socket is not variadic and the only expected input is complete.
    return not socket.is_variadic and socket_inputs[0]["value"] is not _NO_OUTPUT_PRODUCED


def all_predecessors_executed(component: dict, inputs: dict) -> bool:
    """
    Checks if all predecessors of a component have executed.

    :param component: Component metadata and the component instance.
    :param inputs: Inputs for the component.
    """
    return all(
        all_socket_predecessors_executed(socket, inputs.get(socket_name, []))
        for socket_name, socket in component["input_sockets"].items()
    )


def are_all_lazy_variadic_sockets_resolved(component: dict, inputs: dict) -> bool:
    """
    Checks if the final state for all lazy variadic sockets of a component is resolved.

    :param component: Component metadata and the component instance.
    :param inputs: Inputs for the component.
    """
    for socket_name, socket in component["input_sockets"].items():
        if socket.is_lazy_variadic:
            socket_inputs = inputs.get(socket_name, [])

            # Checks if a lazy variadic socket is ready to run.
            # A socket is ready if either:
            # - it has received all expected inputs, or
            # - all its predecessors have executed
            # If none of the conditions are met, the socket is not ready to run and we defer the component.
            if not (
                has_lazy_variadic_socket_received_all_inputs(socket, socket_inputs)
                or all_socket_predecessors_executed(socket, socket_inputs)
            ):
                return False

    return True


def is_permanently_blocked(component: dict, inputs: dict) -> bool:
    """
    Checks if a component is permanently blocked and will never be able to run.

    A component is permanently blocked when:
    1. It cannot run (``can_component_run`` returns False), AND
    2. At least one mandatory socket will *never* be satisfied because every sender for that socket
       has already executed and none produced a real value (all sent `_NO_OUTPUT_PRODUCED`).

    :param component: Component metadata and the component instance.
    :param inputs: Inputs for the component.
    """
    if can_component_run(component, inputs):
        return False

    for socket_name, socket in component["input_sockets"].items():
        if not socket.is_mandatory or not socket.senders:
            continue

        socket_inputs = inputs.get(socket_name, [])
        if not socket_inputs:
            continue

        # Determine which senders have already executed (regardless of what they produced).
        executed_senders = {inp["sender"] for inp in socket_inputs if inp["sender"] is not None}
        all_senders = set(socket.senders)

        if not all_senders.issubset(executed_senders):
            # Not all senders have executed yet — we can't call this permanently blocked.
            continue

        # All senders executed.  Identify those that only produced _NO_OUTPUT_PRODUCED.
        if _is_socket_permanently_unsatisfied(socket, socket_inputs):
            return True

    return False


def _is_socket_permanently_unsatisfied(socket: InputSocket, socket_inputs: list[dict]) -> bool:
    """
    Given that every sender for *socket* has already executed, checks whether the socket will never be satisfied.

    For a non-variadic socket this is the case when the single input is `_NO_OUTPUT_PRODUCED`.
    For a variadic socket this is the case when no sender delivered a real value.
    """
    if not socket.is_variadic and not socket.is_lazy_variadic:
        return socket_inputs[0]["value"] is _NO_OUTPUT_PRODUCED

    # Variadic (greedy or lazy): permanently unsatisfied if no sender provided a real value.
    return not any(inp["value"] is not _NO_OUTPUT_PRODUCED for inp in socket_inputs if inp["sender"] is not None)


def is_any_greedy_socket_ready(component: dict, inputs: dict) -> bool:
    """
    Checks if the component has any greedy socket that is ready to run.

    :param component: Component metadata and the component instance.
    :param inputs: Inputs for the component.
    """
    for socket_name, socket in component["input_sockets"].items():
        if socket.is_greedy and has_socket_received_all_inputs(socket, inputs.get(socket_name, [])):
            return True

    return False
